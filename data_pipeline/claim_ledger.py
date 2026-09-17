"""Atomic work claiming for the rolling pipeline.

Why this exists: with a static split (`records[i::shard_count]`) a producer that
finishes early exits and idles while a few slow producers finish the batch tail.
Measured cost: at 1997/2000 completed only 2 of 96 producers were still working and
load had fallen from ~230 to 12-50.

With a claim ledger every producer keeps pulling the next unclaimed battle until the
queue is empty, so the tail is bounded by ONE battle instead of by the slowest
producer's whole share.

The ledger lives on the NVMe (not tmpfs): a state file on tmpfs cannot be used to
recover after a reboot, because it is lost with the data it describes.

Concurrency: SQLite in WAL mode with BEGIN IMMEDIATE gives an atomic
select-and-update across processes. Claims carry a worker id and a lease timestamp,
so a crashed producer's claims can be recovered instead of being lost forever -
without re-running work whose artifacts are already complete.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

PENDING = 'pending'
RUNNING = 'running'
USABLE = 'usable'
QUARANTINED = 'quarantined'
DONE_STATES = (USABLE, QUARANTINED)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=60.0, isolation_level=None)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=NORMAL')
    connection.execute('PRAGMA busy_timeout=60000')
    return connection


def init_ledger(path: Path, records: list, *, reset: bool = False) -> dict:
    """Create the queue from the selection. Safe to call again: already-known tags are
    left untouched so a restart resumes instead of redoing finished work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(path)
    try:
        connection.execute(
            'CREATE TABLE IF NOT EXISTS queue ('
            ' tag TEXT PRIMARY KEY,'
            ' source_file TEXT,'
            ' state TEXT NOT NULL,'
            ' worker TEXT,'
            ' claimed_at REAL,'
            ' finished_at REAL,'
            ' detail TEXT)')
        connection.execute('CREATE INDEX IF NOT EXISTS queue_state ON queue(state)')
        if reset:
            connection.execute('DELETE FROM queue')
        before = connection.execute('SELECT COUNT(*) FROM queue').fetchone()[0]
        connection.execute('BEGIN IMMEDIATE')
        connection.executemany(
            'INSERT OR IGNORE INTO queue(tag, source_file, state) VALUES(?,?,?)',
            [(record['battle_tag'], record.get('source_file', ''), PENDING) for record in records])
        connection.execute('COMMIT')
        after = connection.execute('SELECT COUNT(*) FROM queue').fetchone()[0]
        counts = dict(connection.execute('SELECT state, COUNT(*) FROM queue GROUP BY state'))
        return dict(path=str(path), submitted=len(records), added=after - before,
                    total=after, states=counts)
    finally:
        connection.close()


def claim_next(path: Path, worker: str, *, lease_seconds: float = 1800.0) -> dict | None:
    """Atomically take one battle. Returns None when nothing is left to claim.

    A claim whose lease expired is handed to the next caller: the producer that held it
    is presumed dead. Its artifacts are NOT deleted - the caller checks for a complete
    result first and reuses it rather than recomputing."""
    path = Path(path)
    connection = _connect(path)
    try:
        now = time.time()
        connection.execute('BEGIN IMMEDIATE')
        row = connection.execute(
            'SELECT tag, source_file FROM queue'
            ' WHERE state = ? OR (state = ? AND claimed_at < ?)'
            ' ORDER BY rowid LIMIT 1',
            (PENDING, RUNNING, now - lease_seconds)).fetchone()
        if row is None:
            connection.execute('COMMIT')
            return None
        connection.execute(
            'UPDATE queue SET state = ?, worker = ?, claimed_at = ? WHERE tag = ?',
            (RUNNING, worker, now, row[0]))
        connection.execute('COMMIT')
        return dict(tag=row[0], source_file=row[1], worker=worker, claimed_at=now)
    finally:
        connection.close()


def finish(path: Path, tag: str, state: str, detail: dict | None = None) -> None:
    if state not in DONE_STATES:
        raise ValueError(f'finish() state must be one of {DONE_STATES}, got {state!r}')
    connection = _connect(Path(path))
    try:
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('UPDATE queue SET state = ?, finished_at = ?, detail = ? WHERE tag = ?',
                           (state, time.time(), json.dumps(detail or {}, ensure_ascii=False)[:2000], tag))
        connection.execute('COMMIT')
    finally:
        connection.close()


def stats(path: Path) -> dict:
    connection = _connect(Path(path))
    try:
        counts = dict(connection.execute('SELECT state, COUNT(*) FROM queue GROUP BY state'))
        total = sum(counts.values())
        done = sum(counts.get(name, 0) for name in DONE_STATES)
        return dict(total=total, states=counts, done=done, remaining=total - done,
                    progress=round(done / total, 4) if total else None)
    finally:
        connection.close()


def recovered_artifacts(path: Path, tag: str) -> bool:
    """True when a previously-claimed battle already has a complete result recorded, so
    a recovering producer can skip the work instead of re-running it."""
    connection = _connect(Path(path))
    try:
        row = connection.execute('SELECT state, detail FROM queue WHERE tag = ?', (tag,)).fetchone()
        return bool(row and row[0] in DONE_STATES and row[1])
    finally:
        connection.close()
