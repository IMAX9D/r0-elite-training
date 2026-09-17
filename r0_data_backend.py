#!/usr/bin/env python3
"""CPU data backend: one decode per shard, four consumptions, bounded shared-memory slots.

DESIGN CONSTRAINTS (from the worklist):
  * a shard is decoded ONCE; its 16 ticks are delivered as four T=4 chunks from a CPU cache. The
    1,329 shards/s figure was an artefact of re-decoding, not a property of T=4.
  * three boundaries stay separate: read/decode (whole shard) | batch/transfer (T=4 chunk) |
    training (unchanged TBPTT, loss, backward, optimizer).
  * workers write directly into preallocated reusable shared slots; only DESCRIPTORS cross the
    process boundary.
  * the cache holds inputs, labels and metadata only -- no model outputs, no graph tensors, no
    LSTM hidden/cell.
  * one shard is owned by ONE worker for all four chunks (shard affinity), so the four slice
    requests land in the process that already holds the decoded cache.
  * the next shard is prefetched while the current one is still consumed: 2048 lanes crossing a
    shard boundary need ~3.27 s of decode against ~1.37 s of consumption at 6000 decisions/s.

SLOT LAYOUT (shared memory, fixed size, no allocation during training):

    header : int64[3]                  magic, generation, n_entries
    index  : int32[n_entries*2]        (byte_offset, byte_length) per entry
    shape  : int32[n_entries*MAX_DIM]  ndim then dims   <- required: fields are variable length
    data   : raw bytes

An entry is one (lane_position, tick, field) triple in a fixed tree order. A small pickled template
travels as the descriptor; the bytes never cross the process boundary.
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/root')
from training_r0.il_dataset import load_sequence

MAGIC = 0x52304442  # 'R0DB'
MAX_DIM = 6


# ---------------------------------------------------------------- tree flattening

def flatten_tree(obj):
    """(tensors, template). The template is tiny and picklable; the tensors carry the bytes."""
    tensors: list = []

    def rec(node):
        if isinstance(node, torch.Tensor):
            tensors.append(node)
            return len(tensors) - 1
        if dataclasses.is_dataclass(node):
            return ('dc', type(node).__module__, type(node).__qualname__,
                    [(f.name, rec(getattr(node, f.name))) for f in dataclasses.fields(node)])
        if isinstance(node, tuple):
            return ('tuple', [rec(v) for v in node])
        if isinstance(node, list):
            return ('list', [rec(v) for v in node])
        return ('leaf', node)

    return tensors, rec(obj)


def resolve_type(module: str, qualname: str):
    import importlib
    obj = importlib.import_module(module)
    for part in qualname.split('.'):
        obj = getattr(obj, part)
    return obj


def unflatten_tree(tensors, template, resolve=resolve_type):
    if isinstance(template, int):
        return tensors[template]
    kind = template[0]
    if kind == 'dc':
        cls = resolve(template[1], template[2])
        return cls(**{name: unflatten_tree(tensors, sub, resolve) for name, sub in template[3]})
    if kind == 'tuple':
        return tuple(unflatten_tree(tensors, s, resolve) for s in template[1])
    if kind == 'list':
        return [unflatten_tree(tensors, s, resolve) for s in template[1]]
    return template[1]


def to_np_dtype(torch_dtype):
    return torch.empty(0, dtype=torch_dtype).numpy().dtype


# ---------------------------------------------------------------- slot

class SlotLayout:
    """Geometry of one slot: B * T * n_fields entries, plus index and shape tables."""

    def __init__(self, batch: int, tbptt: int, n_fields: int, bytes_per_entry: int):
        self.batch = batch
        self.tbptt = tbptt
        self.n_fields = n_fields
        self.n_entries = batch * tbptt * n_fields
        self.header_bytes = 3 * 8
        self.index_bytes = self.n_entries * 2 * 4
        self.shape_offset = self.header_bytes + self.index_bytes
        self.shape_bytes = self.n_entries * MAX_DIM * 4
        self.data_offset = self.shape_offset + self.shape_bytes
        # 1.6x headroom: padding to the batch maximum and late-game entity growth both push the
        # payload above the tick-0 average. Exceeding it raises instead of truncating silently.
        # bytes_per_entry is ONE (lane, tick)'s worth across all fields, so the payload scales with
        # batch*tbptt, NOT with n_entries (which already includes n_fields). Multiplying by
        # n_entries overstated the slot by n_fields (90x) and overran int32 offsets at B=64.
        self.data_bytes = int(bytes_per_entry * self.batch * self.tbptt * 1.6)
        self.total_bytes = self.data_offset + self.data_bytes

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ('batch', 'tbptt', 'n_fields', 'n_entries', 'header_bytes', 'index_bytes',
                 'shape_offset', 'shape_bytes', 'data_offset', 'data_bytes', 'total_bytes')}

    @classmethod
    def from_dict(cls, payload):
        obj = cls.__new__(cls)
        for key, value in payload.items():
            setattr(obj, key, value)
        return obj


def reset_slot(buffer, layout: SlotLayout, generation: int):
    """Clear index and shape tables ONCE, in the parent, before dispatching to workers.

    Workers must NOT clear: several of them write disjoint entries of the SAME slot, so a worker
    that zeroed the tables would wipe entries another worker had just written. That race produced
    "entry 0 was never written" even though every worker reported success.
    """
    header = np.frombuffer(buffer[:layout.header_bytes], dtype=np.int64)
    header[0] = MAGIC
    header[1] = generation
    header[2] = layout.n_entries
    index = np.frombuffer(buffer[layout.header_bytes:layout.shape_offset], dtype=np.int32)
    shape_table = np.frombuffer(buffer[layout.shape_offset:layout.data_offset], dtype=np.int32)
    index[:] = 0
    shape_table[:] = 0
    del header, index, shape_table


def write_slot(buffer, layout: SlotLayout, generation: int, entries, shapes,
               region_start: int, region_end: int):
    """entries: [(entry_index, uint8 blob)]; shapes: {entry_index: tuple}.

    Writes only this worker's own entries, and only inside [region_start, region_end).

    THE REGION IS NOT OPTIONAL: several workers fill disjoint entries of the SAME slot, and if each
    started its data cursor at layout.data_offset they would overwrite one another. That is exactly
    what happened -- one lane's entity_features came back off by 5.6e-05 because another worker's
    payload had landed on top of it, while the index table still pointed at the right offsets.
    """
    index = np.frombuffer(buffer[layout.header_bytes:layout.shape_offset], dtype=np.int32)
    shape_table = np.frombuffer(buffer[layout.shape_offset:layout.data_offset], dtype=np.int32)
    cursor = region_start
    written = 0
    for entry_index, blob in entries:
        size = blob.nbytes
        if cursor + size > region_end:
            raise BufferError(
                f'worker region overflow: {cursor + size} > {region_end}; the per-worker share of '
                f'{layout.data_bytes} bytes is too small -- raise the byte budget')
        buffer[cursor:cursor + size] = blob
        index[entry_index * 2] = cursor
        index[entry_index * 2 + 1] = size
        shape = shapes[entry_index]
        base = entry_index * MAX_DIM
        shape_table[base] = len(shape)
        for k, dim in enumerate(shape):
            shape_table[base + 1 + k] = dim
        cursor += size
        written += size
    del index, shape_table
    return written


def read_slot(buffer, layout: SlotLayout, expected_generation: int):
    """-> list of (bytes, shape) in entry order. Copies out so no view pins the buffer."""
    header = np.frombuffer(buffer[:layout.header_bytes], dtype=np.int64)
    if int(header[0]) != MAGIC:
        raise ValueError('slot magic mismatch')
    if int(header[1]) != expected_generation:
        raise ValueError(f'stale generation: slot has {int(header[1])}, trainer expects '
                         f'{expected_generation} -- a late worker message must never be accepted')
    index = np.frombuffer(buffer[layout.header_bytes:layout.shape_offset], dtype=np.int32)
    shape_table = np.frombuffer(buffer[layout.shape_offset:layout.data_offset], dtype=np.int32)
    out = []
    for entry_index in range(layout.n_entries):
        start, length = int(index[entry_index * 2]), int(index[entry_index * 2 + 1])
        if length == 0:
            raise ValueError(f'entry {entry_index} was never written: the chunk is incomplete')
        blob = bytes(buffer[start:start + length])
        base = entry_index * MAX_DIM
        ndim = int(shape_table[base])
        shape = tuple(int(shape_table[base + 1 + k]) for k in range(ndim))
        out.append((blob, shape))
    del header, index, shape_table
    return out


# ---------------------------------------------------------------- worker

def worker_main(worker_index, n_workers, lane_specs, slot_names, layout_dict, command_q, ready_q,
                stop_event):
    """Owns its shards for their whole life: decodes once, serves four T=4 slices from cache."""
    from multiprocessing import shared_memory

    layout = SlotLayout.from_dict(layout_dict)
    slots = [shared_memory.SharedMemory(name=name) for name in slot_names]
    # Disjoint slice of the data area per worker: concurrent writers into one shared slot must not
    # share a cursor.
    share = layout.data_bytes // n_workers
    region_start = layout.data_offset + worker_index * share
    region_end = region_start + share
    cache: dict = {}
    decodes = 0
    hits = 0

    def get_sequence(shard_path, sha):
        nonlocal decodes, hits
        if shard_path not in cache:
            cache[shard_path] = load_sequence(Path(shard_path), expected_sha256=sha)
            decodes += 1
        else:
            hits += 1
        return cache[shard_path]

    ready_q.put(('worker_up', worker_index, len(lane_specs)))
    while not stop_event.is_set():
        try:
            command = command_q.get(timeout=0.5)
        except Exception:
            continue
        kind = command[0]
        if kind == 'stop':
            break
        if kind == 'fill':
            # (slot_id, generation, [(lane_pos, tick, shard_path, sha)], tick_base)
            _, slot_id, generation, requests, tick_base = command
            try:
                entries, shapes, labels = [], {}, []
                for lane_pos, tick, shard_path, sha in requests:
                    sequence = get_sequence(shard_path, sha)
                    obs_tensors, _ = flatten_tree(sequence.observations[tick])
                    act_tensors, _ = flatten_tree(sequence.actions[tick])
                    tensors = list(obs_tensors) + list(act_tensors)
                    if len(tensors) != layout.n_fields:
                        raise ValueError(f'field count {len(tensors)} != {layout.n_fields}')
                    row = (lane_pos * layout.tbptt) + (tick - tick_base)
                    # label_known lives on ILSequence, not on the observation: it must be
                    # reported explicitly, or the loss denominator silently becomes B*T.
                    labels.append((row, bool(sequence.label_known[tick])))
                    for field_index, tensor in enumerate(tensors):
                        entry = row * layout.n_fields + field_index
                        entries.append((entry,
                                        tensor.detach().contiguous().cpu().numpy()
                                        .view(np.uint8).ravel()))
                        shapes[entry] = tuple(tensor.shape)
                written = write_slot(slots[slot_id].buf, layout, generation, entries, shapes,
                                      region_start, region_end)
                ready_q.put(('filled', worker_index, slot_id, generation, written, decodes,
                             hits, labels))
            except Exception as error:
                import traceback
                ready_q.put(('error', worker_index,
                             f'{type(error).__name__}: {error} | {traceback.format_exc()[-400:]}'))
        elif kind == 'prefetch':
            _, shard_path, sha = command
            try:
                get_sequence(shard_path, sha)
                ready_q.put(('prefetched', worker_index, shard_path, decodes, hits))
            except Exception as error:
                ready_q.put(('error', worker_index, f'prefetch {type(error).__name__}: {error}'))
        elif kind == 'evict':
            _, shard_path = command
            cache.pop(shard_path, None)
        elif kind == 'stats':
            ready_q.put(('stats', worker_index, decodes, hits, len(cache)))
    ready_q.put(('worker_down', worker_index, decodes, hits))


# ---------------------------------------------------------------- backend

class DataBackend:
    """Schedules decodes and slot fills; hands the trainer per-lane CPU tensors.

    The trainer never sees a file path. It asks for a chunk and gets tensors back.
    """

    def __init__(self, lane_specs, layout: SlotLayout, n_workers=2, n_slots=2, verbose=True):
        self.lane_specs = lane_specs          # [(tag, [(shard_path, sha), ...])]
        self.layout = layout
        self.n_workers = n_workers
        self.n_slots = n_slots
        self.verbose = verbose
        self._affinity: dict = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        from multiprocessing import shared_memory
        self.shms = [shared_memory.SharedMemory(create=True, size=self.layout.total_bytes)
                     for _ in range(self.n_slots)]
        self.command_qs = [mp.Queue() for _ in range(self.n_workers)]
        self.ready_q = mp.Queue()
        self.stop_event = mp.Event()
        self.workers = []
        for index in range(self.n_workers):
            mine = [spec for position, spec in enumerate(self.lane_specs)
                    if position % self.n_workers == index]
            process = mp.Process(target=worker_main,
                                 args=(index, self.n_workers, mine, [s.name for s in self.shms],
                                       self.layout.to_dict(), self.command_qs[index],
                                       self.ready_q, self.stop_event),
                                 daemon=True)
            process.start()
            self.workers.append(process)
        up = 0
        deadline = time.time() + 180
        while up < self.n_workers and time.time() < deadline:
            message = self._next_message(timeout=2)
            if message is None:
                continue
            if message[0] == 'worker_up':
                up += 1
            elif message[0] == 'error':
                raise RuntimeError(f'worker failed at startup: {message[2]}')
        if up < self.n_workers:
            raise RuntimeError(f'only {up}/{self.n_workers} workers came up')
        if self.verbose:
            print(f'backend up: {self.n_workers} workers, {self.n_slots} slots, '
                  f'{self.layout.total_bytes/2**20:.1f} MiB/slot', flush=True)
        return self

    def stop(self):
        self.stop_event.set()
        for queue_ in self.command_qs:
            queue_.put(('stop',))
        for process in self.workers:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
        for shm in self.shms:
            try:
                shm.close()
            except BufferError:
                pass
            shm.unlink()

    def _next_message(self, timeout):
        try:
            return self.ready_q.get(timeout=timeout)
        except Exception:
            return None

    # -- affinity ----------------------------------------------------------
    def worker_for(self, shard_path):
        """Deterministic round-robin: the same shard always reaches the worker that cached it.

        `hash(path) % n_workers` is stable but clusters badly on small shard sets -- in the first
        process test all four shards landed on worker 1 while worker 0 idled. Assignment by
        first-seen order is equally stable per shard and spreads the load.
        """
        if shard_path not in self._affinity:
            self._affinity[shard_path] = len(self._affinity) % self.n_workers
        return self._affinity[shard_path]

    # -- data path ---------------------------------------------------------
    def prefetch(self, shard_path, sha):
        self.command_qs[self.worker_for(shard_path)].put(('prefetch', shard_path, sha))

    def fill(self, slot_id, generation, requests, tick_base):
        """requests: [(lane_pos, tick, shard_path, sha)].

        Returns (bytes_written, labels) where labels maps row -> bool, so the caller can supervise
        exactly the decisions the shard marks as known.
        """
        # Reset ONCE here, before any worker touches the slot: several workers write disjoint
        # entries of the same slot and must never clear each other's table rows.
        reset_slot(self.shms[slot_id].buf, self.layout, generation)
        per_worker = {}
        for request in requests:
            per_worker.setdefault(self.worker_for(request[2]), []).append(request)
        for index, group in per_worker.items():
            self.command_qs[index].put(('fill', slot_id, generation, group, tick_base))
        written, pending, labels = 0, len(per_worker), {}
        deadline = time.time() + 900
        while pending > 0:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f'fill timed out with {pending} worker(s) outstanding')
            message = self._next_message(timeout=min(5, remaining))
            if message is None:
                continue
            if message[0] == 'filled' and message[2] == slot_id and message[3] == generation:
                written += message[4]
                for row, value in message[7]:
                    labels[row] = value
                pending -= 1
            elif message[0] == 'error':
                raise RuntimeError(f'worker error: {message[2]}')
        return written, labels

    def read(self, slot_id, generation, dtypes):
        """-> list of one tensor-list per (lane, tick) row, rebuilt with shapes and dtypes."""
        entries = read_slot(self.shms[slot_id].buf, self.layout, generation)
        rows = self.layout.batch * self.layout.tbptt
        out = []
        for row in range(rows):
            tensors = []
            for field_index, dtype in enumerate(dtypes):
                blob, shape = entries[row * self.layout.n_fields + field_index]
                array = np.frombuffer(blob, dtype=to_np_dtype(dtype)).reshape(shape).copy()
                tensors.append(torch.from_numpy(array))
            out.append(tensors)
        return out

    def stats(self):
        for queue_ in self.command_qs:
            queue_.put(('stats',))
        collected = {}
        deadline = time.time() + 30
        while len(collected) < self.n_workers and time.time() < deadline:
            message = self._next_message(timeout=2)
            if message and message[0] == 'stats':
                collected[message[1]] = {'decodes': message[2], 'hits': message[3],
                                         'cached': message[4]}
        return collected
