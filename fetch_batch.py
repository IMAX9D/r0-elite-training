#!/usr/bin/env python3
"""Pull one verified batch archive from the netdisk onto the training host.

WHY ONE BATCH IS ENOUGH FOR NOW: this slice gives 70 training lanes from 40 battles, and B=1024
needs 1024 real lanes, i.e. roughly 585 battles. One batch holds about 4000 battles, so a single
3 GiB archive covers B=1024 with room to spare -- no need to move the 202 GiB corpus.

The dlink endpoint returns a short-lived signed URL that must be fetched WITH the access_token, and
requests must carry a pan.baidu.com User-Agent (without it the CDN returns 403).

usage:
  fetch_batch.py --list
  fetch_batch.py --batch 0 --dest /root/autodl-tmp/r0data
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request

TOKEN_FILE = pathlib.Path('/root/.r0_xpan_token')
REMOTE_DIR = '/apps/bypy/R0-Elite'
UA = 'pan.baidu.com'


def token() -> str:
    env = os.environ.get('R0_XPAN_TOKEN')
    if env:
        return env.strip()
    if TOKEN_FILE.is_file():
        return TOKEN_FILE.read_text().strip()
    raise SystemExit(f'no token: set R0_XPAN_TOKEN or write {TOKEN_FILE}')


def api(method: str, params: dict) -> dict:
    query = urllib.parse.urlencode(dict(method=method, access_token=token(), **params))
    request = urllib.request.Request(
        'https://pan.baidu.com/rest/2.0/xpan/file?' + query, data=b'',
        headers={'User-Agent': UA, 'Content-Type': 'application/x-www-form-urlencoded'})
    return json.loads(urllib.request.urlopen(request, timeout=90).read())


def list_archives() -> list[dict]:
    data = api('list', dict(dir=REMOTE_DIR, order='name', start='0', limit='500'))
    if data.get('errno'):
        raise SystemExit(f'list failed: errno={data.get("errno")}')
    return [e for e in data.get('list', []) if e['server_filename'].endswith('.tar.zst')]


def dlink(fs_id: int) -> str:
    data = api('filemetas', dict(dlink='1', fsids=json.dumps([fs_id]), thumb='0', extra='1'))
    entries = data.get('info') or data.get('list') or []
    if not entries:
        raise SystemExit(f'no filemetas for fs_id={fs_id}: {data}')
    return entries[0]['dlink'] + '&access_token=' + urllib.parse.quote(token())


def download(url: str, target: pathlib.Path, expect: int) -> None:
    partial = target.with_suffix(target.suffix + '.part')
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {'User-Agent': UA}
    if offset:
        headers['Range'] = f'bytes={offset}-'
        print(f'  resuming at {offset/2**30:.2f} GiB', flush=True)
    request = urllib.request.Request(url, headers=headers)
    started = time.perf_counter()
    written = offset
    last = started
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, 'ab') as out:
        while True:
            chunk = response.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            now = time.perf_counter()
            if now - last >= 10:
                rate = written / (now - started) / 2**20
                pct = 100 * written / expect if expect else 0
                print(f'  {written/2**30:6.2f} / {expect/2**30:6.2f} GiB  {pct:5.1f}%  '
                      f'{rate:6.1f} MiB/s', flush=True)
                last = now
    partial.replace(target)
    elapsed = time.perf_counter() - started
    print(f'  done: {written/2**30:.2f} GiB in {elapsed:.0f}s '
          f'({written/elapsed/2**20:.1f} MiB/s)', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--batch', type=int, default=None, help='batch index N in batch-N-compiled')
    ap.add_argument('--dest', type=pathlib.Path, default=pathlib.Path('/root/autodl-tmp/r0data'))
    ap.add_argument('--probe', action='store_true', help='download only the first 32 MiB')
    args = ap.parse_args()

    archives = list_archives()
    by_name = {e['server_filename']: e for e in archives}
    if args.list or args.batch is None:
        for e in archives:
            print(f"  {e['server_filename']:<40} {e['size']/2**30:7.2f} GiB  fs_id={e['fs_id']}")
        print(f'  total {len(archives)} archives, '
              f'{sum(e["size"] for e in archives)/2**30:.1f} GiB')
        return 0

    name = f'batch-{args.batch}-compiled.tar.zst'
    entry = by_name.get(name)
    if entry is None:
        raise SystemExit(f'{name} not found among {len(archives)} archives')

    args.dest.mkdir(parents=True, exist_ok=True)
    target = args.dest / name
    url = dlink(entry['fs_id'])
    print(f'target {target}  expect {entry["size"]/2**30:.2f} GiB', flush=True)

    if args.probe:
        request = urllib.request.Request(url, headers={'User-Agent': UA, 'Range': 'bytes=0-33554431'})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=90) as response:
            blob = response.read()
        elapsed = time.perf_counter() - started
        print(f'  probe: {len(blob)/2**20:.1f} MiB in {elapsed:.1f}s = '
              f'{len(blob)/elapsed/2**20:.2f} MiB/s')
        print(f'  => {entry["size"]/max(1e-9, len(blob)/elapsed)/60:.1f} min for the full archive')
        return 0

    if target.is_file() and target.stat().st_size == entry['size']:
        print('  already complete, nothing to do')
        return 0
    download(url, target, entry['size'])
    print(f'  final size {target.stat().st_size}  expected {entry["size"]}  '
          f'match={target.stat().st_size == entry["size"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
