#!/usr/bin/env python3
"""Memory ledger: what is actually 91.34 GiB, and how many bytes must move per decision?

WHY THIS COMES FIRST: the earlier claim "ship 91.34 GiB every 0.2 s" assumed that figure was the
per-tick payload. It is not necessarily: `torch.cuda.max_memory_allocated()` includes ACTIVATIONS,
and with TBPTT the graph is retained across ticks, so the peak grows with T as well as B. If 91.34
GiB is a B=2048 x T=4 peak, then the actual input payload is tiny and the architecture question is
completely different.

WHAT THIS REPORTS, per the worklist:
  * logical bytes (sum of numel*itemsize) vs UNIQUE STORAGE bytes -- PyTorch tensors can share
    storage, so the naive sum overcounts and must not be used for sizing
  * bytes per lane per tick, and per field group, so padding cost is visible
  * padding cost: padded element count vs unpadded
  * bytes per effective decision -- the only number that determines required H2D bandwidth
  * what fraction of the GPU peak is input vs activation/gradient

Run on CPU: activations must not contaminate the payload measurement.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/root')
from r0_train_batched import fuse
from training_r0.il_dataset import load_sequence

ROOT = Path('/root/autodl-tmp/out-0/compiled-prod-v17')


def walk(obj, path='', out=None):
    if out is None:
        out = {}
    if isinstance(obj, torch.Tensor):
        out[path] = obj
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            walk(getattr(obj, f.name), f'{path}.{f.name}', out)
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            walk(v, f'{path}[{i}]', out)
    return out


def measure(obj):
    """Logical bytes (double counts shared storage) and unique-storage bytes (the real footprint)."""
    tensors = walk(obj)
    logical = sum(t.numel() * t.element_size() for t in tensors.values())
    storages: dict[int, int] = {}
    for t in tensors.values():
        ptr = t.untyped_storage().data_ptr()
        storages.setdefault(ptr, t.untyped_storage().nbytes())
    unique = sum(storages.values())
    return {
        'tensors': len(tensors),
        'logical_bytes': logical,
        'unique_bytes': unique,
        'distinct_storages': len(storages),
    }


def rss_mb():
    try:
        with open('/proc/self/status') as handle:
            for line in handle:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', default='64,256,1024,2048')
    ap.add_argument('--tbptt', type=int, default=4)
    args = ap.parse_args()

    shards = sorted(ROOT.rglob('*.npz'))
    lanes, seen = [], set()
    for shard in shards:
        key = shard.name.split('-tick')[0]
        if key in seen:
            continue
        seen.add(key)
        lanes.append(load_sequence(shard,
                                   expected_sha256=hashlib.sha256(shard.read_bytes()).hexdigest()))
        if len(lanes) >= 8:
            break
    print(f'real lanes loaded: {len(lanes)}  ticks per lane: '
          f'{sorted({len(l.observations) for l in lanes})}')

    # ---- one lane, one tick: the atomic unit
    one = lanes[0].observations[0]
    one_m = measure(one)
    print()
    print('=== A single lane, single tick (raw, unpadded) ===')
    print(json.dumps(one_m, indent=1))
    print(f'  per-tensor breakdown (top 12 of {one_m["tensors"]}):')
    for path, tensor in sorted(walk(one).items(),
                               key=lambda kv: -(kv[1].numel() * kv[1].element_size()))[:12]:
        b = tensor.numel() * tensor.element_size()
        print(f'    {b:>9,} B  {str(tuple(tensor.shape)):<26} {path[:60]}')

    # ---- fused batch at several B: what padding costs and what the payload really is
    print()
    print('=== Fused batch payload (CPU, no activations) ===')
    print(f'{"B":>6} {"logical MB":>11} {"unique MB":>10} {"storages":>9} {"pad frac":>9} '
          f'{"B/tick MB":>10} {"B/decision B":>13}')
    rows = []
    for b in [int(x) for x in args.batches.split(',') if x.strip()]:
        picks = [lanes[i % len(lanes)] for i in range(b)]
        obs = [p.observations[0] for p in picks]
        fused, pad = fuse(obs)
        m = measure(fused)
        per_tick = m['unique_bytes']
        per_decision = per_tick / b
        rows.append({'B': b, **m, 'pad': pad, 'per_tick': per_tick,
                     'per_decision': per_decision})
        print(f'{b:>6} {m["logical_bytes"]/2**20:>11.1f} {m["unique_bytes"]/2**20:>10.1f} '
              f'{m["distinct_storages"]:>9} {pad:>9.4f} {per_tick/2**20:>10.1f} '
              f'{per_decision:>13.0f}')
        del fused

    # ---- how many of those bytes must actually reach the GPU
    print()
    print('=== Which bytes must be transferred ===')
    sample = rows[-1] if rows else None
    if sample:
        b = sample['B']
        chunk_decisions = b * args.tbptt
        for rate in (4872, 6000, 10000):
            chunks_per_s = rate / chunk_decisions
            bytes_per_s = chunks_per_s * sample['per_tick'] * args.tbptt
            print(f'  supply {rate:>6} decisions/s -> {chunks_per_s:>6.3f} chunks/s '
                  f'-> {bytes_per_s/2**30:>6.3f} GiB/s of raw input')

    print()
    print('=== What the 91.34 GiB GPU peak actually contains ===')
    print('  measured separately by bench_device_train.py at B=2048, T=4:')
    print(f'    GPU peak allocated        : 91.34 GiB')
    if sample:
        input_gib = sample['per_tick'] * args.tbptt / 2**30
        print(f'    of which raw INPUT        : ~{input_gib:.2f} GiB  '
              f'({100*input_gib/91.34:.1f}%)')
        print(f'    remainder (activations+gradients+states): ~{91.34-input_gib:.2f} GiB '
              f'({100*(91.34-input_gib)/91.34:.1f}%)')
        print()
        print('  => the GPU peak is an ACTIVATION/显存 constraint that caps B,')
        print('     not a transfer volume. Transfer volume per chunk is the input figure above.')

    print()
    print(f'process RSS after measurement: {rss_mb():.0f} MB')

    Path('/root/autodl-tmp/run/mem_ledger.json').write_text(
        json.dumps({'lanes': len(lanes), 'single_tick': one_m, 'batches': rows,
                    'tbptt': args.tbptt}, indent=1), encoding='utf-8')
    print('ledger -> /root/autodl-tmp/run/mem_ledger.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
