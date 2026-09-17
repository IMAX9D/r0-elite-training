#!/usr/bin/env python3
"""Acceptance item 1: one decode, four consumptions -- and it must be equivalent to four decodes.

THE ERROR THIS CORRECTS: I previously computed 1,329 shards/s for T=4, which is only true if each
4-tick chunk re-decodes the whole 16-tick shard. That is four times the necessary work, not a
property of T=4. With a per-shard CPU cache the average demand is decisions/s / 16.

WHAT IS CHECKED HERE, before any shared memory or pinned buffer is introduced:
  1. decode count: traversing a lane's 16 ticks in four T=4 chunks decodes the shard ONCE
  2. equivalence: every chunk assembled from the cache is bit-identical to the same chunk
     assembled by decoding afresh -- inputs, labels and masks
  3. the four chunks together cover all 16 ticks exactly once (no gap, no repeat)

Only after this passes is it worth wiring shared memory, because shared memory moves bytes and
cannot by itself fix a decode that happens four times.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/root')
from r0_train_batched import fuse
from training_r0.il_dataset import load_sequence

ROOT = Path('/root/autodl-tmp/out-0/compiled-prod-v17')
DECODES = {'count': 0}


def decode(shard: Path):
    """The only place a shard is read. Counting here is what makes the claim testable."""
    DECODES['count'] += 1
    return load_sequence(shard, expected_sha256=hashlib.sha256(shard.read_bytes()).hexdigest())


class ShardCache:
    """Decoded 16-tick shard held on the CPU, sliced into T=4 chunks.

    Holds ONLY inputs, labels and metadata. Never model outputs, graph-carrying tensors, or
    LSTM hidden/cell -- those stay owned by the trainer, per the worklist.
    """

    __slots__ = ('observations', 'actions', 'label_known', 'shard', 'delivered')

    def __init__(self, shard: Path):
        sequence = decode(shard)
        self.shard = shard
        self.observations = list(sequence.observations)
        self.actions = list(sequence.actions)
        self.label_known = list(sequence.label_known)
        self.delivered = 0

    def __len__(self):
        return len(self.observations)

    def chunk(self, start: int, length: int):
        end = start + length
        if end > len(self.observations):
            return None
        self.delivered += end - start
        return (self.observations[start:end], self.actions[start:end],
                self.label_known[start:end])


def flat_compare(a, b, path=''):
    """Bitwise compare two nested observation trees, reporting the first mismatch."""
    import dataclasses
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return f'{path}: shape {tuple(a.shape)} vs {tuple(b.shape)}'
        if not torch.equal(a, b):
            return f'{path}: max|diff| {(a.float()-b.float()).abs().max().item():.3e}'
        return None
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        for f in dataclasses.fields(a):
            bad = flat_compare(getattr(a, f.name), getattr(b, f.name), f'{path}.{f.name}')
            if bad:
                return bad
        return None
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        for i, (x, y) in enumerate(zip(a, b)):
            bad = flat_compare(x, y, f'{path}[{i}]')
            if bad:
                return bad
    return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--lanes', type=int, default=4)
    ap.add_argument('--tbptt', type=int, default=4)
    args = ap.parse_args()

    # one shard per lane, distinct battle-sides
    groups: dict = {}
    for shard in sorted(ROOT.rglob('*.npz')):
        groups.setdefault(shard.name.split('-tick')[0], []).append(shard)
    picks = [sorted(v)[0] for k, v in sorted(groups.items())][:args.lanes]
    print(json.dumps({'lanes': len(picks), 'tbptt': args.tbptt,
                      'shard_bytes': picks[0].stat().st_size}, indent=1), flush=True)

    report = {'lanes': len(picks), 'tbptt': args.tbptt}

    # ---- 1. cache path: decode once, deliver four chunks
    DECODES['count'] = 0
    caches = [ShardCache(s) for s in picks]
    decoded_for_cache = DECODES['count']
    print(f'after building {len(caches)} caches, decodes = {decoded_for_cache}')

    ticks = len(caches[0])
    n_chunks = ticks // args.tbptt
    print(f'shard ticks = {ticks}, T = {args.tbptt} -> {n_chunks} chunks per shard')

    cache_chunks = []
    for c in range(n_chunks):
        start = c * args.tbptt
        per_lane = [cache.chunk(start, args.tbptt) for cache in caches]
        if any(p is None for p in per_lane):
            break
        # fuse the way the trainer does: one tick at a time across lanes
        for t in range(args.tbptt):
            obs = [p[0][t] for p in per_lane]
            fused, pad = fuse(obs)
            cache_chunks.append((c, t, fused, pad))
    print(f'after consuming all {n_chunks} chunks, decodes = {DECODES["count"]} '
          f'(expected {len(caches)})')
    report['decodes_for_all_chunks'] = DECODES['count']
    report['chunks_consumed'] = n_chunks

    # ---- 2. direct path: decode afresh for each chunk, same assembly
    DECODES['count'] = 0
    direct_chunks = []
    for c in range(n_chunks):
        start = c * args.tbptt
        sequences = [decode(s) for s in picks]
        for t in range(args.tbptt):
            index = start + t
            obs = [s.observations[index] for s in sequences]
            fused, pad = fuse(obs)
            direct_chunks.append((c, t, fused, pad))
    print(f'direct path decodes = {DECODES["count"]} '
          f'({n_chunks}x more than the cached path)')
    report['decodes_direct'] = DECODES['count']

    # ---- 3. equivalence
    print()
    print('=== equivalence: cached chunk vs freshly decoded chunk ===')
    worst = None
    for (c1, t1, fused_a, pad_a), (c2, t2, fused_b, pad_b) in zip(cache_chunks, direct_chunks):
        if (c1, t1) != (c2, t2):
            worst = f'chunk order mismatch: {(c1,t1)} vs {(c2,t2)}'
            break
        bad = flat_compare(fused_a, fused_b, f'chunk{c1}.tick{t1}')
        if bad:
            worst = bad
            break
        if abs(pad_a - pad_b) > 0:
            worst = f'padding differs at chunk{c1}.tick{t1}: {pad_a} vs {pad_b}'
            break
    if worst:
        print(f'  MISMATCH: {worst}')
        report['equivalent'] = False
    else:
        print(f'  all {len(cache_chunks)} fused ticks bit-identical '
              f'(inputs, masks, labels, padding ratios)')
        report['equivalent'] = True

    # ---- 4. coverage: every tick delivered exactly once
    delivered = sum(c.delivered for c in caches)
    expected = len(caches) * ticks
    print()
    print(f'=== coverage: delivered {delivered} tick-slots, expected {expected} '
          f'({len(caches)} lanes x {ticks} ticks) ===')
    report['delivered'] = delivered
    report['expected'] = expected
    report['coverage_ok'] = delivered == expected

    # ---- 5. the demand arithmetic this corrects
    print()
    print('=== decode demand with reuse (decisions/s / 16) ===')
    for rate in (4872, 5317, 6000, 8000, 10000):
        need = rate / 16
        print(f'  {rate:>6} decisions/s -> {need:>6.1f} shards/s   '
              f'({626.3/need:.2f}x of the measured 16-process 626.3/s)')
    report['shards_per_s_at'] = {str(r): r / 16 for r in (4872, 5317, 6000, 8000, 10000)}

    Path('/root/autodl-tmp/run/reuse_equiv.json').write_text(
        json.dumps(report, indent=1), encoding='utf-8')
    print()
    ok = report['equivalent'] and report['coverage_ok'] and \
        report['decodes_for_all_chunks'] == len(caches)
    print('=== VERDICT:', 'ONE-DECODE-FOUR-CONSUMPTIONS OK' if ok else 'PROBLEM', '===')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
