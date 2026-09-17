#!/usr/bin/env python3
"""Batched R0 imitation training: turn the measured 319x batch win into an actual trainer.

WHY THIS FILE EXISTS: the batch sweep shows the GPU runs at 25% utilisation at B=1 (16 decisions/s)
and 98% at B=4096 (5108 decisions/s), but r0_train.py walks one lane at a time and cannot express
a batch. This drives the same update as training_r0.il_learning.update_imitation_sequence, over B
independent lanes advancing one tick per step, which is exactly what a batched trainer must do --
the RNN dependency is along a lane, never across lanes, so lanes are freely parallel.

PADDING RULE (verified earlier with verify_padding2.py: outputs differ by ~2e-6 with zero diverging
fields, so it is transparent):
  * a field whose name contains 'uid' is masked by `uid >= 0`  -> pad with -1
  * a bool field is a mask                                     -> pad with False
  * every other numeric field                                  -> pad with 0
  * pad dim 1 (dim 0 is the batch axis) up to the max seen in this batch
Padding to the batch's own maximum needs no capacity table, and a hand-written field list provably
misses fields (it missed semantic.dynamic), so the walk below is structural.

usage:
  r0_train_batched.py --dataset <root> --ledger <file> --output <dir> --batch-size 256 --steps 50
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import json
import multiprocessing
import math
import sys
import time
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F

# Only /root is on the path: BOTH /root/data_pipeline and /root/r0/data_pipeline exist on this
# host, and letting /root/r0 win made the import resolve to a different module than the one whose
# contract matched the data. training_r0 must also be importable as "training_r0/..." relative to
# the CWD, which is what current_contract() hashes.
sys.path.insert(0, '/root')

from data_pipeline.r0_admission import current_contract, require_batch
from data_pipeline.r0_train import lane_shards, usable_battles
from training_r0.catalog import CardVocabulary
from training_r0.config import ModelConfig, Temperatures
from training_r0.il_dataset import load_sequence
from training_r0.model import R0Policy, RecurrentState


# ---------------------------------------------------------------- padding


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


def sentinel_for(name, tensor):
    if 'uid' in name.lower():
        return -1
    if tensor.dtype == torch.bool:
        return False
    return 0


def pad_to(obj, targets, path=''):
    """Pad EVERY non-batch axis up to the batch maximum.

    Dim-1-only padding was not enough: `semantic.membership` has shape (1, 1, N) with N in
    {6, 8, 10}, so its variable axis is dim 2 while dim 1 is pinned at 1. Padding any axis that is
    individually too small handles both cases with one rule, because F.pad's spec is written from
    the last axis backwards.
    """
    if isinstance(obj, torch.Tensor):
        want = targets.get(path)
        if want is None or obj.dim() < 2 or len(want) != obj.dim() - 1:
            return obj
        have = list(obj.shape[1:])
        deltas = [w - h for w, h in zip(want, have)]
        if all(d <= 0 for d in deltas):
            return obj
        spec: list[int] = []
        for d in reversed(deltas):
            spec += [0, max(0, d)]
        return F.pad(obj, spec, mode='constant', value=sentinel_for(path, obj))
    if dataclasses.is_dataclass(obj):
        return type(obj)(**{f.name: pad_to(getattr(obj, f.name), targets, f'{path}.{f.name}')
                            for f in dataclasses.fields(obj)})
    if isinstance(obj, tuple):
        return tuple(pad_to(v, targets, f'{path}[{i}]') for i, v in enumerate(obj))
    if isinstance(obj, list):
        return [pad_to(v, targets, f'{path}[{i}]') for i, v in enumerate(obj)]
    return obj


def tree_cat(objs):
    """Concatenate along dim 0 through a nested dataclass/tuple structure."""
    first = objs[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(objs, dim=0)
    if dataclasses.is_dataclass(first):
        return type(first)(**{f.name: tree_cat([getattr(o, f.name) for o in objs])
                              for f in dataclasses.fields(first)})
    if isinstance(first, tuple):
        return tuple(tree_cat([o[i] for o in objs]) for i in range(len(first)))
    if isinstance(first, list):
        return [tree_cat([o[i] for o in objs]) for i in range(len(first))]
    if isinstance(first, int) and not isinstance(first, bool):
        # A plain int field is a per-item count or size, so it must add up, not take the first
        # value -- taking objs[0] would report a batch of 1 for a batch of B.
        return sum(objs)
    return first


# Global shape tables, learned once from a sample of shards and then used to pad each observation
# AT LOAD TIME. This is the fix for the measured bottleneck: padding every lane on every micro step
# cost ~887 ms of the 1017 ms step at B=512 (87%), because it walked 512 observation trees of ~200
# tensors twice per step. Padding once per shard (16 observations per ~16 steps) moves that cost
# off the critical path and reduces batch assembly to torch.cat.
SHAPE_OBS: dict = {}
SHAPE_ACTION: dict = {}

# Bounds how many shard loads may be outstanding. Without it, constructing B lanes queues
# 2*B futures at once (1024 at B=512) and the pool dies on fd exhaustion.
INFLIGHT = threading.BoundedSemaphore(16)
LOADER_ERRORS: list = []


def learn_shapes(objs, table) -> None:
    """Grow `table` to cover every observation seen, so padded tensors always concatenate."""
    for obj in objs:
        for path, tensor in walk(obj).items():
            if tensor.dim() < 2:
                continue
            shape = list(tensor.shape[1:])
            current = table.get(path)
            if current is None or len(current) != len(shape):
                table[path] = shape
            else:
                table[path] = [max(a, b) for a, b in zip(current, shape)]


def fuse(objs):
    """Pad each variable axis to THIS batch's maximum, then concatenate.

    Returns (batch, padding_fraction) so the run can report how much of the batch is padding, which
    the worklist asks for and which also guards against a silent blow-up in batch size.
    """
    targets: dict = {}
    for obj in objs:
        for path, tensor in walk(obj).items():
            if tensor.dim() < 2:
                continue
            shape = list(tensor.shape[1:])
            current = targets.get(path)
            if current is None or len(current) != len(shape):
                targets[path] = shape
            else:
                targets[path] = [max(a, b) for a, b in zip(current, shape)]
    padded = [pad_to(o, targets) for o in objs]
    real, total = 0, 0
    for obj in padded:
        for path, tensor in walk(obj).items():
            if tensor.dim() >= 2:
                prod = 1
                for dim in tensor.shape[1:]:
                    prod *= dim
                total += prod
    for obj in objs:
        for path, tensor in walk(obj).items():
            if tensor.dim() >= 2:
                prod = 1
                for dim in tensor.shape[1:]:
                    prod *= dim
                real += prod
    return tree_cat(padded), (1.0 - real / total if total else 0.0)


# ---------------------------------------------------------------- lanes


def _noop(index: int = 0) -> int:
    """Trivial worker task whose only purpose is to force the pool to spawn its processes.

    Must be a module-level function: os.getpid takes no arguments so it cannot be mapped, and a
    lambda cannot be pickled to the workers.
    """
    return os.getpid()


def _init_worker(obs_table: dict, action_table: dict) -> None:
    """Give each loader process the padding tables it must apply."""
    global SHAPE_OBS, SHAPE_ACTION
    SHAPE_OBS = obs_table
    SHAPE_ACTION = action_table


def load_and_pad(path: str, sha: str):
    """Worker entry point: read one shard, pad it to the learned tables, return plain lists.

    Runs in a separate PROCESS because the decode path holds the GIL; padding here rather than in
    the trainer keeps pad_to out of the micro-step loop entirely.
    """
    sequence = load_sequence(Path(path), expected_sha256=sha)
    # Decode only. Padding here produced hundreds of MB per shard to ship back, which killed the
    # worker; and padding to a global maximum inflates every tensor to the corpus-wide worst case.
    # fuse() pads to the maximum of the batch actually being trained instead.
    return (list(sequence.observations), list(sequence.actions), list(sequence.label_known))


class Lane:
    """One battle-side: a sequence of shards walked in tick order with its own recurrent state.

    Observations and forced actions are padded to the learned shape table the moment the shard is
    read, so that the per-micro-step batch assembly is a plain concatenation. Padding here costs
    one pass per shard instead of one pass per lane per step.
    """

    __slots__ = ('tag', 'dataset', 'key', 'shards', 'index', 'observations', 'actions',
                 'label_known', 'position', 'state', 'finished', 'executor', 'pending',
                 'pending_index')

    def __init__(self, tag, dataset, key, shards, executor=None):
        self.tag, self.dataset, self.key, self.shards = tag, dataset, key, shards
        self.executor = executor
        self.index = -1
        self.pending = None
        self.pending_index = -1
        self.observations = []
        self.actions = []
        self.label_known = []
        self.position = 0
        self.state = None
        self.finished = False
        self._submit(0)
        self._advance_shard()

    def _submit(self, index):
        """Start loading shard `index`, in the pool when there is one, otherwise inline."""
        self.pending_index = index
        if index >= len(self.shards):
            self.pending = None
            return
        record = self.shards[index]
        target = str((self.dataset / record['path']).resolve())
        if self.executor is None:
            future: Future = Future()
            future.set_result(load_and_pad(target, record['sha256']))
            self.pending = future
        else:
            INFLIGHT.acquire()
            future = self.executor.submit(load_and_pad, target, record['sha256'])
            future.add_done_callback(lambda _f: INFLIGHT.release())
            self.pending = future

    def _advance_shard(self):
        if self.pending is None:
            self.finished = True
            return
        record = self.shards[self.pending_index]
        if record.get('starts_episode') is False and self.state is None:
            # A mid-battle shard without its predecessor cannot carry memory across.
            self.finished = True
            return
        try:
            self.observations, self.actions, self.label_known = self.pending.result()
        except Exception as error:
            # Never turn a loader failure into "this lane simply ended": that silently drops data
            # and would show up as a throughput number with no error attached.
            LOADER_ERRORS.append(f'{type(error).__name__}: {str(error)[:200]}')
            self.finished = True
            return
        self.index = self.pending_index
        self.position = 0
        self.pending = None
        # Kick off the NEXT shard immediately, so its load overlaps this shard's training steps.
        self._submit(self.index + 1)

    @property
    def alive(self):
        return not self.finished and bool(self.observations)

    def current(self):
        return (self.observations[self.position],
                self.actions[self.position],
                self.label_known[self.position])

    def step_done(self, next_state):
        self.state = next_state
        self.position += 1
        if self.position >= len(self.observations):
            self.state = self.state.detach() if self.state is not None else None
            self._advance_shard()


def cat_states(states, device):
    return RecurrentState(torch.cat([s.hidden for s in states], 0).to(device),
                          torch.cat([s.cell for s in states], 0).to(device))


def split_state(state, sizes):
    """Split a fused recurrent state back into per-lane states, staying on the GPU.

    Moving these to CPU between steps would add a device round-trip every micro step for no
    benefit: at B=4096 the whole state is 4096*768*4*2 bytes = 25 MB, which is nothing next to a
    95 GiB card.
    """
    # NO detach here: the graph must survive to the TBPTT chunk boundary, exactly as
    # update_imitation_sequence keeps it. Detaching per tick turns TBPTT into a 1-step window.
    out, offset = [], 0
    for n in sizes:
        out.append(RecurrentState(state.hidden[offset:offset + n],
                                  state.cell[offset:offset + n]))
        offset += n
    return out


# ---------------------------------------------------------------- training


def build_lanes(roots, ledgers, contract):
    battles, counts = usable_battles(roots, ledgers)
    lanes, evaluated = [], []
    for tag, dataset in sorted(battles.items()):
        manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('contract_sha256') != contract['sha256']:
            continue
        evaluated.append(dataset)
        for key, shards in lane_shards(dataset, manifest, 'train').items():
            lanes.append((tag, dataset, key, shards))
    return lanes, counts, evaluated


def _train_legacy(args) -> dict:
    torch.manual_seed(0)
    device = args.device
    roots = [Path(r) for r in args.dataset]
    contract = current_contract()
    lanes, counts, evaluated = build_lanes(roots, [Path(p) for p in args.ledger], contract)
    source_hashes = contract.get('source_hashes') or {}
    run_log = {
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda_build': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0),
        'capability': list(torch.cuda.get_device_capability(0)),
        'cwd': os.getcwd(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'contract_sha256': contract['sha256'],
        'contract_source_hashes': len(source_hashes),
        'usable_battles': counts['usable'],
        'training_lanes': len(lanes),
        'datasets_evaluated': len(evaluated),
        'batch_size': args.batch_size,
        'tbptt_steps': args.tbptt_steps,
        'loaders': args.loaders,
        'reuse_lanes': args.reuse_lanes,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    print('RUN LOG ' + json.dumps(run_log, sort_keys=True), flush=True)
    if not source_hashes:
        # An empty source_hashes is exactly how every compiled battle came to be rejected as
        # "compiled under another contract"; never paper over it by relaxing admission.
        raise SystemExit('REFUSING TO RUN: contract source_hashes is empty')
    print(f'usable battles: {counts["usable"]}  training lanes: {len(lanes)}  '
          f'datasets: {len(evaluated)}', flush=True)
    if not lanes:
        raise SystemExit('no training lanes')

    # Learn the padding target ONCE, from a spread of lanes, before any Lane pads against it.
    # Observation and action trees keep separate tables because their field names overlap (both
    # carry a target_cell), and a shared table would pad one against the other's widths.
    obs_sample, act_sample = [], []
    for _, dataset, _, shards in lanes[:args.shape_sample]:
        record = shards[0]
        sequence = load_sequence((dataset / record['path']).resolve(),
                                 expected_sha256=record['sha256'])
        obs_sample.extend(sequence.observations)
        act_sample.extend(sequence.actions)
    learn_shapes(obs_sample, SHAPE_OBS)
    learn_shapes(act_sample, SHAPE_ACTION)
    print(f'shape table: {len(SHAPE_OBS)} obs paths / {len(SHAPE_ACTION)} action paths '
          f'learned from {len(obs_sample)} observations', flush=True)

    # The pool MUST be created before any CUDA context exists: forking a process that already owns
    # a CUDA context leaves the child with a copy it cannot use, which killed the workers.
    model = R0Policy(CardVocabulary.from_native(), ModelConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # Pool created AFTER CUDA, matching bisect stage C, which loads real shards successfully. The
    # workers are deliberately NOT pre-warmed: the executor forks lazily on first submit, which is
    # the path stage C exercised.
    pool = None
    if args.loaders:
        pool = ProcessPoolExecutor(
            max_workers=args.loaders,
            # spawn, not fork: a forked child inherits the parent's CUDA context and dies on
            # first use. Verified by test_spawn.py, which loads real shards through a spawn pool
            # while the parent holds a live CUDA context.
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_worker,
            initargs=(SHAPE_OBS, SHAPE_ACTION))
        print(f'loader processes: {args.loaders} (start method: '
              f'{multiprocessing.get_context("spawn").get_start_method()})', flush=True)
    # batch = min(requested, available) is not enough on this slice: it holds only ~70 training
    # lanes (40 battles), so every batch above 70 silently ran as B=70 and the whole sweep came out
    # flat. --reuse-lanes cycles the pool so a large batch can actually be built and timed; it is a
    # measurement tool, not a training mode, because it repeats episodes.
    batch = min(args.batch_size, len(lanes)) if not args.reuse_lanes else args.batch_size
    supply = cycle(lanes) if args.reuse_lanes else iter(lanes)
    active = []
    for _ in range(batch):
        try:
            active.append(Lane(*next(supply), executor=pool))
        except StopIteration:
            break
    batch = len(active)
    print(f'active lanes at start: {batch} (pool {len(lanes)}, reuse={args.reuse_lanes})',
          flush=True)
    for lane in active:
        lane.state = model.initial_state(1, device=device)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    micro = 0
    pending: list = []
    optimizer_steps = 0
    total_loss = 0.0
    decisions = 0
    peak_batch = 0

    def alive_lanes():
        return [l for l in active if l.alive]

    while optimizer_steps < args.steps:
        current = alive_lanes()
        if not current:
            break
        peak_batch = max(peak_batch, len(current))
        observations, actions, known = [], [], []
        for lane in current:
            o, a, k = lane.current()
            observations.append(o)
            actions.append(a)
            known.append(k)
        fused_obs, pad_ratio = fuse(observations)
        fused_obs = fused_obs.to(device)
        fused_actions = fuse(actions)[0].to(device)
        fused_known = torch.cat(known).to(device)
        state = cat_states([l.state for l in current], device)

        require_batch(fused_obs, model)
        out = model(fused_obs, state, forced=fused_actions, temperatures=Temperatures())
        if not torch.isfinite(out.logp[fused_known]).all():
            raise SystemExit('expert labels conflict with native candidate legality')

        count = int(fused_known.sum())
        if count == 0:
            raise SystemExit('no accepted imitation targets in batch')
        loss = -torch.where(fused_known, out.logp, torch.zeros_like(out.logp)).sum() / count
        pending.append(loss)
        micro += 1
        total_loss += float(loss.detach())
        decisions += count

        # Every lane contributes exactly ONE row to the fused batch (one tick each), so the split
        # sizes are all 1. An earlier version passed len(known) repeated, which would have split a
        # B-row state into B chunks of B rows.
        next_states = split_state(out.next_state, [1] * len(current))
        for lane, ns in zip(current, next_states):
            lane.step_done(ns)

        # Refill: a lane that walked its whole battle frees a slot, and without a replacement the
        # effective batch would shrink to zero over a long run.
        for i, lane in enumerate(active):
            if not lane.alive:
                try:
                    fresh = Lane(*next(supply), executor=pool)
                except StopIteration:
                    continue
                fresh.state = model.initial_state(1, device=device)
                active[i] = fresh

        if micro % args.tbptt_steps == 0:
            # One backward for the whole chunk, matching update_imitation_sequence.
            torch.stack(pending).sum().backward()
            pending = []
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # The chunk boundary is where the old trainer detaches, so this is where the graph is
            # cut -- one cut per TBPTT window, not one per tick.
            for lane in active:
                if lane.state is not None:
                    lane.state = lane.state.detach()
            optimizer_steps += 1
            if optimizer_steps % args.log_every == 0:
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    'step': optimizer_steps,
                    'decisions': decisions,
                    'decisions_per_s': round(decisions / elapsed, 1),
                    'mean_loss': round(total_loss / max(1, micro), 5),
                    'grad_norm': round(float(norm), 4),
                    'active_lanes': len(current),
                    'pad_frac': round(pad_ratio, 4),
                    'gpu_GiB': round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    'elapsed_s': round(elapsed, 1),
                }), flush=True)

    elapsed = time.perf_counter() - started
    summary = {
        'optimizer_steps': optimizer_steps,
        'micro_steps': micro,
        'batch_size': batch,
        'peak_active_lanes': peak_batch,
        'decisions': decisions,
        'seconds': round(elapsed, 1),
        'decisions_per_s': round(decisions / elapsed, 1) if elapsed else 0,
        'mean_loss': round(total_loss / max(1, micro), 6),
        'gpu_peak_GiB': round(torch.cuda.max_memory_allocated() / 2**30, 2),
        'contract_sha256': contract['sha256'],
        'loader_errors': len(LOADER_ERRORS),
        'loader_error_sample': LOADER_ERRORS[:3],
    }
    (output / 'batched_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    return summary


# ---------------------------------------------------------------- backend-driven training

# ---------------------------------------------------------------- data sources

class InlineSource:
    """The original path: Lanes load their own shards in-process."""

    def __init__(self, lanes, batch, executor=None):
        from itertools import cycle
        self.queue = list(lanes)
        self.supply = cycle(lanes)
        self.active = []
        for _ in range(min(batch, len(lanes)) if len(lanes) < batch else batch):
            self.active.append(Lane(*next(self.supply), executor=executor))
        for lane in self.active:
            lane.state = None
        self.batch = len(self.active)

    def _ensure_states(self, model, device):
        for lane in self.active:
            if lane.state is None:
                lane.state = model.initial_state(1, device=device)

    def next_tick(self, model, device):
        self._ensure_states(model, device)
        self.current = [lane for lane in self.active if lane.alive]
        observations, actions, known = [], [], []
        for lane in self.current:
            o, a, k = lane.current()
            observations.append(o)
            actions.append(a)
            known.append(k)
        return observations, actions, known

    def commit(self, next_states):
        for lane, state in zip(self.current, next_states):
            lane.step_done(state)
        self.refill()

    def refill(self):
        for index, lane in enumerate(self.active):
            if not lane.alive:
                try:
                    fresh = Lane(*next(self.supply))
                except StopIteration:
                    continue
                fresh.state = None
                self.active[index] = fresh

    def alive_count(self):
        return sum(1 for lane in self.active if lane.alive)

    def stats(self):
        return {'source': 'inline', 'decodes': None}


class BackendSource:
    """Same interface; observations, expert actions and labels all come from shared slots."""

    def __init__(self, lanes, batch, tbptt, args, device):
        import r0_data_backend as backend_mod
        self.mod = backend_mod
        self.tbptt = tbptt
        self.batch = min(batch, len(lanes)) if len(lanes) < batch else batch
        # One spec per SLOT; a slot is refilled from the queue when its lane finishes, which is
        # what makes this a continuous corpus entry point rather than "first B lanes only".
        self.specs = []
        for tag, dataset, key, shards in lanes[:self.batch]:
            self.specs.append((f'{tag}/{key[0]}-{key[1]}',
                               [(str((dataset / r['path']).resolve()), r['sha256'])
                                for r in shards]))
        self.queue = list(lanes[self.batch:])
        self.cursor = 0

        first_path, first_sha = self.specs[0][1][0]
        probe = load_sequence(Path(first_path), expected_sha256=first_sha)
        obs_tensors, self.template = backend_mod.flatten_tree(probe.observations[0])
        act_tensors, self.action_template = backend_mod.flatten_tree(probe.actions[0])
        self.n_obs_fields = len(obs_tensors)
        self.dtypes = [t.dtype for t in obs_tensors] + [t.dtype for t in act_tensors]
        bytes_per_entry = (sum(t.numel() * t.element_size() for t in obs_tensors)
                           + sum(t.numel() * t.element_size() for t in act_tensors))
        del probe

        self.layout = backend_mod.SlotLayout(batch=self.batch, tbptt=tbptt,
                                             n_fields=len(self.dtypes),
                                             bytes_per_entry=bytes_per_entry)
        self.data = backend_mod.DataBackend(self.specs, self.layout, n_workers=args.loaders,
                                            n_slots=max(2, args.slots)).start()
        self.ticks_per_shard = 16
        self.chunks_per_shard = self.ticks_per_shard // tbptt
        self.row_cache = None
        self.row_labels = None
        self.row_pos = tbptt            # force a fill on the first call
        self.generation = 0
        self.shard_index = 0
        self.offset_seconds = 0.0

    def _shard_ordinal(self):
        """Which shard of each lane this chunk needs.

        self.shard_index counts CHUNKS (one increment per T=4 chunk); len(spec[1]) counts SHARDS.
        Comparing them directly made a 35-shard lane look exhausted after 35 chunks instead of 140.
        """
        return self.shard_index // self.chunks_per_shard

    def _live(self):
        """Slots whose lane still has shards left, refilled from the queue until the width is full.

        EVERY dead slot is replaced, not just the first: a batch that quietly narrows also
        invalidates the fused recurrent state.
        """
        ordinal = self._shard_ordinal()
        while self.queue:
            dead = [i for i, spec in enumerate(self.specs) if ordinal >= len(spec[1])]
            if not dead:
                break
            for index in dead:
                if not self.queue:
                    break
                spec = self.queue.pop(0)
                self.specs[index] = (
                    spec[0],
                    [(str((spec[1] / r['path']).resolve()), r['sha256']) for r in spec[3]])
        return [i for i, spec in enumerate(self.specs) if ordinal < len(spec[1])]

    def _live_slots(self):
        ordinal = self._shard_ordinal()
        return [i for i, spec in enumerate(self.specs) if ordinal < len(spec[1])]

    def next_tick(self, model, device):
        live = self._live()
        if not live:
            return None
        if self.row_pos >= self.tbptt:
            tick_base = (self.shard_index % self.chunks_per_shard) * self.tbptt
            shard_index = self._shard_ordinal()
            requests = []
            for slot in live:
                records = self.specs[slot][1]
                shard_path, sha = records[shard_index]
                for t in range(self.tbptt):
                    requests.append((slot, tick_base + t, shard_path, sha))
            self.generation += 1
            slot_id = self.generation % self.data.n_slots
            start = time.perf_counter()
            _, self.row_labels = self.data.fill(slot_id, self.generation, requests, tick_base)
            self.row_cache = self.data.read(slot_id, self.generation, self.dtypes)
            self.offset_seconds += time.perf_counter() - start
            self.row_pos = 0
            self.shard_index += 1
        row = self.row_pos
        self.row_pos += 1
        self.current = live
        observations, actions, known = [], [], []
        for slot in live:
            tensors = self.row_cache[slot * self.tbptt + row]
            observations.append(self.mod.unflatten_tree(tensors[:self.n_obs_fields],
                                                        self.template))
            actions.append(self.mod.unflatten_tree(tensors[self.n_obs_fields:],
                                                   self.action_template))
            known.append(torch.tensor([self.row_labels[slot * self.tbptt + row]]))
        return observations, actions, known

    def commit(self, next_states):
        self.state = next_states

    def refill(self):
        return None

    def alive_count(self):
        return len(self._live())

    def stats(self):
        return {'source': 'backend', 'workers': self.data.stats(),
                'slot_MiB': round(self.layout.total_bytes / 2**20, 2),
                'read_seconds': round(self.offset_seconds, 2)}

    def close(self):
        self.data.stop()


# ---------------------------------------------------------------- the single training loop

def run_training(args, source, model, optimizer, device, contract, run_log):
    args.tbptt = args.tbptt_steps   # the CLI spells it tbptt_steps
    """THE training loop. Both data sources drive this; nothing here knows which one it is."""
    torch.manual_seed(0)
    print('RUN LOG ' + json.dumps(run_log, sort_keys=True), flush=True)

    started = time.perf_counter()
    optimizer_steps = 0
    micro = 0
    pending: list = []
    total_loss = 0.0
    decisions = 0
    pad_max = 0.0
    t_fuse = t_h2d = t_fwd = t_bwd = 0.0
    state = None

    while optimizer_steps < args.steps:
        got = source.next_tick(model, device)
        if got is None:
            break
        observations, actions, known = got
        live = getattr(source, 'current', None)
        if live is None:
            live = []

        t0 = time.perf_counter()
        fused_obs, pad_ratio = fuse(observations)
        fused_actions = fuse(actions)[0]
        fused_known = torch.cat(known).to(device)
        pad_max = max(pad_max, pad_ratio)
        t_fuse += time.perf_counter() - t0

        t0 = time.perf_counter()
        fused_obs = fused_obs.to(device)
        fused_actions = fused_actions.to(device)
        t_h2d += time.perf_counter() - t0

        if state is None:
            state = model.initial_state(fused_obs.batch_size, device=device)
        require_batch(fused_obs, model)

        t0 = time.perf_counter()
        out = model(fused_obs, state, forced=fused_actions, temperatures=Temperatures())
        loss = None
        count = int(fused_known.sum())
        if not torch.isfinite(out.logp[fused_known]).all():
            raise SystemExit('expert labels conflict with native candidate legality')
        loss = -torch.where(fused_known, out.logp, torch.zeros_like(out.logp)).sum() / count
        t_fwd += time.perf_counter() - t0

        pending.append(loss)
        micro += 1
        total_loss += float(loss.detach())
        decisions += count
        state = out.next_state
        source.commit(out.next_state if source.__class__.__name__ == 'BackendSource'
                      else split_state(out.next_state, [1] * len(live)))

        if micro % args.tbptt == 0:
            t0 = time.perf_counter()
            torch.stack(pending).sum().backward()
            pending = []
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                  error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state = state.detach()
            t_bwd += time.perf_counter() - t0
            optimizer_steps += 1
            if args.log_every and optimizer_steps % args.log_every == 0:
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    'step': optimizer_steps, 'decisions': decisions,
                    'decisions_per_s': round(decisions / elapsed, 1),
                    'mean_loss': round(total_loss / max(1, micro), 5),
                    'grad_norm': round(float(norm), 4),
                    'alive': source.alive_count(), 'pad_frac': round(pad_ratio, 4),
                    't_fuse': round(t_fuse, 2), 't_h2d': round(t_h2d, 2),
                    't_fwd': round(t_fwd, 2), 't_bwd': round(t_bwd, 2),
                }), flush=True)

    elapsed = time.perf_counter() - started
    stats = source.stats()
    summary = {
        'optimizer_steps': optimizer_steps, 'micro_steps': micro,
        'batch_size': getattr(source, 'batch', 0), 'decisions': decisions,
        'seconds': round(elapsed, 2),
        'decisions_per_s': round(decisions / elapsed, 1) if elapsed else 0.0,
        'mean_loss': round(total_loss / max(1, micro), 6),
        'max_pad_frac': round(pad_max, 4),
        'gpu_peak_GiB': round(torch.cuda.max_memory_allocated() / 2**30, 2),
        'time_fuse_s': round(t_fuse, 2), 'time_h2d_s': round(t_h2d, 2),
        'time_forward_s': round(t_fwd, 2), 'time_backward_opt_s': round(t_bwd, 2),
        'share_fuse': round(t_fuse / elapsed, 3) if elapsed else 0,
        'share_h2d': round(t_h2d / elapsed, 3) if elapsed else 0,
        'share_forward': round(t_fwd / elapsed, 3) if elapsed else 0,
        'share_backward_opt': round(t_bwd / elapsed, 3) if elapsed else 0,
        'unaccounted_s': round(elapsed - (t_fuse + t_h2d + t_fwd + t_bwd), 2),
        'source': stats, 'contract_sha256': contract['sha256'],
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / 'training_summary.json').write_text(json.dumps(summary, indent=1),
                                                            encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def _prepare(args):
    """Shared front matter: contract, lanes, model, optimizer, run log."""
    device = args.device
    roots = [Path(r) for r in args.dataset]
    contract = current_contract()
    lanes, counts, evaluated = build_lanes(roots, [Path(p) for p in args.ledger], contract)
    source_hashes = contract.get('source_hashes') or {}
    if not source_hashes:
        raise SystemExit('REFUSING TO RUN: contract source_hashes is empty')
    if not lanes:
        raise SystemExit('no training lanes')
    print(f'usable battles: {counts["usable"]}  training lanes: {len(lanes)}  '
          f'datasets: {len(evaluated)}', flush=True)
    model = R0Policy(CardVocabulary.from_native(), ModelConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    return device, contract, lanes, model, optimizer, source_hashes


def _run_log(mode, args, contract, source_hashes, batch, extra=None):
    log = {
        'mode': mode, 'python': sys.version.split()[0], 'torch': torch.__version__,
        'cuda_build': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'contract_sha256': contract['sha256'], 'contract_source_hashes': len(source_hashes),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'batch': batch, 'tbptt': args.tbptt_steps, 'cwd': os.getcwd(),
        'seed': 0,
    }
    if extra:
        log.update(extra)
    return log


def train(args) -> dict:
    """Inline loading. Drives the SAME run_training loop as the backend path."""
    device, contract, lanes, model, optimizer, source_hashes = _prepare(args)
    source = InlineSource(lanes, args.batch_size)
    log = _run_log('inline', args, contract, source_hashes, source.batch,
                   {'training_lanes': len(lanes)})
    return run_training(args, source, model, optimizer, device, contract, log)


def train_with_backend(args) -> dict:
    """Shared-slot backend. Drives the SAME run_training loop as the inline path."""
    device, contract, lanes, model, optimizer, source_hashes = _prepare(args)
    source = BackendSource(lanes, args.batch_size, args.tbptt_steps, args, device)
    log = _run_log('backend', args, contract, source_hashes, source.batch,
                   {'training_lanes': len(lanes), 'loaders': args.loaders,
                    'slots': args.slots,
                    'slot_MiB': round(source.layout.total_bytes / 2**20, 2)})
    try:
        return run_training(args, source, model, optimizer, device, contract, log)
    finally:
        source.close()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', nargs='+', required=True)
    ap.add_argument('--ledger', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--steps', type=int, default=50, help='optimizer steps')
    ap.add_argument('--tbptt-steps', type=int, default=4)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--backend', action='store_true',
                    help='fetch observations from r0_data_backend shared slots instead of inline loads')
    ap.add_argument('--slots', type=int, default=2,
                    help='shared-memory slots to allocate in backend mode')
    ap.add_argument('--loaders', type=int, default=8,
                    help='processes prefetching and padding shards; 0 disables the pool')
    ap.add_argument('--shape-sample', type=int, default=40,
                    help='lanes sampled to learn the padding table before training starts')
    ap.add_argument('--reuse-lanes', action='store_true',
                    help='cycle the lane pool so batches larger than the lane count can be timed')
    args = ap.parse_args()
    print(json.dumps({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                      'batch_size': args.batch_size, 'tbptt': args.tbptt_steps,
                      'backend': args.backend}, indent=1))
    if args.backend:
        train_with_backend(args)
    else:
        train(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
