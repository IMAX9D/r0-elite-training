#!/usr/bin/env python3
"""Stage 2A: does the batched path reproduce the old trainer at B=1?

WHY THIS IS THE DECIDING TEST: everything downstream (throughput, capacity planning) assumes the
batched loop optimises the same objective with the same truncation as
training_r0.il_learning.update_imitation_sequence. If it does not, a faster number is meaningless.

SETUP: same seed -> same initial weights; same lane; same optimizer; tbptt set to the whole
sequence so the old trainer's single optimizer step (it accumulates gradients across chunks and
steps once at the end) corresponds to exactly one chunk boundary in the batched loop.

COMPARED, at identical parameters: total loss and its denominator, every parameter gradient, and
the parameters after one optimizer update. Tolerances are declared up front, not tuned afterwards.

usage: verify_b1_equiv.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/root')
from training_r0.catalog import CardVocabulary
from training_r0.config import ModelConfig, Temperatures
from training_r0.il_dataset import load_sequence
from training_r0.il_learning import update_imitation_sequence
from training_r0.model import R0Policy
from r0_train_batched import cat_states, fuse, split_state

ROOT = Path('/root/autodl-tmp/out-0/compiled-prod-v17')
TOL_LOSS = 1e-5      # relative, on a sum of ~16 per-tick terms
TOL_GRAD = 1e-5      # relative, per parameter max|diff| / max|ref|
TOL_PARAM = 1e-6     # relative, after one Adam update from identical gradients


def one_lane():
    shard = sorted(ROOT.rglob('*.npz'))[0]
    seq = load_sequence(shard, expected_sha256=hashlib.sha256(shard.read_bytes()).hexdigest())
    return shard, seq


def build(seed):
    torch.manual_seed(seed)
    model = R0Policy(CardVocabulary.from_native(), ModelConfig()).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, optimizer


def grads_of(model):
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()
            if p.grad is not None}


def params_of(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def batched_path(model, optimizer, seq, tbptt):
    """The trainer's loop, restricted to a single lane so it can be compared directly."""
    state = model.initial_state(1, device='cuda')
    total = 0.0
    micro = 0
    pending: list = []
    count_total = int(seq.label_known.sum())
    losses = []
    for position in range(len(seq.observations)):
        obs, actions, known = seq.observations[position], seq.actions[position], seq.label_known[position]
        fused_obs, pad_frac = fuse([obs])
        fused_actions = fuse([actions])[0]
        fused_known = known.unsqueeze(0).to('cuda')
        out = model(fused_obs.to('cuda'), state, forced=fused_actions.to('cuda'),
                    temperatures=Temperatures())
        count = int(fused_known.sum())
        loss = -torch.where(fused_known, out.logp, torch.zeros_like(out.logp)).sum() / count_total
        pending.append(loss)
        losses.append((loss.item(), count, pad_frac))
        total += float(loss.detach())
        state = split_state(out.next_state, [1])[0]
        micro += 1
        if micro % tbptt == 0 or position == len(seq.observations) - 1:
            torch.stack(pending).sum().backward()
            pending = []
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            # Snapshot before zero_grad: the old trainer leaves its gradients in place, so
            # clearing them here would leave nothing to compare against.
            final_grads = grads_of(model)
            optimizer.zero_grad(set_to_none=True)
            state = state.detach()
    return total, state, losses, final_grads


def main() -> int:
    shard, seq = one_lane()
    tbptt = len(seq.observations)
    print(json.dumps({
        'shard': shard.name,
        'observations': len(seq.observations),
        'label_known_total': int(seq.label_known.sum()),
        'tbptt': tbptt,
        'starts_episode': bool(seq.starts_episode),
        'tol_loss': TOL_LOSS, 'tol_grad': TOL_GRAD, 'tol_param': TOL_PARAM,
    }, indent=1), flush=True)

    model_a, opt_a = build(1234)
    # Snapshot BEFORE training A: comparing state_dict() afterwards would compare a trained model
    # against a fresh one and always report a mismatch (which is exactly what the first run did).
    initial_a = {name: tensor.clone() for name, tensor in model_a.state_dict().items()}
    metrics, state_a = update_imitation_sequence(model_a, opt_a, seq,
                                                 initial_state=None, tbptt_steps=tbptt)
    grads_a, params_a = grads_of(model_a), params_of(model_a)

    model_b, opt_b = build(1234)
    same_init = all(torch.equal(tensor, model_b.state_dict()[name])
                    for name, tensor in initial_a.items())
    print('initial weights identical:', same_init, flush=True)
    if not same_init:
        raise SystemExit('ABORT: the two models did not start from the same weights')

    total_b, state_b, losses, grads_b = batched_path(model_b, opt_b, seq, tbptt)
    params_b = params_of(model_b)

    print()
    print('=== loss ===')
    print(f'  old path  : {metrics["loss"]:.8f}   valid_decisions={metrics["valid_decisions"]} '
          f'chunks={metrics["tbptt_chunks"]}')
    print(f'  batched   : {total_b:.8f}')
    rel = abs(metrics['loss'] - total_b) / max(abs(metrics['loss']), 1e-12)
    print(f'  relative  : {rel:.3e}   tol {TOL_LOSS:.0e}   {"PASS" if rel <= TOL_LOSS else "FAIL"}')

    print()
    print('=== gradients ===')
    worst, worst_name = 0.0, ''
    missing = []
    for name, ref in grads_a.items():
        got = grads_b.get(name)
        if got is None:
            missing.append(name)
            continue
        denom = ref.abs().max().item()
        diff = (ref - got).abs().max().item()
        rel_g = diff / denom if denom > 1e-12 else diff
        if rel_g > worst:
            worst, worst_name = rel_g, name
    print(f'  parameters compared : {len(grads_a)}   missing in batched: {len(missing)}')
    print(f'  worst relative diff : {worst:.3e} on {worst_name}')
    print(f'  tolerance           : {TOL_GRAD:.0e}   {"PASS" if worst <= TOL_GRAD else "FAIL"}')
    if missing:
        print('  MISSING:', missing[:5])

    print()
    print('=== parameters after one optimizer update ===')
    worst_p, worst_pn = 0.0, ''
    for name, ref in params_a.items():
        got = params_b[name]
        denom = ref.abs().max().item()
        diff = (ref - got).abs().max().item()
        rel_p = diff / denom if denom > 1e-12 else diff
        if rel_p > worst_p:
            worst_p, worst_pn = rel_p, name
    print(f'  worst relative diff : {worst_p:.3e} on {worst_pn}')
    print(f'  tolerance           : {TOL_PARAM:.0e}   {"PASS" if worst_p <= TOL_PARAM else "FAIL"}')

    print()
    print('=== recurrent state after the sequence ===')
    for field in ('hidden', 'cell'):
        ref = getattr(state_a, field).detach().float()
        got = getattr(state_b, field).detach().float().reshape(ref.shape)
        diff = (ref - got).abs().max().item()
        denom = ref.abs().max().item()
        print(f'  {field}: max|diff| {diff:.3e}  (max|ref| {denom:.3e})')

    ok = (rel <= TOL_LOSS and worst <= TOL_GRAD and worst_p <= TOL_PARAM and not missing)
    print()
    print('=== VERDICT:', 'B=1 EQUIVALENT' if ok else 'DIVERGENT', '===')
    payload = {
        'loss_old': metrics['loss'], 'loss_batched': total_b, 'loss_rel': rel,
        'grad_worst_rel': worst, 'grad_worst_param': worst_name,
        'param_worst_rel': worst_p, 'param_worst_param': worst_pn,
        'missing_grads': missing, 'equivalent': bool(ok),
        'pad_frac_max': max(p for _, _, p in losses) if losses else 0.0,
        'per_tick': [{'loss': l, 'count': c, 'pad_frac': round(p, 4)} for l, c, p in losses],
    }
    Path('/root/autodl-tmp/run/b1_equiv.json').write_text(json.dumps(payload, indent=1),
                                                         encoding='utf-8')
    print('report -> /root/autodl-tmp/run/b1_equiv.json')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
