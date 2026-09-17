"""One traversal, both verification passes.

`strict_input_verification` and `strict_action_verification` each iterate
`manifest['shards']` and call `load_sequence` on exactly the same paths, so every
compiled shard is read from disk, decompressed and objectified twice per battle.
At ~50 npz per battle that is the dominant I/O and object-construction cost of the
input check.

This module runs the two bodies over a SINGLE traversal. Every check is kept
verbatim - nothing is dropped, relaxed or reordered - and only the current shard is
held at a time. The original two-pass functions remain in `r0_certify` and stay the
default, so the fused path can be A/B'd against them on the same dataset.

The A/B contract: for the same dataset_dir/capture_dir the fused function must
return reports equal to the two originals' reports.
"""
from __future__ import annotations

import json
from pathlib import Path

from training_r0.il_dataset import load_sequence
from training_r0.config import ModelConfig

from .r0_admission import require_batch

_CERT_MISSING = 'compiled contract differs from current R0; recompile'


def _fused(dataset_dir, capture_dir, contract, device='cuda', max_ticks=0):
    from training_r0.actions import OFFSET_TICKS, ShadowLegality
    from training_r0.config import DECISION_TICKS
    from .input_validator import StrictInputValidator
    from .r0_compile import canonical_cell, compilation_end_tick, index_capture
    import torch

    dataset_dir = Path(dataset_dir).resolve()
    manifest = json.loads((dataset_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('contract_sha256') != contract['sha256']:
        raise ValueError(_CERT_MISSING)
    config = ModelConfig()
    if config.allow_incomplete_capture:
        raise ValueError('diagnostic R0 cannot certify training data')
    validator = StrictInputValidator(config)

    capture_dir = Path(capture_dir)
    raw = capture_dir / 'native-frames.jsonl.zst'
    if not raw.exists():
        raw = capture_dir / 'native-frames.jsonl.gz'
    episode, commands_index, trace = index_capture(raw)
    scope = manifest.get('compiled_scope') or {}
    end_tick = compilation_end_tick(trace['last_tick'], scope.get('requested_max_tick'))
    if max_ticks:
        end_tick = min(end_tick, max_ticks)
    bad_ticks = trace['bad_ticks']

    problems: list[str] = []
    readback_failures: list[str] = []
    shards = manifest['shards']
    partitions = {shard.get('partition') for shard in shards}
    battles = {shard.get('battle_uid') for shard in shards}
    battle_level_split = bool(shards) and len(partitions) == 1 and len(battles) == 1 and \
        partitions <= {'train', 'validation', 'test'}
    if not battle_level_split:
        problems.append(f'battle split inconsistent: partitions={sorted(partitions)} battles={len(battles)}')

    # input-side counters (names mirror strict_input_verification)
    samples = known_bits = sequences = labelled = 0
    # action-side counters (names mirror strict_action_verification)
    a_samples = a_labelled = cited = wait_labels = accounted_mismatch = 0

    for shard in shards:
        path = (dataset_dir / shard['path']).resolve()
        if not path.is_relative_to(dataset_dir):
            readback_failures.append(f'{shard["path"]}: outside dataset')
            continue

        # ---- ONE load for both passes ----
        # The loader verifies the recorded SHA256, so a shard that does not match its
        # manifest entry cannot be read silently.
        sequence = load_sequence(path, expected_sha256=shard['sha256'])

        # ================= pass 1: input contract (verbatim) =================
        labelled += int(sequence.label_known.sum())
        for observation in sequence.observations:
            batch = observation.to(device)
            require_batch(batch, validator)
            banks = (batch.semantic.elite.child, batch.semantic.elite.tower, batch.semantic.elite.groups,
                     batch.semantic.elite.own_cards, batch.semantic.elite.enemy_cards,
                     batch.semantic.elite.match, batch.semantic.elite.combat_features,
                     batch.semantic.elite.candidate_ability_features)
            for bank in banks:
                known = bank[..., 1::2]
                if not ((known == 0) | (known == 1)).all():
                    raise ValueError('invalid feature known bits')
                known_bits += int(known.numel())
            samples += 1
        sequences += 1

        # ================= pass 2: action labels (verbatim) =================
        ticks = [int(observation.ticks[0]) for observation in sequence.observations]
        if len(sequence.observations) != shard['steps']:
            readback_failures.append(f'{shard["path"]}: steps {len(sequence.observations)} != {shard["steps"]}')
        if ticks and (ticks[0] != shard['first_tick'] or ticks[-1] != shard['last_tick']):
            readback_failures.append(f'{shard["path"]}: ticks {ticks[0]}..{ticks[-1]} != '
                                     f'{shard["first_tick"]}..{shard["last_tick"]}')
        if int(sequence.label_known.sum()) != shard['known_labels']:
            readback_failures.append(f'{shard["path"]}: known labels {int(sequence.label_known.sum())} '
                                     f'!= {shard["known_labels"]}')
        if bool(sequence.starts_episode) != bool(shard['starts_episode']):
            readback_failures.append(f'{shard["path"]}: starts_episode mismatch')

        for step, observation in enumerate(sequence.observations):
            batch = observation.to('cpu')
            action = sequence.actions[step]
            tick = int(batch.ticks[0])
            side = int(batch.sides[0])
            a_samples += 1
            try:
                action.validate(1)
            except ValueError as error:
                problems.append(f'{shard["path"]}@{tick}: invalid action sequence: {error}')
                continue
            for order in range(int(action.count[0])):
                if int(action.offset_bin[0, order]) not in OFFSET_TICKS:
                    problems.append(f'{shard["path"]}@{tick}: delay {int(action.offset_bin[0, order])} '
                                    f'is not a legal offset')
            commands = commands_index.get((tick, side), ())
            complete = tick + DECISION_TICKS <= end_tick and not any(tick <= t <= tick + DECISION_TICKS
                                                                    for t in bad_ticks)
            representable = True
            reason = None
            if not complete:
                representable, reason = False, 'incomplete_window'
            elif len(commands) > 2:
                representable, reason = False, 'more_than_two_actions'
            elif any(not c.accepted for c in commands):
                representable, reason = False, 'source_action_rejected_or_unresolved'
            else:
                probe = ShadowLegality(batch)
                for order, command in enumerate(commands):
                    command_action = command.action or {}
                    kind = command_action.get('type', 'play')
                    uid = (command_action.get('deck_index') if kind == 'play'
                           else 10000000 + command_action.get('entity_id', -1) if kind == 'ability' else -1)
                    try:
                        target = (canonical_cell(command_action.get('x'), command_action.get('y'), command.side)
                                  if kind == 'play' else -1)
                    except ValueError as error:
                        representable, reason = False, f'unrepresentable:{error}'
                        break
                    if command.side != side or command.window != tick:
                        representable, reason = False, 'wrong_owner_or_window'
                        break
                    legal = (batch.candidate_uids[0] == uid) & probe.candidate_mask(order)[0]
                    if int(legal.sum()) != 1:
                        representable, reason = False, f'expert_candidate_unavailable:{uid}'
                        break
                    index = int(legal.nonzero()[0, 0])
                    selected = torch.tensor([index])
                    if bool(batch.grid_targets[0, index]) != (target >= 0):
                        representable, reason = False, 'target_mode_mismatch'
                        break
                    if target >= 0 and not bool(probe.placement(selected)[0, target]):
                        representable, reason = False, 'expert_target_illegal'
                        break
                    delay = command.execution_tick - 1 - tick
                    if delay not in OFFSET_TICKS or not bool(probe.offset_mask(order)[0, delay]):
                        representable, reason = False, 'expert_delay_illegal'
                        break
                    probe.apply(torch.tensor([True]), selected, torch.tensor([target]), torch.tensor([delay]))
            expected_known = bool(representable)
            known = bool(sequence.label_known[step][0])
            if known != expected_known:
                accounted_mismatch += 1
                if len(problems) < 24:
                    problems.append(f'{shard["path"]}@{tick} side{side}: label_known={known} but the raw window '
                                    f'has {len(commands)} commands (complete={complete}, '
                                    f'accepted={[c.accepted for c in commands]}, reason={reason})')
            if not known:
                continue
            a_labelled += 1
            if not commands:
                wait_labels += 1
            shadow = ShadowLegality(batch)
            for order in range(int(action.count[0])):
                uid = int(action.candidate_uid[0, order])
                target = int(action.target_cell[0, order])
                delay = int(action.offset_bin[0, order])
                matches = (batch.candidate_uids[0] == uid) & shadow.candidate_mask(order)[0]
                if int(matches.sum()) != 1:
                    problems.append(f'{shard["path"]}@{tick}: labelled candidate {uid} '
                                    f'not uniquely legal at step {order}')
                    break
                index = int(matches.nonzero()[0, 0])
                selected = torch.tensor([index])
                if bool(batch.grid_targets[0, index]) != (target >= 0):
                    problems.append(f'{shard["path"]}@{tick}: labelled target mode mismatch for {uid}')
                    break
                if target >= 0 and not bool(shadow.placement(selected)[0, target]):
                    problems.append(f'{shard["path"]}@{tick}: labelled target cell {target} '
                                    f'not placeable for {uid}')
                    break
                if not bool(shadow.offset_mask(order)[0, delay]):
                    problems.append(f'{shard["path"]}@{tick}: labelled delay {delay} illegal at step {order}')
                    break
                shadow.apply(torch.tensor([True]), selected, torch.tensor([target]), torch.tensor([delay]))
                cited += 1
        del sequence  # only the current shard is ever held

    # ===================== tails (verbatim) =====================
    coverage = json.loads((dataset_dir / 'coverage.json').read_text(encoding='utf-8'))
    expected_samples = coverage.get('retained_samples', coverage['actor_samples'])
    if samples != expected_samples:
        raise ValueError(f'loader read {samples} samples but the coverage report accounts for {expected_samples} retained')
    excluded = int(coverage.get('excluded_samples', 0))
    if coverage['actor_samples'] != expected_samples + excluded:
        raise ValueError('excluded sample accounting does not add up to the actor samples')
    if excluded and not coverage.get('exclusion_reasons'):
        raise ValueError('excluded samples were not attributed to a reason')
    input_report = dict(passed=samples > 0, samples=samples, expected_samples=expected_samples,
                        actor_samples=coverage['actor_samples'], excluded_samples=excluded,
                        exclusion_reasons=coverage.get('exclusion_reasons', {}), shards=sequences,
                        known_bits_checked=known_bits, labelled_decisions=labelled,
                        allowance='strict model, allow_incomplete_capture=False')

    counts = manifest.get('label_counts') or {}
    accounted = int(counts.get('action', 0)) + int(counts.get('observed_wait', 0))
    if accounted != a_labelled:
        problems.append(f'label accounting: manifest says {accounted} labelled, shards hold {a_labelled}')
    offsets_ok = not any('invalid action sequence' in p or 'is not a legal offset' in p for p in problems)
    alignment_ok = not any('not uniquely legal' in p or 'target mode mismatch' in p
                           or 'not placeable' in p or 'illegal at step' in p for p in problems)
    action_report = dict(
        passed=bool(not problems and battle_level_split and not readback_failures),
        samples=a_samples, labelled_samples=a_labelled, cited_micro_actions=cited,
        observed_wait_labels=wait_labels, manifest_labelled=accounted,
        window_mismatches=accounted_mismatch, label_reasons=dict(counts),
        end_tick=end_tick, bad_ticks=len(bad_ticks), shards=len(shards), partitions=sorted(partitions),
        battle_level_split=battle_level_split, shard_readback=not readback_failures,
        readback_failures=readback_failures[:8], action_offsets=offsets_ok,
        legal_candidate_alignment=alignment_ok,
        rejected_action_loss_masks=bool(accounted == a_labelled and not accounted_mismatch),
        problems=problems[:16], problem_count=len(problems),
        scope='action labels re-derived from the stored shards and the raw command stream')

    return input_report, action_report


def strict_input_and_action_verification(dataset_dir, capture_dir, contract, device='cuda', max_ticks=0):
    """Fused entry point. Returns (input_report, action_report) shaped exactly like the
    two originals. Falls back to the two-pass path when asked to, so the caller can
    A/B the two on the same dataset without changing any check."""
    return _fused(dataset_dir, capture_dir, contract, device=device, max_ticks=max_ticks)
