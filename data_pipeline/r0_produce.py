"""Production data path: capture a source battle with the frozen bundle, gate it
on the necessary checks only, compile it, and verify the tensors are usable.

Per battle the stages are: capture -> necessary checks -> compile -> input check.
A battle that fails a stage is quarantined with its reason and is not retried by
default; the ledger makes the run resumable and the yield countable.

Necessary checks (deliberately limited, per the production plan):
  identity/ruleset  the run really used the authoritative processing contract
  continuity        frames, damage sequence and lifecycle sequence have no holes
  attribution       no effect generation is left without a native lineage
  labels            rejected/unresolved actions are masked, not invented
  tensors           finite, known bits in {0,1}, complete lifecycle ledger
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from .prepare_600k import atomic_json, digest


@dataclass
class Outcome:
    battle_tag: str
    status: str = 'pending'
    stage: str = 'none'
    reason: str | None = None
    seconds: float = 0.0
    stage_seconds: dict = field(default_factory=dict)
    frames: int | None = None
    last_tick: int | None = None
    terminal: bool | None = None
    samples: int | None = None
    shards: int | None = None
    raw_sha256: str | None = None
    capture_mode: str | None = None
    gamemode: int | None = None
    king_level: int | None = None
    lineage_unknown_units: int | None = None
    lineage_damaging: list = field(default_factory=list)
    lineage_silent: int | None = None
    command_counts: dict = field(default_factory=dict)

    def line(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)


def _load_ledger(path: Path) -> dict:
    seen = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen[record['battle_tag']] = record
    return seen


def _append_ledger(path: Path, outcome: Outcome):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(outcome.line() + '\n')


def capture_stage(record, target: Path, *, host, port, seed, bundle, downloader_root, maximum_seeds,
                  capture_masks=True, ability_policy='newest-eligible', attempts=4):
    """Capture one battle into `target` (atomic) using the frozen bundle.

    A worker closes its port while it reloads the native process between battles,
    so a capture that starts in that window fails immediately with a connection
    error. Retrying the same port a few times turns that transient infrastructure
    state into a short wait instead of a lost battle.
    """
    import shutil
    import time
    from .pilot100 import capture_one
    transient = ('ConnectionRefusedError', 'ConnectionResetError', 'ConnectionAbortedError',
                 'ConnectionError', 'TimeoutError')
    temporary = target.with_name(target.name + '.partial')
    target.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, max(1, attempts) + 1):
        if temporary.exists():
            shutil.rmtree(temporary)
        started = time.perf_counter()
        report = capture_one(Path(record['source_file']), temporary, host, port, seed, bundle,
                             ability_policy, True, maximum_seeds, True, downloader_root, capture_masks, None, False)
        elapsed = time.perf_counter() - started
        last = (report, elapsed)
        error = str(report.get('error') or '')
        retryable = report.get('status') == 'capture_failed' and any(name in error for name in transient)
        if not retryable or attempt == attempts:
            break
        print(f'  {record["battle_tag"]}: port {port} unavailable ({error[:90]}); '
              f'retrying ({attempt}/{attempts - 1})', flush=True)
        time.sleep(15 * attempt)
    report, elapsed = last
    # Only a real capture may be published. A failed attempt leaves a directory
    # holding a report but no frame stream; if that were renamed into place, every
    # later run would read report.json, skip the capture, and fail in the compiler
    # with a missing-file error instead of retrying the capture.
    if report.get('status') not in ('native_terminal', 'source_duration_reached'):
        if temporary.exists():
            shutil.rmtree(temporary)
        return report, elapsed
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)
    return report, elapsed


def episode_row(capture_dir: Path) -> dict:
    """Read only the episode line: identity and ruleset need no full stream pass."""
    from training_r0.replay_adapter import raw_capture_lines
    with raw_capture_lines(Path(capture_dir) / 'native-frames.jsonl.zst') as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get('kind') == 'episode':
                return row
    raise ValueError('raw capture has no episode record')


def gate_from_compiled(capture_dir: Path, dataset_dir: Path, bundle_manifest_path: Path) -> tuple[bool, dict, str | None]:
    """Necessary checks taken from the compile's own coverage report.

    The compiler already walks the whole stream once and records continuity,
    capability coverage and lineage issues there, so the gate does not repeat
    that parse. Only the episode line is read again, for identity and ruleset.
    """
    from .r0_certify import verify_raw_checks
    coverage = json.loads((Path(dataset_dir) / 'coverage.json').read_text(encoding='utf-8'))
    episode = episode_row(capture_dir)
    facts = dict(episode=episode, identity=episode.get('identity') or {},
                 processing=json.loads((Path(capture_dir) / 'processing-input.json').read_text(encoding='utf-8'))
                 if (Path(capture_dir) / 'processing-input.json').is_file() else None,
                 capture_dir=str(Path(capture_dir).resolve()),
                 raw_sha256=None, report_sha256=None)
    bundle = json.loads(Path(bundle_manifest_path).read_text(encoding='utf-8'))
    worker = {entry['name']: entry['sha256'] for entry in bundle.get('worker_files', ())}
    identity = facts['identity']
    identity_ok = bool(bundle.get('libg_sha256') == identity.get('libg_sha256')
                       and worker.get('libg.so') == identity.get('libg_sha256')
                       and worker.get('lifecycle-probe.jar') == identity.get('host_sha256')
                       and worker.get('libnative_host_bridge.so') == identity.get('bridge_sha256')
                       and identity.get('elite_telemetry') == 'x86-v1')
    replay_battle = ((episode.get('replay') or {}).get('battle') or {})
    plan_path = Path(capture_dir) / 'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.is_file() else {}
    expected = (facts['processing'] or {}).get('native_execution_game_mode_id')
    ruleset_ok = bool(episode.get('capture_mode') == 'production_processing_contract'
                      and episode.get('native_replay_ready') is True
                      and expected is not None
                      and replay_battle.get('gamemode') == expected == plan.get('native_execution_game_mode_id')
                      and plan.get('replay_tier') == 'native_replay_candidate')
    samples = coverage['actor_samples']
    counts = coverage['capability_sample_counts']
    # capability_sample_counts holds the samples each domain WAS captured for, so
    # the shortfall is the difference, not the value itself.
    shortfall = {name: samples - value for name, value in counts.items() if value < samples}
    retained = int(coverage.get('retained_samples', samples))
    excluded = int(coverage.get('excluded_samples', 0))
    failures = []
    if not identity_ok:
        failures.append('native_identity')
    if not ruleset_ok:
        failures.append('ruleset_resolved')
    # A persistent capture failure is recorded, not fatal on its own. It is sticky
    # from the first bad event onward, so everything it invalidates is already
    # excluded sample by sample: a sample is only stored when its capability set is
    # complete, and a complete sample implies the attribution ledger was still
    # intact at that tick. Failing the whole battle here would discard the sound
    # prefix that the sample-level exclusion exists to keep.
    capture_failures = list(coverage.get('persistent_capture_failures') or ())
    # Samples with an incomplete capture cannot enter a production batch, so the
    # compiler excludes them and records why. The battle is only useful if at
    # least one complete sample survives; a battle whose attribution is broken
    # everywhere keeps none and is rejected here rather than at load time.
    if retained <= 0:
        failures.append('no_complete_samples')
    # The strict model also refuses a sample whose entity archetype, group or kind
    # could not be resolved (elite_tensorizer capture_complete), so the gate has to
    # reject it here instead of letting it fail later at load time.
    issues = coverage.get('issues') or {}
    unresolved = sorted(key for key in issues
                        if key.startswith('archetype_unresolved') or key.startswith('entity_details_missing'))
    if unresolved:
        failures.append('unresolved_entity:' + ','.join(unresolved[:4]))
    detail = dict(identity_ok=identity_ok, ruleset_ok=ruleset_ok, samples=samples,
                  retained_samples=retained, excluded_samples=excluded,
                  retained_fraction=round(retained / samples, 6) if samples else 0.0,
                  exclusion_reasons=coverage.get('exclusion_reasons') or {},
                  capability_shortfalls=shortfall,
                  persistent_capture_failures=capture_failures,
                  last_capability_failures=coverage.get('last_capability_failures'),
                  issues=coverage.get('issues'), observed_gamemode=replay_battle.get('gamemode'),
                  expected_gamemode=expected)
    return (not failures), detail, ('; '.join(failures) if failures else None)


def gate_capture(capture_dir: Path, bundle_manifest_path: Path, contract, *, strict_lineage=False
                 ) -> tuple[bool, dict, str | None]:
    """The necessary checks on a raw capture.

    A generation without native lineage only blocks the battle when it actually
    dealt damage: that is the case that can leave a damage receipt uncredited and
    suspend the lifecycle ledger. Silent lineage-less effects are recorded.
    """
    from .r0_certify import stream_facts, verify_raw_checks
    dataset_stub = capture_dir.parent.parent  # only used for artifacts the gate reads itself
    facts = stream_facts(capture_dir)
    checks = verify_raw_checks(facts, contract, dataset_stub, bundle_manifest_path)
    failures = [name for name in ('native_identity', 'ruleset_resolved', 'continuous_frames',
                                  'continuous_events', 'entity_generation_identity')
                if not checks[name]['passed']]
    sources = {int(key): value for key, value in facts['damage_sources'].items()}
    silent, damaging = [], []
    for row in facts['effects_without_lineage']:
        total = sources.get(row['key'], 0.0)
        (damaging if total > 0 else silent).append(dict(row, damage=round(total, 1)))
    if damaging or (strict_lineage and silent):
        failures.append('effect_without_lineage_damage')
    detail = dict(checks=checks, damaging_without_lineage=damaging[:8], silent_without_lineage=silent[:8],
                  units_without_lineage=facts['units_without_lineage'][:8],
                  frame_count=facts['frame_count'], last_tick=facts['last_tick'],
                  damage_source_count=len(sources), strict_lineage=strict_lineage)
    return (not failures), detail, ('; '.join(failures) if failures else None)


def compile_stage(capture_dir: Path, target: Path, *, sequence_steps: int, max_tick: int | None = None):
    from .r0_compile import compile_capture
    if target.exists():
        import shutil
        shutil.rmtree(target)
    started = time.perf_counter()
    manifest = compile_capture(capture_dir, target, sequence_steps=sequence_steps, max_tick=max_tick)
    return manifest, time.perf_counter() - started


def input_check(dataset_dir: Path, contract, device='cuda', capture_dir: Path | None = None) -> tuple[bool, dict, str | None]:
    """Tensors usable: finite, known bits, complete damage ledger, labels legal.

    The authoritative checks are the strict passes below: the loader reads every
    stored sample back through the official entry and runs the production batch
    contract on it, and the action verifier re-derives the label rule from the
    stored shards plus the raw command stream. Together they cover the necessary
    conditions -- continuity, sound attribution, no future information, legal
    action labels and finite tensors -- rather than trusting the compiler.
    The coverage counters are only an invariant assertion, and they are compared
    against the samples actually retained instead of every sample the capture
    produced, because incomplete samples are excluded by the compiler and never
    reach a batch.
    """
    from .r0_certify import strict_action_verification, strict_input_verification
    from training_r0.elite_tensorizer import REQUIRED_CAPTURE
    coverage = json.loads((dataset_dir / 'coverage.json').read_text(encoding='utf-8'))
    counts = coverage['capability_sample_counts']
    actor = coverage['actor_samples']
    samples = int(coverage.get('retained_samples', actor))
    problems = []
    if samples <= 0:
        problems.append('no_complete_samples')
    # A duplicate-Tick resample can account for a capability more than once, so the
    # requirement is coverage of every retained sample, not an exact equality.
    for name in REQUIRED_CAPTURE:
        if counts.get(name, 0) < samples:
            problems.append(f'{name}_incomplete:{counts.get(name, 0)}/{samples}')
    started = time.perf_counter()
    # R0_FUSED_VERIFY=1 runs both strict passes over ONE traversal of the shards.
    # They each load_sequence every shard, so the default two-pass path reads and
    # objectifies ~50 npz per battle twice. The fused path keeps every check; it is
    # opt-in so the two can be A/B'd on the same dataset and the old path stays the
    # control.
    if capture_dir is not None and os.environ.get('R0_FUSED_VERIFY', '0') == '1':
        from .fused_verify import strict_input_and_action_verification
        verified, actions = strict_input_and_action_verification(dataset_dir, capture_dir, contract,
                                                                 device=device)
    else:
        verified = strict_input_verification(dataset_dir, contract, device=device)
        actions = strict_action_verification(dataset_dir, capture_dir, contract) if capture_dir else None
    if actions is not None:
        for name in ('action_offsets', 'legal_candidate_alignment', 'rejected_action_loss_masks',
                     'battle_level_split', 'shard_readback'):
            if not actions[name]:
                problems.append(f'{name}:' + ';'.join(actions['problems'][:2]))
    seconds = time.perf_counter() - started
    detail = dict(coverage_sample_counts=counts, actor_samples=actor, retained_samples=samples,
                  excluded_samples=int(coverage.get('excluded_samples', 0)),
                  exclusion_reasons=coverage.get('exclusion_reasons') or {},
                  persistent_capture_failures=list(coverage.get('persistent_capture_failures') or ()),
                  strict=verified, actions=actions, problems=problems)
    return (not problems), detail, ('; '.join(problems) if problems else None)


def process_one(record, *, output, bundle_manifest_path, bundle, contract, host, port, seed,
                downloader_root, attempt, compile_attempt, device, maximum_seeds, sequence_steps,
                max_tick=None, force=False, strict_lineage=False, post_semaphore=None, stage='all') -> Outcome:
    tag = record['battle_tag']
    outcome = Outcome(battle_tag=tag)
    started = time.perf_counter()
    capture_dir = Path(output) / attempt / tag
    dataset_dir = Path(output) / compile_attempt / tag
    try:
        if stage == 'post' and not (capture_dir / 'report.json').is_file():
            outcome.status, outcome.stage, outcome.reason = 'skipped', 'post', 'no_capture'
            return outcome
        if stage in ('all', 'capture') and (force or not (capture_dir / 'report.json').is_file()):
            # Static eligibility costs milliseconds and decides whether a native
            # capture of this source can ever be admitted. Screen first and record
            # the reason, instead of spending a worker on a battle the pinned
            # native contract will refuse (and reporting it as a capture failure).
            from .r0_plan_survey import survey_source
            eligible, reason, screen = survey_source(Path(record['source_file']), Path(downloader_root))
            if not eligible:
                reasons = screen.get('reasons') or [reason or 'unknown']
                outcome.status, outcome.stage = 'quarantined', 'static_eligibility'
                outcome.reason = 'static_eligibility:' + ','.join(reasons)[:200]
                return outcome
            outcome.stage = 'capture'
            report, seconds = capture_stage(record, capture_dir, host=host, port=port, seed=seed,
                                            bundle=bundle, downloader_root=downloader_root,
                                            maximum_seeds=maximum_seeds)
            outcome.stage_seconds['capture'] = round(seconds, 2)
            outcome.frames = report.get('frames')
            outcome.last_tick = report.get('last_tick')
            outcome.command_counts = report.get('command_counts', {})
            if report.get('status') not in ('native_terminal', 'source_duration_reached'):
                outcome.status, outcome.reason = 'quarantined', f"capture_status:{report.get('status')}:{report.get('error')}"
                return outcome
            outcome.terminal = report.get('status') == 'native_terminal'
        else:
            report = json.loads((capture_dir / 'report.json').read_text(encoding='utf-8'))
            outcome.frames = report.get('frames')
            outcome.last_tick = report.get('last_tick')
            outcome.command_counts = report.get('command_counts', {})
            outcome.stage_seconds['capture'] = 0.0
        if stage == 'capture':
            outcome.status, outcome.stage = 'captured', 'capture'
            return outcome

        outcome.stage = 'gate'
        _post_guard = post_semaphore if post_semaphore is not None else nullcontext()
        with _post_guard:
            outcome.stage = 'compile'
            manifest, seconds = compile_stage(capture_dir, dataset_dir, sequence_steps=sequence_steps,
                                              max_tick=max_tick)
            outcome.stage_seconds['compile'] = round(seconds, 2)
            outcome.shards = len(manifest.get('shards', []))

            # The gate reads the compiler's own report instead of parsing the
            # raw stream a second time.
            passed, detail, reason = gate_from_compiled(capture_dir, dataset_dir, bundle_manifest_path)
            outcome.capture_mode = 'production_processing_contract' if detail['ruleset_ok'] else None
            outcome.gamemode = detail.get('observed_gamemode')
            outcome.samples = detail.get('samples')
            outcome.lineage_damaging = []
            if not passed:
                atomic_json(Path(output) / 'gate-detail' / f'{tag}.json', detail)
                outcome.status, outcome.reason = 'quarantined', f'gate:{reason}'
                return outcome

            outcome.stage = 'input_check'
            usable, detail, reason = input_check(dataset_dir, contract, device=device, capture_dir=capture_dir)
            outcome.samples = detail['actor_samples']
            if not usable:
                outcome.status, outcome.reason = 'quarantined', f'input_check:{reason}'
                return outcome
        outcome.status = 'usable'
    except Exception as error:  # one bad battle must not stop the queue
        outcome.status = 'quarantined'
        outcome.reason = f'{outcome.stage}:{type(error).__name__}:{str(error)[:200]}'
    finally:
        outcome.seconds = round(time.perf_counter() - started, 2)
    return outcome


def run(args) -> dict:
    from .r0_admission import current_contract
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    records = selection['records']
    if args.tags:
        wanted = set(args.tags)
        records = [r for r in records if r['battle_tag'] in wanted]
    if args.limit:
        records = records[:args.limit]
    # The compile stage is CPU-bound Python, so thread-based post workers inside one
    # process serialise on the GIL no matter how many are configured. Splitting the
    # selection into independent processes (each with its own output and ledger) is
    # what actually parallelises it; the stride keeps the split deterministic.
    #
    # --claim-ledger replaces the static stride with a rolling claim: producers keep
    # pulling the next battle until the queue is empty, so a fast producer never idles
    # while a slow one finishes its whole share (measured tail: at 1997/2000 only 2 of
    # 96 producers were still working). The ledger lives on NVMe because a state file
    # on tmpfs is lost together with the data it describes.
    claim_path = Path(args.claim_ledger) if getattr(args, 'claim_ledger', None) else None
    worker_id = getattr(args, 'worker_id', None) or f'pid{os.getpid()}'
    if claim_path is not None:
        from .claim_ledger import init_ledger
        queue_state = init_ledger(claim_path, records)
        print(f'claim ledger {claim_path}: total={queue_state["total"]} '
              f'added={queue_state["added"]} states={queue_state["states"]} worker={worker_id}', flush=True)
        records = []          # work is pulled from the queue, not from a fixed slice
    elif args.shard_count and args.shard_count > 1:
        if not 0 <= args.shard_index < args.shard_count:
            raise SystemExit('--shard-index must be in [0, --shard-count)')
        records = records[args.shard_index::args.shard_count]
        print(f'shard {args.shard_index}/{args.shard_count}: {len(records)} sources', flush=True)
    contract = current_contract()
    bundle = json.loads(Path(args.bundle_manifest).read_text(encoding='utf-8'))
    ledger_path = Path(args.ledger)
    ledger = _load_ledger(ledger_path)
    output = Path(args.output)
    (output / 'quarantine').mkdir(parents=True, exist_ok=True)
    (output / 'gate-detail').mkdir(parents=True, exist_ok=True)
    # Two producers sharing one output directory fight over the same worker ports
    # and rewrite each other's directories, which surfaces as spurious connection
    # failures and missing-file errors rather than as a clear conflict. A lock left
    # behind by a process that was killed never runs its own cleanup, so a lock
    # whose owner is gone is taken over instead of blocking every later run.
    import atexit
    # `os` is imported at module scope on purpose: a local `import os` here made the
    # name function-scoped, so the earlier worker_id line raised UnboundLocalError and
    # every producer died in ~2s while the loop happily advanced 97 batches.
    import re as _re
    # Named distinctly from the threading.Lock below: sharing the name made the
    # exit handler close over the thread lock and silently fail to release this one.
    produce_lock = output / 'produce.lock'

    def _lock_owner_alive(holder: str) -> bool:
        match = _re.search(r'pid=(\d+)', holder)
        if not match:
            return False
        try:
            os.kill(int(match.group(1)), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    descriptor = None
    for _attempt in range(3):
        try:
            descriptor = os.open(produce_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            holder = produce_lock.read_text(encoding='utf-8', errors='replace').strip().replace('\n', ' ')
            if _lock_owner_alive(holder):
                raise SystemExit(f'another producer owns {output} (produce.lock: {holder})')
            print(f'taking over stale {produce_lock} from a process that is gone ({holder})', flush=True)
            produce_lock.unlink(missing_ok=True)
    if descriptor is None:
        raise SystemExit(f'could not acquire {produce_lock}')
    os.write(descriptor, f'pid={os.getpid()} started={time.time():.0f}\n'.encode())
    os.close(descriptor)
    atexit.register(lambda: produce_lock.unlink(missing_ok=True))
    ports = Queue()
    for port in range(args.port, args.port + args.workers):
        ports.put(port)
    results = []
    lock = __import__('threading').Lock()
    post = __import__('threading').Semaphore(max(1, args.post_workers))

    def one(index, record, claim_tag=None):
        tag = record['battle_tag']

        def settle(outcome_dict):
            # Mark the claim exactly once, whatever path produced the result, so a
            # crashed producer's battle is recoverable rather than stuck RUNNING.
            if claim_tag is not None:
                from .claim_ledger import finish as _finish
                try:
                    _finish(claim_path, claim_tag, outcome_dict['status'], outcome_dict)
                except Exception as error:            # never lose a finished battle to bookkeeping
                    print(f'[{index}] claim settle failed for {claim_tag}: {error}', flush=True)
            return outcome_dict

        previous = ledger.get(tag)
        if previous and previous['status'] == 'usable' and not args.force:
            return settle(dict(previous, skipped=True))
        if previous and previous['status'] == 'quarantined' and not args.retry_quarantined:
            return settle(dict(previous, skipped=True))
        port = ports.get()
        try:
            outcome = process_one(record, output=output, bundle_manifest_path=Path(args.bundle_manifest),
                                  bundle=bundle, contract=contract, host=args.host, port=port,
                                  seed=args.seed, downloader_root=Path(args.downloader_root),
                                  attempt=args.attempt, compile_attempt=args.compile_attempt,
                                  device=args.device, maximum_seeds=args.maximum_seeds,
                                  sequence_steps=args.sequence_steps, max_tick=args.max_tick,
                                  force=args.force, strict_lineage=args.strict_lineage,
                                  post_semaphore=post, stage=args.stage)
        finally:
            ports.put(port)
        with lock:
            _append_ledger(ledger_path, outcome)
            ledger[tag] = outcome.__dict__
            if outcome.status == 'quarantined':
                atomic_json(output / 'quarantine' / f'{tag}.json',
                            dict(battle_tag=tag, stage=outcome.stage, reason=outcome.reason,
                                 seconds=outcome.seconds, frames=outcome.frames))
            print(f'[{index}/{len(records)}] {tag}: {outcome.status} (port {port}, {outcome.stage}, '
                  f'{outcome.seconds}s, frames={outcome.frames}, samples={outcome.samples})'
                  + (f' reason={outcome.reason}' if outcome.reason else ''), flush=True)
        return settle(outcome.__dict__)

    if getattr(args, 'pipeline_backlog', 0) > 0 and args.stage == 'all':
        from .bounded_pipeline import two_stage
        def process_stage(record, port, stage):
            return process_one(record, output=output, bundle_manifest_path=Path(args.bundle_manifest),
                bundle=bundle, contract=contract, host=args.host, port=port, seed=args.seed,
                downloader_root=Path(args.downloader_root), attempt=args.attempt,
                compile_attempt=args.compile_attempt, device=args.device, maximum_seeds=args.maximum_seeds,
                sequence_steps=args.sequence_steps, max_tick=args.max_tick, force=args.force,
                strict_lineage=args.strict_lineage, post_semaphore=post, stage=stage)

        def collect(item):
            index,record=item;tag=record['battle_tag'];previous=ledger.get(tag)
            if previous and not args.force and (previous['status']=='usable' or
                    previous['status']=='quarantined' and not args.retry_quarantined):
                return dict(previous,skipped=True)
            port=ports.get()
            try:return process_stage(record,port,'capture')
            finally:ports.put(port)  # Release before compile/verification begins.

        def finish(item,captured):
            _,record=item;outcome=process_stage(record,None,'post')
            outcome.stage_seconds['capture']=captured.stage_seconds.get('capture',0.)
            outcome.terminal=captured.terminal
            outcome.seconds=round(captured.seconds+outcome.seconds,2)
            return outcome

        for (index,record),captured,finished in two_stage(enumerate(records,1),collect,finish,
                capture_workers=args.workers,post_workers=max(1,args.post_workers),
                backlog=args.pipeline_backlog,
                proceed=lambda result:isinstance(result,Outcome) and result.status=='captured'):
            outcome=finished if finished is not None else captured
            if isinstance(outcome,dict):results.append(outcome);continue
            with lock:
                _append_ledger(ledger_path,outcome);ledger[record['battle_tag']]=outcome.__dict__
                if outcome.status=='quarantined':
                    atomic_json(output/'quarantine'/f'{outcome.battle_tag}.json',
                        dict(battle_tag=outcome.battle_tag,stage=outcome.stage,reason=outcome.reason,
                             seconds=outcome.seconds,frames=outcome.frames))
            results.append(outcome.__dict__)
            print(f'[{index}/{len(records)}] {outcome.battle_tag}: {outcome.status} '
                  f'(pipeline, {outcome.seconds}s, samples={outcome.samples})',flush=True)
    elif claim_path is not None:
        # Rolling claim: keep `args.workers` battles in flight and pull the next one the
        # moment any finishes, until the queue is empty. The tail is then bounded by one
        # battle instead of by the slowest producer's entire share.
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        from .claim_ledger import claim_next
        lease = float(getattr(args, 'claim_lease', 1800.0))
        inflight = {}
        index = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while True:
                while len(inflight) < args.workers:
                    claim = claim_next(claim_path, worker_id, lease_seconds=lease)
                    if claim is None:
                        break
                    index += 1
                    record = {'battle_tag': claim['tag'], 'source_file': claim['source_file']}
                    inflight[pool.submit(one, index, record, claim['tag'])] = claim['tag']
                if not inflight:
                    break
                done, _pending = wait(list(inflight), return_when=FIRST_COMPLETED)
                for future in done:
                    inflight.pop(future, None)
                    results.append(future.result())
        print(f'claim pool drained: this producer handled {index} battles', flush=True)
    elif args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(one, index, record) for index, record in enumerate(records, 1)]
            for future in futures:
                results.append(future.result())
    else:
        for index, record in enumerate(records, 1):
            results.append(one(index, record))
    summary = aggregate(results)
    atomic_json(output / 'yield.json', summary)
    (output / 'ledger-copy.jsonl').write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in results) + '\n',
        encoding='utf-8')
    return summary


def aggregate(results: list) -> dict:
    from collections import Counter
    statuses = Counter(row['status'] for row in results)
    reasons = Counter((row.get('reason') or '').split(':')[0] + ':' + (row.get('reason') or '').split(':')[1]
                      if row.get('reason') and ':' in row['reason'] else (row.get('reason') or 'none')
                      for row in results if row['status'] != 'usable')
    usable = [row for row in results if row['status'] == 'usable']
    seconds = [row['seconds'] for row in results if row.get('seconds')]
    return dict(
        schema='r0-production-yield.v1', attempted=len(results),
        usable=statuses.get('usable', 0), quarantined=statuses.get('quarantined', 0),
        yield_rate=round(statuses.get('usable', 0) / len(results), 4) if results else None,
        statuses=dict(statuses), quarantine_reasons=dict(reasons),
        seconds_total=round(sum(seconds), 1),
        seconds_per_battle_mean=round(sum(seconds) / len(seconds), 1) if seconds else None,
        samples_total=sum(row.get('samples') or 0 for row in usable),
        shards_total=sum(row.get('shards') or 0 for row in usable),
        battles=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle-manifest', type=Path, required=True)
    parser.add_argument('--downloader-root', type=Path, default=Path(r'D:/Deepseek/work/royaleapi-downloader'))
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=39531)
    parser.add_argument('--workers', type=int, default=1,
                        help='capture this many battles concurrently across --port..--port+workers-1')
    parser.add_argument('--post-workers', type=int, default=2,
                        help='how many battles may run gate/compile/input-check concurrently')
    parser.add_argument('--pipeline-backlog', type=int, default=0,
                        help='opt-in bounded capture/post overlap; 0 preserves existing scheduling')
    parser.add_argument('--claim-ledger', type=Path, default=None,
                        help='SQLite queue on NVMe; when set, producers pull work by atomic claim '
                             'instead of taking a static --shard-index/--shard-count slice')
    parser.add_argument('--worker-id', default=None,
                        help='identity recorded on each claim; defaults to pid<N>')
    parser.add_argument('--claim-lease', type=float, default=1800.0,
                        help='seconds before a RUNNING claim is considered abandoned')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--maximum-seeds', type=int, default=512)
    parser.add_argument('--sequence-steps', type=int, default=16)
    parser.add_argument('--max-tick', type=int)
    parser.add_argument('--attempt', default='capture-prod-v17')
    parser.add_argument('--compile-attempt', default='compiled-prod-v17')
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--tags', nargs='*')
    parser.add_argument('--shard-index', type=int, default=0,
                        help='process only records where index %% --shard-count == this value')
    parser.add_argument('--shard-count', type=int, default=1,
                        help='split the selection across this many independent producer processes')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--retry-quarantined', action='store_true')
    parser.add_argument('--stage', choices=('all', 'capture', 'post'), default='all',
                        help='two-phase production: capture first, then gate/compile/check')
    parser.add_argument('--strict-lineage', action='store_true',
                        help='also quarantine battles whose lineage-less effects never dealt damage')
    args = parser.parse_args()
    if args.pipeline_backlog < 0:
        parser.error('--pipeline-backlog must be nonnegative')
    if args.pipeline_backlog and args.stage != 'all':
        parser.error('--pipeline-backlog requires --stage all')
    summary = run(args)
    print(json.dumps({k: v for k, v in summary.items() if k != 'battles'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
