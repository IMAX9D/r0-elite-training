#!/usr/bin/env bash
# The dataset root must be each out-N/compiled-prod-v17, not the batch directory: usable_battles()
# indexes roots with base.glob('*/manifest.json'), which only looks one level down.
set -u
cd /root
export PYTHONPATH=/root
PY=/root/miniconda3/bin/python

BATCH=${BATCH:-/root/autodl-tmp/r0data/batch-0}
mapfile -t ROOTS < <(find "$BATCH" -maxdepth 2 -type d -name 'compiled-prod-v17' | sort)
mapfile -t LEDGERS < <(find "$BATCH" -name 'ledger-copy.jsonl' | sort)
echo "roots: ${#ROOTS[@]}   ledgers: ${#LEDGERS[@]}"
echo "first root: ${ROOTS[0]}"

"$PY" - "$BATCH" <<'PY'
import sys, time, json
from pathlib import Path
sys.path.insert(0, '/root')
from data_pipeline.r0_admission import current_contract
from r0_train_batched import build_lanes

batch = Path(sys.argv[1])
roots = sorted(p for p in batch.glob('out-*/compiled-prod-v17') if p.is_dir())
ledgers = sorted(batch.rglob('ledger-copy.jsonl'))
print(f'  roots={len(roots)} ledgers={len(ledgers)}')
started = time.perf_counter()
lanes, counts, evaluated = build_lanes(roots, ledgers, current_contract())
elapsed = time.perf_counter() - started
print(f'  usable battles : {counts["usable"]}')
print(f'  training lanes : {len(lanes)}')
print(f'  datasets       : {len(evaluated)}')
print(f'  build time     : {elapsed:.1f}s')
print(f'  B=1024 : {"SUPPORTED" if len(lanes) >= 1024 else "NOT supported"}')
print(f'  B=2048 : {"SUPPORTED" if len(lanes) >= 2048 else "NOT supported"}')
if lanes:
    tag, dataset, key, shards = lanes[0]
    print(f'  sample lane: {tag} side={key[1]} shards={len(shards)}')
    print(f'    shard0: {shards[0]["path"]}')
Path('/root/autodl-tmp/r0data/roots.txt').write_text('\n'.join(str(r) for r in roots) + '\n')
Path('/root/autodl-tmp/r0data/ledgers.txt').write_text('\n'.join(str(l) for l in ledgers) + '\n')
print('  wrote roots.txt / ledgers.txt')
PY
echo "=== LANECOUNT DONE ==="
