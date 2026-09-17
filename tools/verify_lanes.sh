#!/usr/bin/env bash
# Confirm what the saved root/ledger lists contain and what build_lanes sees from them.
set -u
cd /root
export PYTHONPATH=/root
echo "=== list sizes ==="
wc -l /root/autodl-tmp/r0data/roots.txt /root/autodl-tmp/r0data/ledgers.txt
echo "first 3 roots:"; head -3 /root/autodl-tmp/r0data/roots.txt
echo
/root/miniconda3/bin/python - <<'PY'
import sys
sys.path.insert(0, '/root')
from pathlib import Path
from data_pipeline.r0_admission import current_contract
from r0_train_batched import build_lanes

roots = [Path(l.strip()) for l in open('/root/autodl-tmp/r0data/roots.txt') if l.strip()]
ledgers = [Path(l.strip()) for l in open('/root/autodl-tmp/r0data/ledgers.txt') if l.strip()]
print(f'roots={len(roots)} ledgers={len(ledgers)}')
print(f'  roots[0] exists: {roots[0].is_dir()}')
lanes, counts, evaluated = build_lanes(roots, ledgers, current_contract())
print(f'  usable battles : {counts["usable"]}')
print(f'  training lanes : {len(lanes)}')
print(f'  datasets       : {len(evaluated)}')
if lanes:
    print(f'  lanes[:3] tags : {[l[0] for l in lanes[:3]]}')
    print(f'  shard counts   : {[len(l[3]) for l in lanes[:6]]}')
    print(f'  min/max shards : {min(len(l[3]) for l in lanes)} / {max(len(l[3]) for l in lanes)}')
PY
echo "=== VERIFY DONE ==="
