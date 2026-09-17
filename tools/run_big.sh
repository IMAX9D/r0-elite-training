#!/usr/bin/env bash
# B=1024 on the downloaded batch, same entry point, same loop, 8 updates per configuration.
#
# THE DATASET ARGUMENTS ARE THE POINT: build_lanes() indexes each root with glob('*/manifest.json'),
# one level down, so every out-N/compiled-prod-v17 must be passed explicitly. Passing the batch
# directory yields "usable battles: 1975 / training lanes: 0" -- which is exactly what happened on
# the first attempt and would otherwise look like 1975 battles of unusable data.
set -u
cd /root
export PYTHONPATH=/root
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
ulimit -n 65535 2>/dev/null || true

PY=/root/miniconda3/bin/python
B=${B:-1024}
S=${S:-8}
MODE=${MODE:-backend}

mapfile -t ROOTS < /root/autodl-tmp/r0data/roots.txt
mapfile -t LEDGERS < /root/autodl-tmp/r0data/ledgers.txt
echo "roots=${#ROOTS[@]} ledgers=${#LEDGERS[@]} batch=$B steps=$S mode=$MODE"

# ONE --dataset followed by every root. Repeating the flag does not accumulate: argparse keeps
# the last occurrence only, which silently shrank the corpus to a single out-N.
DATASET_ARGS=(--dataset "${ROOTS[@]}")
LEDGER_ARGS=(--ledger "${LEDGERS[@]}")

EXTRA=()
if [ "$MODE" = "backend" ]; then
  EXTRA=(--backend --loaders "${LOADERS:-8}" --slots "${SLOTS:-2}")
fi

"$PY" -u /root/r0_train_batched.py \
  "${DATASET_ARGS[@]}" "${LEDGER_ARGS[@]}" \
  --output "/root/autodl-tmp/run/big-$MODE-B$B" \
  --batch-size "$B" --steps "$S" --tbptt-steps 4 \
  --log-every 1 "${EXTRA[@]}" 2>&1 | tail -40
echo "=== BIGRUN DONE $(date -u +%H:%M:%S) ==="
