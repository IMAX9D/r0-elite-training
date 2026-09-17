#!/usr/bin/env bash
# One-shot training from a batch directory.
# usage: [MODE=backend] [LOADERS=8] bash tools/run_training.sh <batch-dir> <output-dir> <batch> <steps>
set -euo pipefail
BATCH_DIR=${1:?usage: run_training.sh <batch-dir> <output-dir> <batch> <steps>}
OUT=${2:?}
B=${3:-1024}
S=${4:-8}
MODE=${MODE:-inline}
LOADERS=${LOADERS:-8}
SLOTS=${SLOTS:-2}
TBPTT=${TBPTT:-4}

# The contract is hashed relative to the CWD, so training must start from the repo root.
cd "$(dirname "$0")/.."
echo "repo root: $(pwd)"

ROOTS=$(find "$BATCH_DIR" -maxdepth 2 -type d -name 'compiled-prod-v17' | sort)
LEDGERS=$(find "$BATCH_DIR" -name 'ledger-copy.jsonl' | sort)
NROOTS=$(printf '%s\n' "$ROOTS" | grep -c . || true)
echo "dataset roots: $NROOTS"

if [ "$NROOTS" -eq 0 ]; then
  echo "ERROR: no compiled-prod-v17 under $BATCH_DIR" >&2
  echo "       the loader indexes each root one level down; pass the batch directory" >&2
  echo "       that contains out-N/compiled-prod-v17." >&2
  exit 2
fi

EXTRA=()
if [ "$MODE" = "backend" ]; then EXTRA=(--backend --loaders "$LOADERS" --slots "$SLOTS"); fi

# ONE --dataset followed by every root: repeating the flag would keep only the last one.
exec python -u r0_train_batched.py \
  --dataset $ROOTS \
  --ledger $LEDGERS \
  --output "$OUT" \
  --batch-size "$B" --steps "$S" --tbptt-steps "$TBPTT" \
  --log-every 1 "${EXTRA[@]}"
