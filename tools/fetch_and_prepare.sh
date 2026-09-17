#!/usr/bin/env bash
# Extract a batch archive and build the root/ledger lists in one go.
# usage: bash tools/fetch_and_prepare.sh <dest-dir> <batch-index>
set -euo pipefail
DEST=${1:-/data/r0data}
BATCH=${2:-0}
cd "$(dirname "$0")/.."

python fetch_batch.py --batch "$BATCH" --dest "$DEST"

python - "$DEST" <<'PY'
import pathlib, sys, tarfile, zstandard
dest = pathlib.Path(sys.argv[1])
archive = sorted(dest.glob('batch-*-compiled.tar.zst'))[-1]
out = dest / archive.name.replace('-compiled.tar.zst', '')
out.mkdir(parents=True, exist_ok=True)
print(f'extracting {archive.name} -> {out}', flush=True)
with archive.open('rb') as fh:
    with tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(fh), mode='r|') as tar:
        tar.extractall(out, filter='data')
roots = sorted(p for p in out.glob('out-*/compiled-prod-v17') if p.is_dir())
ledgers = sorted(out.rglob('ledger-copy.jsonl'))
(dest / 'roots.txt').write_text('\n'.join(str(r) for r in roots) + '\n')
(dest / 'ledgers.txt').write_text('\n'.join(str(l) for l in ledgers) + '\n')
print(f'roots: {len(roots)}  ledgers: {len(ledgers)}')
print(f'wrote {dest}/roots.txt and {dest}/ledgers.txt')
PY
