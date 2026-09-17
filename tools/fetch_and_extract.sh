#!/usr/bin/env bash
# Pull one batch archive, extract it, and inventory the result.
#
# The archive holds out-N/compiled-prod-v17/<battle>/... plus the batch ledger, so after extraction
# the training host has the same shape of data the 40-battle slice already used -- just far more
# battles, which is what B=1024 needs.
set -u
PY=/root/miniconda3/bin/python
DEST=/root/autodl-tmp/r0data
BATCH=${BATCH:-0}
mkdir -p "$DEST"

echo "=== zstandard (zstd binary is absent on this image) ==="
"$PY" -m pip install --no-input -q zstandard --index-url https://mirrors.aliyun.com/pypi/simple 2>&1 | tail -2
"$PY" -c 'import zstandard; print("zstandard", zstandard.__version__)'

echo
echo "=== download batch-$BATCH ==="
"$PY" /root/fetch_batch.py --batch "$BATCH" --dest "$DEST"

echo
echo "=== extract ==="
"$PY" - <<'PY'
import pathlib, tarfile, time, zstandard
dest = pathlib.Path('/root/autodl-tmp/r0data')
archive = next(dest.glob('batch-*-compiled.tar.zst'))
out = dest / archive.name.replace('-compiled.tar.zst', '')
out.mkdir(parents=True, exist_ok=True)
print(f'archive {archive}  ({archive.stat().st_size/2**30:.2f} GiB) -> {out}', flush=True)
started = time.perf_counter()
with archive.open('rb') as handle:
    reader = zstandard.ZstdDecompressor().stream_reader(handle)
    with tarfile.open(fileobj=reader, mode='r|') as tar:
        tar.extractall(out, filter='data')
print(f'extracted in {time.perf_counter()-started:.0f}s', flush=True)
PY

echo
echo "=== inventory ==="
echo -n "compiled-prod-v17 dirs : "; find "$DEST" -maxdepth 3 -type d -name 'compiled-prod-v17' | wc -l
echo -n "battles (manifest.json): "; find "$DEST" -name 'manifest.json' | wc -l
echo -n "npz shards             : "; find "$DEST" -name '*.npz' | wc -l
echo -n "ledger files           : "; find "$DEST" -name 'ledger-copy.jsonl' | wc -l
echo "top level:"; ls "$DEST" | head -5
echo "disk:"; df -h /root/autodl-tmp / | tail -2
echo "=== FETCH DONE $(date -u +%H:%M:%S) ==="
