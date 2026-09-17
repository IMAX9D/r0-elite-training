# R0 Elite — Batched Imitation Trainer

Batched training pipeline for the R0 policy network, plus the CPU data backend that feeds it
from shared memory. This repository contains **only what is needed to train**: the model, the
dataset loader, the admission/contract logic, the batched trainer, and the data backend.

---

## 1. What this is

| Component | Role |
|---|---|
| `training_r0/` | model (`R0Policy`), observation/tensorizer, `ILSequence` dataset format, imitation update |
| `data_pipeline/` | `r0_admission` (contract + admission), `r0_train` (lane construction), supporting modules |
| `native_core/`, `expert_v1/`, `bindings/` | runtime dependencies imported by the above |
| `r0_train_batched.py` | the trainer: one loop, two data sources (`inline` and `backend`) |
| `r0_data_backend.py` | CPU data backend: one decode per shard, shared-memory slots, shard affinity |
| `verify_*.py` | correctness tests (B=1 equivalence, multi-lane, decode reuse, serialization) |
| `tools/` | data fetch + run helpers |

### The one thing to understand first

`current_contract()` hashes `training_r0/*.py` **and `training_r0/data/*.json`**, resolving those
paths **relative to the current working directory**. Two consequences:

1. **You must run from the repository root.** If `training_r0/` is not directly under the CWD, the
   hash comes out wrong (empty `source_hashes`), and every compiled battle will be rejected as
   "compiled under another contract" — which looks like a data problem but is a path problem.
2. **`training_r0/data/*.json` must not be edited.** They are part of the contract. Changing them
   invalidates every existing dataset.

The trainer prints a `RUN LOG` with the contract hash and the number of source hashes, and
**refuses to start if `contract.source_hashes` is empty**.

---

## 2. Environment

Verified working combination:

```
OS        Ubuntu 22.04 (glibc 2.35)
Python    3.12
PyTorch   2.8.0+cu128
CUDA      12.8 (build), driver >= 570
GPU       tested on RTX PRO 6000 Blackwell (sm_120, 95 GiB)
          also runs on any sm_80+ card; see the VRAM table below
```

Install:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`torch` must be a CUDA build. Verify before going further:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**Do not use CPU-only torch.** It imports fine but fails at the first GPU kernel with
`libcublasLt.so.* not found` or an sm-architecture error.

### VRAM required per batch size (measured, real data)

| Batch | Peak VRAM | Fits on |
|---:|---:|---|
| 256 | 12.5 GiB | 24 GB card |
| 512 | 23.7 GiB | 24 GB card (tight) |
| 1024 | 46.3 GiB | 48 GB+ |
| 2048 | 91.3 GiB | 96 GB card |

Peak is dominated by **activations**, not inputs: at B=2048 the raw input payload is only ~0.51 GiB
of the 91.3 GiB. Measured throughput on the reference GPU: **5,317 decisions/s at B=2048** (CUDA
Events, inputs preloaded).

---

## 3. Data

Training data is **not** in this repository (the full compiled corpus is ~772 GB). Per-batch
archives live on Baidu Netdisk. Each archive is one batch: `out-N/compiled-prod-v17/<battle>/…`
plus its `out-N/ledger-copy.jsonl`.

### Get credentials

The downloader reads an access token from `R0_XPAN_TOKEN` or `/root/.r0_xpan_token`:

```bash
export R0_XPAN_TOKEN='<your token>'
```

### List and download

```bash
python fetch_batch.py --list                 # what is on the netdisk
python fetch_batch.py --probe --batch 0      # measure speed, download 32 MiB only
python fetch_batch.py --batch 0 --dest /data/r0data
```

### Extract

`zstd` may be absent on minimal images; the Python path avoids that dependency:

```bash
python - <<'PY'
import pathlib, tarfile, zstandard
dest = pathlib.Path('/data/r0data')
archive = next(dest.glob('batch-*-compiled.tar.zst'))
out = dest / archive.name.replace('-compiled.tar.zst', '')
out.mkdir(parents=True, exist_ok=True)
with archive.open('rb') as fh:
    with tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(fh), mode='r|') as tar:
        tar.extractall(out, filter='data')
PY
```

### Build the root list

The dataset root must be **each** `out-N/compiled-prod-v17`, one per line — *not* the batch
directory. The loader indexes a root with `glob('*/manifest.json')`, i.e. one level down, so
passing the batch directory yields `usable battles: 1975 / training lanes: 0`.

```bash
find /data/r0data/batch-0 -maxdepth 2 -type d -name 'compiled-prod-v17' | sort > roots.txt
find /data/r0data/batch-0 -name 'ledger-copy.jsonl' | sort > ledgers.txt
```

---

## 4. Run training

**Both arguments are single flags followed by all paths.** Repeating `--dataset` does *not*
accumulate — argparse keeps only the last occurrence, which silently shrinks the corpus to one
`out-N` (this cost real debugging time).

```bash
cd <repo root>          # REQUIRED: contract paths are relative to CWD

python -u r0_train_batched.py \
  --dataset $(cat roots.txt | tr '\n' ' ') \
  --ledger  $(cat ledgers.txt | tr '\n' ' ') \
  --output /data/run/my-run \
  --batch-size 1024 --steps 8 --tbptt-steps 4 \
  --log-every 1
```

Add `--backend` to feed the loop from the shared-memory data backend instead of in-process loads:

```bash
python -u r0_train_batched.py \
  --dataset $(cat roots.txt | tr '\n' ' ') \
  --ledger  $(cat ledgers.txt | tr '\n' ' ') \
  --output /data/run/my-run-backend \
  --batch-size 1024 --steps 8 --tbptt-steps 4 \
  --backend --loaders 8 --slots 2 --log-every 1
```

Both forms enter **the same `run_training` loop**; only the data source differs. `--loaders 0`
disables the backend pool.

### Interpreting the output

`training_summary.json` splits time into `time_fuse_s`, `time_h2d_s`, `time_forward_s`,
`time_backward_opt_s` and `unaccounted_s` (data acquisition and lane bookkeeping). Measured at
B=1024 on the reference machine:

| Segment | Share |
|---|---:|
| data acquisition (in `unaccounted`) | 55% |
| fuse / padding | 29% |
| backward + optimizer | 10% |
| forward | 5% |
| H2D | **0.5%** |

**The bottleneck is CPU-side, not the GPU.** Do not assume H2D is the problem — measure it.

---

## 5. Verify correctness before trusting a run

```bash
# B=1: the batched path must reproduce update_imitation_sequence exactly
python verify_b1_equiv.py

# multi-lane: fused vs solo logits, padding transparency, slot isolation, decode count
python verify_multilane.py --lanes 4 --tbptt 4

# one decode per shard, four consumptions, bit-identical
python verify_reuse.py --lanes 4 --tbptt 4
```

Expected: B=1 loss **identical to 8 decimals**, gradients within ~1e-7 relative, multi-lane fused
vs solo logits within ~2e-6. These tests exist because three semantic bugs were found this way
(per-tick detach truncating TBPTT to 1, a spurious `1/tbptt_steps` on the loss, and backward
running per tick instead of per chunk).

---

## 6. Known traps

1. **`forced=` must be passed.** The model is trained by imitation: `model(obs, state, forced=expert_actions)`.
   Calling it with `forced=None` makes the model **sample**, and the reported loss becomes a random
   action's log-probability. It runs without error and looks plausible.
2. **`--dataset` does not accumulate** (see §4).
3. **Dataset root level** (see §3).
4. **Line endings.** Shell scripts generated on Windows need `sed -i 's/\r$//'` before use on Linux;
   otherwise bash reports `$'\r': command not found` and half-executes them.
5. **`np.frombuffer` views pin shared memory.** Reading a shared slot returns a view that keeps an
   exported pointer alive; `shm.close()` then raises `BufferError`. Copy with `bytes(...)` first.
6. **Multiple writers into one shared slot must not share a cursor.** Each worker writes a disjoint
   region, and the parent resets the index table once before dispatch. Two bugs of this shape
   produced silent corruption (one lane's features off by 5.6e-05) before being isolated.
7. **Process pools and CUDA.** A forked child inherits the parent's CUDA context and dies on first
   use, with no Python traceback (`BrokenProcessPool`). Use `spawn` for such pools; the backend in
   this repository is CPU-only by design and never touches CUDA.

---

## 7. Repository layout

```
training_r0/            model, tensorizers, dataset format, imitation update
  data/                 contract JSON — DO NOT EDIT
data_pipeline/          admission, contract, lane construction
native_core/            runtime dependency
expert_v1/              tick schema dependency
bindings/               runtime dependency
r0_train_batched.py     trainer (single loop, two data sources)
r0_data_backend.py      CPU data backend (shared slots, shard affinity, bounded prefetch)
verify_*.py             correctness tests
fetch_batch.py          netdisk downloader
tools/                  fetch/extract/count/run helpers
```

## License

See `LICENSE`.
