# R0 Elite — 全链路设计、编译与训练

这个仓库是 **R0 Elite 的完整链路**：从原始对局 → 采集 → 编译 → 认证 → 训练 → 正确性验证。
另一台设备克隆下来、装好依赖、拉一批数据，就能直接开始**改模型**和**跑训练**。

数据本身不在仓库里（全量编译语料约 772 GB），用 `fetch_batch.py` 从网盘按批拉取。

---

## 0. 全链路一览

```
原始回放 ──r0_produce.py──► capture (native-frames.jsonl.zst)
                                 │
                                 ├──gate──► 准入判定（usable / quarantined）
                                 │
                            r0_compile.py
                                 │
                                 ▼
                    compiled-prod-v17/<battle>/{manifest.json, contract.json, train/*.npz}
                                 │
                            r0_certify.py  （严格前向/反向认证，可选）
                                 │
                                 ▼
        ┌──────────── r0_train_batched.py ────────────┐
        │   同一个 run_training 循环，两种数据来源：      │
        │   inline（进程内加载） / backend（共享内存）    │
        └──────────────────────────────────────────────┘
                                 │
                    verify_b1_equiv.py / verify_multilane.py
                    verify_reuse.py / verify_serialization.py
```

**训练只依赖 `compiled-prod-v17`**。采集与编译是另一条独立链路，想用现成数据训练的话可以完全不碰它们。

---

## 1. 想改模型，只动这几个文件

| 你想改的东西 | 改哪里 |
|---|---|
| 网络结构（层数、宽度、注意力、空间编码） | `training_r0/model.py` |
| 超参与容量上限（`hidden`/`layers`/`heads`/`max_entities`/`max_candidates`/`max_edges`/`max_history` …） | `training_r0/config.py` |
| 训练目标、损失、TBPTT 语义 | `training_r0/il_learning.py`（权威实现，勿轻易偏离） |
| 输入特征与张量化 | `training_r0/observation.py`、`semantic_tensorizer.py`、`elite_tensorizer.py` |
| 数据集格式（npz 布局、编解码） | `training_r0/il_dataset.py` |
| 训练循环、批次组装、数据源 | `r0_train_batched.py` |
| CPU 数据后端（解码缓存、共享槽、亲和） | `r0_data_backend.py` |

### 两条必须遵守的约束

1. **`training_r0/data/*.json` 不要改。** 它们和 `training_r0/*.py` 一起被 `current_contract()`
   哈希，改了会让**所有既有数据集失效**。
2. **改了 `training_r0/*.py` 就会改变契约哈希**，于是新编译的数据与旧数据不能混训。
   要么重编数据，要么把改动限制在不参与哈希的文件里（`r0_train_batched.py`、
   `r0_data_backend.py`、`data_pipeline/r0_compile.py`、`r0_train.py` 都不在哈希内）。

`config.py` 里改 `max_*` 容量会同时影响显存峰值，改完先跑一次小 batch 确认没有 OOM。

---

## 2. 环境

```
Ubuntu 22.04 / glibc 2.35     Python 3.12
torch 2.8.0+cu128             CUDA 12.8 (build), driver >= 570
```

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**不要用 CPU-only 的 torch**：它能 import，但在第一个 kernel 上就会因为
`libcublasLt.so.* not found` 或架构不匹配而失败。

### 显存需求（实测，真实数据）

| Batch | 峰值显存 | 能跑在 |
|---:|---:|---|
| 256 | 12.5 GiB | 24 GB 卡 |
| 512 | 23.7 GiB | 24 GB 卡（紧） |
| 1024 | 46.3 GiB | 48 GB+ |
| 2048 | 91.3 GiB | 96 GB 卡 |

峰值由**激活**主导，不是输入：B=2048 时输入载荷只占 91.3 GiB 里的约 0.51 GiB。
参考卡实测 **5,317 决策/秒 @ B=2048**（CUDA Events，输入预置）。

---

## 3. 拿数据

```bash
export R0_XPAN_TOKEN='<token>'          # 或写 /root/.r0_xpan_token
python fetch_batch.py --list            # 网盘上有什么
python fetch_batch.py --probe --batch 0 # 只下 32 MiB，先测速
python fetch_batch.py --batch 0 --dest /data/r0data
```

解压（`zstd` 二进制在精简镜像上可能没有，用 Python 路径免依赖）：

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

**建构根列表**——必须是**每个** `out-N/compiled-prod-v17`，不是批次目录：

```bash
find /data/r0data/batch-0 -maxdepth 2 -type d -name 'compiled-prod-v17' | sort > roots.txt
find /data/r0data/batch-0 -name 'ledger-copy.jsonl' | sort > ledgers.txt
```

> 载入器用 `glob('*/manifest.json')` 索引一个 root，只下一层。传批次目录会得到
> `usable battles: 1975 / training lanes: 0`，看起来像数据不可用，其实是路径层级错了。

---

## 4. 一键训练

```bash
cd <repo root>        # 必须：契约路径相对于 CWD 解析

# inline（进程内加载）
bash tools/run_training.sh /data/r0data /data/run/my-run 1024 8

# backend（共享内存数据后端）
MODE=backend bash tools/run_training.sh /data/r0data /data/run/my-run-be 1024 8
```

或直接调用：

```bash
python -u r0_train_batched.py \
  --dataset $(cat roots.txt | tr '\n' ' ') \
  --ledger  $(cat ledgers.txt | tr '\n' ' ') \
  --output /data/run/my-run \
  --batch-size 1024 --steps 8 --tbptt-steps 4 --log-every 1
```

**`--dataset` 是"一个 flag 跟全部路径"。** 重复写 `--dataset` **不会累加**——argparse 只保留
最后一次出现。这会把 3609 条 lane 静默缩成单个 `out-N` 里那 40 条。

### 怎么看输出

`training_summary.json` 把时间拆成 `time_fuse_s` / `time_h2d_s` / `time_forward_s` /
`time_backward_opt_s` / `unaccounted_s`（取数与 lane 记账）。B=1024 实测占比：

| 分段 | 占比 |
|---|---:|
| 取数（在 `unaccounted` 里） | 55% |
| fuse / padding | 29% |
| backward + optimizer | 10% |
| forward | 5% |
| H2D | **0.5%** |

**瓶颈在 CPU 侧，不在 GPU。** 不要凭直觉以为 H2D 是问题——先测。

---

## 5. 信任一次跑之前，先跑验证

```bash
python verify_b1_equiv.py                          # B=1 必须复现 update_imitation_sequence
python verify_multilane.py --lanes 4 --tbptt 4     # 多 lane 等价、padding 透明、slot 隔离
python verify_reuse.py --lanes 4 --tbptt 4         # 一 shard 解码一次、四次消费、位级一致
```

预期：B=1 **loss 小数点后 8 位相同**、梯度相对误差 ~1e-7；多 lane fused vs solo logits ~2e-6。

这些测试不是形式——它们抓出过三个真实语义 bug：**每 tick detach**（把 TBPTT 压成 1 步）、
**loss 多除一个 `tbptt_steps`**、**backward 每 tick 一次而非每 chunk 一次**。

---

## 6. 采编链路（可选，只在你要造新数据时才用）

```bash
# 采集（需要游戏侧 capturer 与 replay 源）
python -m data_pipeline.r0_produce --selection <selection.json> --output <out> ...

# 编译 capture → compiled-prod-v17
python -m data_pipeline.r0_compile <capture-dir> --output <compiled-dir> --sequence-steps 16

# 严格认证（前向/反向可复现证明）
python -m data_pipeline.r0_certify <capture-dir> --dataset <compiled-dir> --device cuda
```

编译产出的 `manifest.json` 里带 `contract_sha256`，训练时会与本地算出的契约比对，不一致就跳过。

---

## 7. 已知陷阱

1. **必须传 `forced=`。** 这是模仿学习：`model(obs, state, forced=expert_actions)`。
   传 `forced=None` 会让模型**自己采样**，报出来的 loss 变成随机动作的 log-prob——
   **不报错，看起来还挺合理**，但训练的不是同一件事。
2. **`--dataset` 不累加**（见 §4）。
3. **数据根必须是 `out-N/compiled-prod-v17` 这一层**（见 §3）。
4. **换行符**：Windows 上生成的 shell 脚本上 Linux 前要 `sed -i 's/\r$//'`，
   否则 bash 报 `$'\r': command not found` 并**半途执行**。
5. **`np.frombuffer` 的 view 会钉住共享内存**：读共享槽拿到的是 view，持有 exported pointer，
   `shm.close()` 会抛 `BufferError`。先 `bytes(...)` 复制出来。
6. **多个 worker 写同一个槽不能共用游标**：各写互不相交区间，索引表由父进程在派发前**统一清零一次**。
   这两类 bug 造成过静默数据损坏（某条 lane 的特征差 5.6e-05）才被定位。
7. **进程池与 CUDA**：fork 出的子进程继承父进程的 CUDA context，首次使用即死，
   且**没有 Python traceback**（只报 `BrokenProcessPool`）。这类池用 `spawn`；
   本仓库的数据后端是纯 CPU 的，不碰 CUDA。

---

## 8. 仓库结构

```
training_r0/            模型、张量化、数据集格式、模仿更新
  model.py              网络结构
  config.py             超参与容量上限
  il_learning.py        权威训练更新（TBPTT / loss / backward 语义）
  il_dataset.py         npz 序列格式的编解码
  data/                 契约 JSON —— 不要改
data_pipeline/          全链路：采集 produce / 编译 compile / 认证 certify / 准入 admission
native_core/            运行期依赖（卡牌目录等）
expert_v1/              运行期依赖（Tick schema）
bindings/               运行期依赖
r0_train_batched.py     训练入口（一个循环，两种数据源）
r0_data_backend.py      CPU 数据后端（共享槽、shard 亲和、有界预取）
verify_*.py             正确性测试
fetch_batch.py          网盘取数
tools/                  取数 / 解压 / 建根列表 / 一键训练
```

## License

见 `LICENSE`。
