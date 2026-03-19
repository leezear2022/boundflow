# Phase 6H Artifact（AE/论文）README：Kick-the-tires + Claims 映射（≤30min）

> 目标：在一个“相对干净的环境”中，用一条命令复现 Phase 6H 的主结果工件（JSONL/CSV/summary/fig），并能审计 meta、处理不可比行（`comparable==0`）。

## 1. Getting Started（Kick-the-tires，≤30min）

### 1.1 环境

- 推荐 Conda 环境：`boundflow`
- Python：3.10+
- 必需依赖：PyTorch
- 可选依赖：`matplotlib`（用于出图；无则不会影响 JSONL/CSV/summary 的生成）
- 可选依赖：`tvm` / `onnx` / `auto_LiRPA`（用于更完整的历史 phase 测试与对齐验证；缺失时相关测试会 skip，不会在 collection 阶段崩）

### 1.2 一键运行（推荐）

```bash
# 默认输出到 artifact_out/phase6h_<timestamp>/
bash scripts/run_phase6h_artifact.sh

# 或指定输出目录（便于 AE 收集）
bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run
```

### 1.2.1 可选：扩展到非 toy workload（仍不改语义）

默认只跑 `1d_relu` 以保证 Kick-the-tires 足够快。若希望把主图扩展到小型非 toy MLP，可传第二个参数（或设置环境变量）：

```bash
# 方式 A：第二个参数（逗号分隔）
bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run "1d_relu,mlp2d_2x16"

# 方式 B：环境变量
PHASE6H_WORKLOADS="1d_relu,mlp2d_2x16" bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run
```

### 1.3 预期产物

在输出目录中应包含（文件名固定）：

- `phase6h_e2e.jsonl`：每行一个 `{meta, rows}`（JSON Lines）
  - `meta.schema_version` 当前为 `phase6h_e2e_v2`
- `phase6h_e2e.csv`：由 report 脚本展平后的表格（switch 组合 × {batch,serial}）
- `phase6h_e2e_summary.md`：自动生成摘要（主表只用 `comparable==1` 行）
- `figs/`：自动生成的 png 图（若无 `matplotlib` 可能为空）
- `env.txt` / `pip_freeze.txt` / `conda_list.txt`：环境审计信息（best-effort）

## 2. Claims 映射（Paper/Artifact Claims）

### C1：端到端 time-to-verify speedup

**Claim**：在相同配置与预算（`max_nodes`）下，启用 node-batch（K>1）以及 cache/branch-hint/prune 等系统优化后，端到端 `time_ms_p50`（以及 tail latency `time_ms_p90`）相比串行 baseline 显著下降（仅对 `comparable==1` 行作为主结论）。

**证据路径**：

- JSONL：`phase6h_e2e.jsonl`
  - `rows[*].batch_ms_p50 / rows[*].serial_ms_p50 / rows[*].speedup`（p50）
  - `rows[*].batch_ms_p90 / rows[*].serial_ms_p90 / rows[*].speedup_p90`（p90）
  - `rows[*].comparable`、`rows[*].note`（不可比显式标注）
- CSV：`phase6h_e2e.csv`
  - `path in {batch,serial}` + `time_ms_p50/time_ms_p90` + `speedup/speedup_p90` + `comparable`
- 图：`figs/*_speedup.png`、`figs/*_speedup_p90.png`（仅绘制 `comparable==1` 行）

### C2：收益归因（可审计 counters）

**Claim**：系统收益可归因到三类机制（开关矩阵）：

- `enable_node_eval_cache`：降低 `oracle_calls`、提高 `cache_hit_rate`
- `use_branch_hint`：降低 `forward_trace_calls`（避免分支阶段二次 forward）
- `enable_batch_infeasible_prune`：降低 `evaluated_nodes_count` / 提高剪枝

**证据路径**：

- JSONL：`phase6h_e2e.jsonl`
  - `rows[*].counts_batch.{oracle_calls,forward_trace_calls,cache_hit_rate,pruned_infeasible_count,evaluated_nodes_count}`
  - `rows[*].batch_stats.avg_batch_fill_rate`（解释吞吐→E2E 转化）
- CSV：`phase6h_e2e.csv`
  - `counts_*` 与 `stats_*` 列
- 图：`figs/*_counters.png`、`figs/*_fill_vs_speedup.png`

## 3. 可比性与限制（Non-claims / Limitations）

- 当前 runtime/solver 只覆盖 **链式 MLP（Linear+ReLU）子集**，不包含一般图结构（skip/branch）与 conv。
- `batch` 与 `serial` 的搜索顺序不同可能导致 `verdict` 不一致；此时 `rows[*].comparable==0`，不作为 speedup 主结论（`note` 会说明）。
- 本套基准主要面向 **系统/编译方向的可复现证据链**（schema/meta 固化 + 一键跑通 + 可归因），不是追求 SOTA verifier 的完整 workload 覆盖。

## 4. 手动工作流（可选）

```bash
# 1) sweep 追加 JSONL
python scripts/sweep_phase6h_e2e.py --out-jsonl /tmp/phase6h_e2e.jsonl

# 2) report 生成 CSV + summary.md
python scripts/report_phase6h_e2e.py \
  --in-jsonl /tmp/phase6h_e2e.jsonl \
  --out-csv /tmp/phase6h_e2e.csv \
  --out-summary-md /tmp/phase6h_e2e_summary.md

# 3) plot 出图（需要 matplotlib）
python scripts/plot_phase6h_e2e.py --in-jsonl /tmp/phase6h_e2e.jsonl --out-dir /tmp/phase6h_figs
```

## 5. 可选依赖与测试收集卫生（CI/AE 友好）

- 默认 `pytest -q tests` 不应因为缺少 `tvm/onnx` 在 **collection 阶段**报错：相关测试文件在缺依赖时会通过 `pytest.importorskip(...)` 机制被跳过。
- 若希望运行这些可选测试，需要自行安装对应依赖（并确保 `tvm.runtime.enabled("llvm")` 为 True）。
