# 变更记录：Phase 6H PR-1（BaB/αβ）——端到端 time-to-verify 基准（开关矩阵 + JSONL 工件）

## 动机

Phase 6G PR-4 已把 *node-eval throughput* 的收益归因口径钉死，但 reviewer 通常会继续追问：

- “整体验证时间（time-to-verify）是否下降？下降多少？”
- “吞吐提升是否能转化为端到端闭环收益？”

因此本 PR 在 **不改求界语义** 的前提下，新增一个端到端 BaB 基准脚本，用与 PR-4 一致的开关矩阵与计数器口径，把收益从 *node-eval 局部吞吐* 升级为 *E2E time-to-verify*。

## 本次改动

### 1) BaB 统计字段补齐（不动语义，只增强可观测性）

- 更新：`boundflow/runtime/bab.py`
  - `BabConfig` 新增 `use_branch_hint: bool=True`（默认保持现状；用于 E2E ablation）。
  - `BabResult` 新增端到端可观测统计（用于 bench 输出）：
    - `nodes_evaluated`：成功完成一次求界并进入“处理该节点结果”的节点数
    - `nodes_expanded`：实际发生 split（push 两个子节点）的节点数
    - `batch_rounds`：node-batch 循环轮数
    - `avg_batch_fill_rate`：平均 batch 填充率（`sum(pop)/ (rounds*K)`）
  - `solve_bab_mlp` 逻辑保持不变，仅在关键路径上累加上述计数器，并在返回时携带。

### 2) 端到端 BaB time-to-verify 基准（开关矩阵 + 固化计数器）

- 新增：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - 与 PR-4 兼容的输出结构：stdout JSON = `{meta, rows}`，每个 `row` 对应一个开关组合。
  - 开关矩阵（2×2×2）：
    - `enable_node_eval_cache`
    - `use_branch_hint`
    - `enable_batch_infeasible_prune`
  - 每个 row 同时输出：
    - `batch_ms_p50 / serial_ms_p50 / speedup`
    - `batch_verdict / serial_verdict`
    - `batch_stats / serial_stats`：包含 `popped_nodes/evaluated_nodes/expanded_nodes/max_queue_size/avg_batch_fill_rate`
    - `counts_batch / counts_serial`：复用 PR-4 的固定计数器字段
  - 额外工件能力：
    - `--jsonl-out <path>`：把整份 `{meta,rows}` 作为一行 JSONL 追加写入（便于 sweep 汇总与出图）。
  - 口径补强（DoD）：
    - `meta` 固化：`git_sha/device_name/torch_version/timer/...`（便于复现实验与出图脚本无歧义解析）
    - `rows` 增加 `comparable/note`：当 `batch_verdict!=serial_verdict` 时显式标注（speedup 仅参考）
    - `batch_stats/serial_stats` 增加审计别名：`popped_nodes_total`、`queue_peak`
    - 支持 `--timer {perf_counter,torch_benchmark}`（与 Phase 6C/6G 一致）

## 如何验证

```bash
# 1D workload：验证脚本能输出 8 行开关组合，并包含固定计数器字段
python scripts/bench_phase6h_bab_e2e_time_to_verify.py \
  --device cpu --dtype float32 --workload 1d_relu \
  --oracle alpha_beta --steps 0 --max-nodes 256 --node-batch-size 32 \
  --warmup 1 --iters 3

# 可选：输出 JSONL 工件（每次运行追加一行 meta+rows）
python scripts/bench_phase6h_bab_e2e_time_to_verify.py \
  --device cpu --dtype float32 --workload 1d_relu \
  --oracle alpha_beta --steps 0 --max-nodes 256 --node-batch-size 32 \
  --warmup 1 --iters 3 \
  --jsonl-out /tmp/phase6h_e2e.jsonl
```

## 最小 schema 钉子（推荐加入回归）

```bash
python -m pytest -q tests/test_phase6h_bench_e2e_schema.py
```

## 备注

- 本 PR 目标是把 “吞吐归因” 升级为 “端到端 time-to-verify 归因”；出图/出表脚本（sweep/plot）建议作为后续 PR（保持本 PR 语义不动、改动最小）。
