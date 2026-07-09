# Phase 7B PR-19：Benchmark Matrix / Crossover Study

**日期**: 2026-07-01

## 背景

Phase 7A 已完成：

```text
PR-15 attribution
PR-16 run-local dense cache
PR-17 final concretization policy
PR-18 hybrid planner / capability table
```

PR-19 进入 Phase 7B，目标不是继续改 bound 数学，而是建立一套可复跑的 benchmark matrix，用来回答：

```text
structured / dense_barrier / auto 在不同 workload 与 scale 下什么时候出现 crossover？
```

这些数据会成为后续 cost model 的输入。

## 主要改动

### 1. 新增 `small` profile

更新 `scripts/bench_phase7a_shared_crown_path_attribution.py`：

- `--profile` 新增 `small`。
- 四个 workload 都增加 `small` 尺寸：
  - `relu_heavy_mlp`
  - `residual_relu_mlp`
  - `concat_relu_mlp`
  - `permute_reshape_linear`

`smoke` 保持轻量测试口径，`bench` 保持较大规模，`small` 用于 Phase 7B scale sweep 的中间点。

### 2. 新增 crossover matrix 脚本

新增 `scripts/bench_phase7b_crossover_matrix.py`。

输出 schema：

```text
phase7b_crossover_matrix.v1
```

支持参数：

```bash
--workloads relu_heavy_mlp,permute_reshape_linear
--scales smoke,small
--policies structured,dense_barrier,auto
--warmup 3
--iters 5
```

脚本复用 PR-18 的 shared benchmark `_collect_row()`，避免复制计时、baseline 与 attribution 逻辑。每个 matrix row 包含：

- `workload`
- `scale_id`
- `policy_request`
- `planner_decision`
- `metrics`
- `raw_row`

其中 `metrics` 抽取：

- `structured_ms_p50`
- `baseline_ms_p50`
- `speedup`
- `materialized_bytes`
- `right_matmul_exact_bytes`
- `final_bound_concretization_bytes`
- `final_bound_dense_barrier_bytes`
- `cache_hits`
- `cache_misses`
- `unknown_materialization_calls`
- `split_pos_neg_dense_total`
- `planner_final_concretization_policy`

同一个 `(workload, scale)` 下的不同 policy 共用同一个 seed，避免 policy 对比被随机输入/权重污染。

### 3. summary

matrix payload 额外输出 `summary`，按 `(workload, scale_id)` 聚合：

- `best_policy_by_structured_ms`
- `best_policy_by_speedup`
- `auto_final_concretization_policy`
- `dense_barrier_vs_structured_ms_ratio`

注意：`warmup=1,iters=1` 的 smoke 只用于验证管线与 schema，不用于性能结论。

### 4. 测试

新增 `tests/test_phase7b_crossover_matrix.py`：

- 验证 `phase7b_crossover_matrix.v1` schema。
- 验证 row / summary 结构。
- 验证 `auto` policy 对 layout-only workload 选择 `dense_barrier`。
- 验证 `unknown_materialization_calls == 0` 与 `split_pos_neg_dense_total == 0`。

## 验证

新增测试：

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7b_crossover_matrix.py
```

结果：

```text
1 passed
```

小矩阵 smoke：

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7b_crossover_matrix.py \
  --device cpu \
  --workloads relu_heavy_mlp,permute_reshape_linear \
  --scales smoke \
  --policies structured,dense_barrier,auto \
  --warmup 1 \
  --iters 1
```

结果：通过，输出 `phase7b_crossover_matrix.v1`，所有 structured rows 的 `unknown_materialization_calls == 0`，且 ReLU / layout workload 的 planner decision 正常写入。

## 下一步

PR-20 应基于 PR-19 的 matrix 数据做 cost model v1：

1. 跑 CPU/CUDA 的 `smoke,small,bench` matrix。
2. 用多 iter / torch benchmark 减少噪声。
3. 以 materialized bytes、cache hit/miss、wrapper depth、policy latency 为输入，形成第一版自动选择规则。
4. 若 layout-only 的 dense barrier crossover 稳定，再把它列为 selective lowering 候选。
