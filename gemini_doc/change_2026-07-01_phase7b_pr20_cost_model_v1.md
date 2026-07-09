# Phase 7B PR-20：Cost Model v1 Evidence Postprocess

**日期**: 2026-07-01

## 背景

PR-19 已经新增 `phase7b_crossover_matrix.v1`，可以按：

```text
workload × scale × policy
```

复跑 shared CROWN 的 structured / dense_barrier / auto 策略。PR-20 的目标是把这些 matrix 数据转成 cost model v1 的证据，而不是直接把 noisy smoke 结果写进 runtime 默认 planner。

## 主要改动

### 1. 新增 cost model 后处理脚本

新增 `scripts/postprocess_phase7b_cost_model.py`。

输入：

```text
phase7b_crossover_matrix.v1
```

输出：

```text
phase7b_cost_model_v1
```

每条 rule 按 `(workload, scale_id)` 聚合，输出：

- `recommended_policy_request`
- `recommended_final_concretization_policy`
- `confidence`
- `relative_gap_to_second_best`
- `dense_barrier_vs_structured_ms_ratio`
- `evidence.policy_ms_p50`
- `evidence.materialized_bytes`
- `evidence.right_matmul_exact_bytes`
- `evidence.cache_hits`
- `evidence.cache_misses`
- `guardrails.unknown_materialization_calls`
- `guardrails.split_pos_neg_dense_total`

### 2. 按 final policy 计算置信度

`structured`、`dense_barrier`、`auto` 是 policy request，但 `auto` 可能解析到 `structured` 或 `dense_barrier`。

因此 PR-20 的 confidence 不按 request 互相竞争，而是按最终执行策略比较：

```text
final_concretization_policy = structured | dense_barrier
```

这避免了 `auto` 和 `dense_barrier` 同时指向 dense barrier 时互相稀释证据。

### 3. 低迭代 smoke 降级

PR-19 smoke 常用：

```text
warmup=1
iters=1
```

这只能验证 schema 和管线，不足以形成性能结论。PR-20 新增：

```bash
--min-iters-for-confidence 3
```

默认 `iters < 3` 时，规则的最高 confidence 会被 cap 到 `low`。

### 4. 测试

新增 `tests/test_phase7b_cost_model.py`：

- 用合成 `phase7b_crossover_matrix.v1` payload 验证 rule 输出。
- 验证 dense barrier 与 structured 的推荐 policy。
- 验证 CLI schema smoke。
- 验证 guardrails 保留 `unknown_materialization_calls` 与 `split_pos_neg_dense_total`。

## 验证

新增测试：

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7b_cost_model.py
```

结果：

```text
2 passed
```

真实 PR-19 smoke 后处理：

```bash
conda run --no-capture-output -n boundflow python scripts/postprocess_phase7b_cost_model.py \
  /tmp/phase7b_pr19_matrix_smoke.json \
  --min-relative-margin 0.05
```

结果：通过，输出 `phase7b_cost_model_v1`。由于来源 matrix 是 `iters=1`，`max_confidence_from_measurement_reliability` 为 `low`。

## 下一步

PR-21 应跑正式矩阵：

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7b_crossover_matrix.py \
  --device cpu \
  --workloads all \
  --scales smoke,small,bench \
  --policies structured,dense_barrier,auto \
  --warmup 5 \
  --iters 20
```

如果有 CUDA：

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7b_crossover_matrix.py \
  --device cuda \
  --workloads all \
  --scales smoke,small,bench \
  --policies structured,dense_barrier,auto \
  --warmup 5 \
  --iters 20
```

然后用 PR-20 后处理脚本生成 cost model evidence。若某些 workload/scale 出现稳定 high-confidence 规则，再把它们推进到 runtime planner v2。
