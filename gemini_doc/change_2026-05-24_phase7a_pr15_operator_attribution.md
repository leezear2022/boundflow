# 2026-05-24：Phase 7A PR-15 opt-in operator attribution

## 摘要

本轮 PR-15 是纯 instrumentation 变更：在不改变 `LinearOperator` 语义和 benchmark timing 口径的前提下，为 shared CROWN ReLU pullback 增加 opt-in operator attribution。

目标从“structured ReLU path 必须立即快过 dense barrier”收缩为“解释 PR-14 后 ReLU path 的剩余 dense / materialization / wrapper 成本，并为后续 hybrid planner 或 selective lowering 提供证据”。

## 主要改动

### 1. Attribution context

文件：`boundflow/runtime/linear_operator.py`

- 新增 `collect_operator_attribution(path_kind, phase)` context manager。
- 使用 `contextvars.ContextVar` 承载 trace、phase 和 materialization reason，避免全局状态污染嵌套调用或并行测试。
- attribution 关闭时不改变执行路径；记录逻辑只在真实 pullback / fallback / materialization 发生时旁路记录，不为统计额外触发 `to_dense()`。

记录内容包括：

- ReLU pullback 调用次数、operator 类型、operator depth、wrapper 创建计数。
- materialization 的 op、shape、numel、bytes、phase、reason。
- fallback 的 reason 与 operator 类型。

固定 reason taxonomy：

- `explicit_split_pos_neg_dense`
- `right_matmul_exact_sign_split_required`
- `slice_pullback_materialize`
- `unsupported_structured_relu_pullback`
- `broadcast_materialization`
- `dense_reference_check`
- `dense_baseline_materialization`
- `final_bound_concretization`
- `unknown_materialization`

### 2. Benchmark nested attribution

文件：`scripts/bench_phase7a_shared_crown_path_attribution.py`

- `_collect_counts()` 阶段启用 attribution。
- timing 阶段不启用 attribution，避免 instrumentation 污染 latency。
- 顶层 schema 保持 `phase7a_shared_crown_path_attribution.v1`。
- 在 `counts_structured` / `counts_baseline` 下新增 optional nested payload：
  - `operator_attribution.schema_version`
  - `operator_attribution.path_kind`
  - `operator_attribution.relu_pullback.by_op`
  - `operator_attribution.materialization.by_phase`
  - `operator_attribution.materialization.by_op`
  - `operator_attribution.materialization.by_reason`
  - `operator_attribution.materialization.events`
  - `operator_attribution.fallback.by_reason`

Phase 语义：

- `structured_execution`：structured path 内部真实执行。
- `benchmark_baseline`：dense barrier baseline。
- `dense_reference_check`：预留给 correctness check，不混入 structured path 成本解释。

### 3. 回归测试

文件：`tests/test_phase7a_pr11_shared_crown_bench.py`

新增或扩展：

- attribution enabled / disabled 的 structured bounds 必须 `allclose`。
- 旧 benchmark 字段保持存在，顶层 schema version 不变。
- `operator_attribution.path_kind` 区分 `structured` 与 `dense_baseline`。
- 每个 materialization event 都必须带 `op / shape / numel / bytes / phase / reason`。
- 若 `RightMatmulLinearOperator` ReLU pullback 发生 dense materialization，必须记录为 `right_matmul_exact_sign_split_required` 并带 shape/bytes。
- 全 workload CPU smoke 中 `unknown_materialization` 调用数为 0；final concretization 与 dense baseline materialization 分别归入独立 reason。
- ReLU workloads 仍锁定 `split_pos_neg_dense_total == 0`。

## 结论

PR-15 不追求 speedup，而是把 Phase 7A 从经验性 operator 优化推进到可解释优化闭环。后续 PR-16 根据 attribution 数据分叉：

- repeated materialization / broadcast 主导：做 pullback-local cache 或 broadcast folding。
- `SliceInput` 主导：做 exact structured fast path。
- `RightMatmul` exact sign split 主导：做 cached dense + hybrid planner policy。
- wrapper / Python dispatch 主导：转向 planner-level fusion 或 selective TVM lowering。

## 验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7a_pr11_shared_crown_bench.py
```

结果：`5 passed in 0.84s`

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7a_pr10_relu_barrier_structured.py tests/test_phase7a_pr9_dag_linear_operator.py
```

结果：`16 passed in 0.83s`

完整 PR-15 回归：

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7a_pr10_relu_barrier_structured.py tests/test_phase7a_pr9_dag_linear_operator.py tests/test_phase7a_pr11_shared_crown_bench.py
```

结果：`21 passed in 1.13s`

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7a_shared_crown_path_attribution.py --device cpu --profile smoke --workloads all --warmup 1 --iters 1
```

结果：4 个 workload 的 nested attribution 中 `unknown_materialization` 总调用数为 0。
