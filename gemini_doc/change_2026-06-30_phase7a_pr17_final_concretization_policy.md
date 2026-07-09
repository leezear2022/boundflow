# 2026-06-30：Phase 7A PR-17 final concretization policy

## 摘要

本轮 PR-17 在 PR-16 run-local dense cache 之后，增加 final concretization policy 机制，让 shared CROWN benchmark 可以显式比较 structured final concretization 与 dense barrier final concretization。

这轮仍不改变默认路径：默认 policy 保持 `structured`。新增的 `dense_barrier` 只在显式 context 或 benchmark CLI 参数下启用。

## 主要改动

### 1. Final concretization policy context

文件：`boundflow/runtime/perturbation.py`

- 新增 `final_concretization_policy(policy)` context manager。
- 支持：
  - `structured`：默认，沿用 `LinearOperator.center_term()` + `row_abs_sum/l2/max()`。
  - `dense_barrier`：在 final concretization 起点把 `LinearOperator` materialize 成 `DenseLinearOperator`，再走 dense exact concretization。
- `dense_barrier` 保持 exact 语义，不改变 bound tightness 或 verification decision。

### 2. Attribution reason 扩展

文件：`boundflow/runtime/linear_operator.py`

- 新增 reason：`final_bound_dense_barrier`。
- `dense_barrier` policy 下 final concretization 的物化会归因到该 reason。
- `unknown_materialization` 继续保持为回归约束。

### 3. Benchmark CLI 支持

文件：`scripts/bench_phase7a_shared_crown_path_attribution.py`

- 新增参数：

```bash
--final-concretization-policy {structured,dense_barrier}
```

- 默认值为 `structured`。
- timing 与 counts 都使用同一个 policy。
- 顶层 schema version 保持 `phase7a_shared_crown_path_attribution.v1`，meta 中新增 `final_concretization_policy` 字段。

## 测试

文件：`tests/test_phase7a_linear_operator_concretize.py`

- 新增 dense barrier policy 与 structured policy 的 bounds 等价测试。
- 验证 `final_bound_dense_barrier` attribution reason 出现，且无 `unknown_materialization`。

文件：`tests/test_phase7a_pr11_shared_crown_bench.py`

- benchmark schema smoke 验证 meta 中的 `final_concretization_policy` 默认是 `structured`。

## 验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7a_linear_operator_concretize.py tests/test_phase7a_pr11_shared_crown_bench.py
```

结果：`18 passed, 1 skipped in 0.69s`

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7a_shared_crown_path_attribution.py --device cpu --profile smoke --workloads relu_heavy_mlp --warmup 1 --iters 1 --final-concretization-policy dense_barrier
```

结果：通过，meta 中 `final_concretization_policy` 为 `dense_barrier`，structured counts 中出现 `final_bound_dense_barrier` reason。

## 后续判断

PR-17 先提供 policy 与观测口径，不把 dense barrier 设成默认。下一步应基于 CPU/CUDA bench 比较：

- `structured`
- `dense_barrier`

若某类 workload 在 dense barrier final concretization 下更稳，则再考虑 PR-18 做 capability table / auto-selection。
