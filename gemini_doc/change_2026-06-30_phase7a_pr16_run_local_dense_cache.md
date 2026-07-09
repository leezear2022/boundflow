# 2026-06-30：Phase 7A PR-16 run-local dense cache

## 摘要

本轮 PR-16 在 PR-15 operator attribution 基础上增加 run-local dense cache，用于复用同一次 shared CROWN backward / final concretization 内的 `LinearOperator.to_dense()` 结果。

这轮不改变 `split_pos_neg()` exact contract，不引入 over-approximation，不做 `SliceInput` fast path，也不把 CROWN/BaB 下沉到 TVM。

## 主要改动

### 1. Run-local dense cache

文件：`boundflow/runtime/linear_operator.py`

- 新增 `operator_dense_cache(enabled=True)` context manager。
- 使用 `contextvars.ContextVar` 承载 run-local cache。
- cache 生命周期仅覆盖当前 CROWN run；退出 context 后释放。
- 嵌套 context 会复用外层 cache；`enabled=False` 可显式禁用内部默认 cache。
- cache value 持有 `(operator, tensor)`，避免只用 `id(op)` 时出现 Python 对象 id 复用导致的语义污染。

已接入的 `to_dense()`：

- `RightMatmulLinearOperator.to_dense()`
- `AddLinearOperator.to_dense()`
- `SliceInputLinearOperator.to_dense()`
- `ScaledInputLinearOperator.to_dense()`

`DenseLinearOperator` / `RepeatedRowLinearOperator` 这类 view-like 路径不接入 cache。

### 2. CROWN backward 默认启用 cache

文件：`boundflow/runtime/crown_ibp.py`

- `run_crown_ibp_mlp()` 的 backward + final concretization 阶段默认进入 `operator_dense_cache(enabled=True)`。
- `run_crown_ibp_mlp_from_forward_trace()` 同样启用 cache，覆盖 alpha / beta / BaB 复用 forward trace 的路径。
- forward IBP trace 不放进 dense cache context，保持 cache 只服务 CROWN backward 相关物化。

### 3. Attribution 增加 cache stats

`operator_attribution` nested payload 新增：

- `cache.hits`
- `cache.misses`
- `cache.by_op`
- `cache.by_reason`

cache hit 不重复记录 materialization bytes；cache miss 才记录真实 materialization event。

## 观测结果

CPU smoke 全 workload 口径：

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7a_shared_crown_path_attribution.py --device cpu --profile smoke --workloads all --warmup 1 --iters 1
```

关键结论：

- `unknown_materialization` 继续为 0。
- `concat_relu_mlp` structured path 出现 `RightMatmulLinearOperator` cache hit。
- `concat_relu_mlp` 的 `right_matmul_exact_sign_split_required` materialization 从 disabled cache 的 `10 calls / 98304 bytes` 降到 enabled cache 的 `6 calls / 57344 bytes`。
- `relu_heavy_mlp` 仍主要是 cache misses，说明该 workload 中重复成本不是同一 operator identity 的重复 `to_dense()`。

## 测试

文件：`tests/test_phase7a_pr11_shared_crown_bench.py`

新增：

- cache enabled / disabled 的 structured bounds 必须 `allclose`。
- 直接对同一个 `RightMatmulLinearOperator.to_dense()` 连续调用时，cache stats 必须记录 `1 miss + 1 hit`，且 materialization 只记录一次。
- `concat_relu_mlp` structured path 下 cache enabled 的 `right_matmul_exact_sign_split_required` calls/bytes 不高于 cache disabled，并且 `RightMatmulLinearOperator` 有 hits/misses。
- benchmark nested schema 包含 `operator_attribution.cache`，且 `unknown_materialization` 仍为 0。

验证：

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7a_pr10_relu_barrier_structured.py tests/test_phase7a_pr9_dag_linear_operator.py tests/test_phase7a_pr11_shared_crown_bench.py
```

结果：`24 passed in 0.71s`

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7a_shared_crown_path_attribution.py --device cpu --profile smoke --workloads all --warmup 1 --iters 1
```

结果：通过；`concat_relu_mlp` structured path 的 `RightMatmulLinearOperator` cache stats 为 `2 hits / 8 misses`，`right_matmul_exact_sign_split_required` 为 `6 calls / 57344 bytes`。

## 后续判断

PR-16 证明 run-local identity cache 是语义安全的，也能在 concat 路径上消掉部分重复 `RightMatmul` 物化。但并不是所有 ReLU-heavy workload 都有同一 operator identity 的复用机会。

下一步若继续追性能，优先考虑：

- final concretization policy / hybrid planner；
- operator capability table；
- selective lowering；

而不是继续尝试伪 structured `RightMatmul.split_pos_neg()`。
