# 2026-04-09：Phase 7A PR-12 缩小版

## 摘要

本轮没有按最初设想去实现 `RightMatmulLinearOperator.split_pos_neg()` 的结构化分解，而是先完成了一个正确性安全的子集：

- `SliceInputLinearOperator.split_pos_neg()` 改为结构化传递，不再直接走 `_split_pos_neg_dense(self)`

同时，把 `RightMatmulLinearOperator` 明确记录为当前 shared CROWN ReLU 路径上的 contract blocker。

## 代码变更

### 1. `SliceInputLinearOperator.split_pos_neg()` 改成结构化传递

文件：`boundflow/runtime/linear_operator.py`

改动前：

- `SliceInputLinearOperator.split_pos_neg()` 直接调用 `_split_pos_neg_dense(self)`

改动后：

- 先对 `base` 做 `base.split_pos_neg()`
- 再分别包装成同参数的 `SliceInputLinearOperator(base=base_pos, ...)`
- 保持 `input_shape`、`start`、`stop` 不变

这满足当前 `split_pos_neg()` 的 exact contract，因为对输入轴做 slice 后的逐元素正负部，等价于先对底层 operator 做逐元素正负部分解，再取同样的 slice。

### 2. 新增 `SliceInputLinearOperator.split_pos_neg()` 精确性测试

文件：`tests/test_phase7a_pr9_dag_linear_operator.py`

新增测试覆盖：

- `pos.to_dense() + neg.to_dense() == op.to_dense()`
- `pos.to_dense() >= 0`
- `neg.to_dense() <= 0`
- `pos/neg` 的 `input_shape` 与 slice 后逻辑形状一致

### 3. 更新 PR-11 bench 热点断言

文件：`tests/test_phase7a_pr11_shared_crown_bench.py`

新的约束是：

- `relu_heavy_mlp` 上不应再出现 `SliceInputLinearOperator`
- `concat_relu_mlp` 上 `SliceInputLinearOperator` 必须归零
- `RightMatmulLinearOperator` 仍然保留为热点

## `RightMatmulLinearOperator` blocker 说明

本轮没有提交 `RightMatmulLinearOperator.split_pos_neg()` 的四项拆分实现，原因是它不满足当前 exact contract。

当前 `split_pos_neg()` 的语义是：

- `pos == to_dense().clamp_min(0)`
- `neg == to_dense().clamp_max(0)`

而对 `A @ rhs` 使用：

- `rhs_pos = rhs.clamp_min(0)`
- `rhs_neg = rhs.clamp_max(0)`
- `pos = A_pos @ rhs_pos + A_neg @ rhs_neg`
- `neg = A_pos @ rhs_neg + A_neg @ rhs_pos`

虽然能保证 `pos + neg == A @ rhs`，但无法保证逐元素正负部精确一致，因为矩阵乘法会在求和维度发生符号抵消。这个问题已经会破坏 PR-10 的 ReLU structured 对齐测试，因此本轮明确不引入这类错误优化。

## bench 复跑结果

### CPU smoke

命令：

```bash
conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cpu \
  --profile smoke \
  --workloads all \
  --warmup 1 \
  --iters 1
```

结果摘要：

- `relu_heavy_mlp`: `0.62x`
- `residual_relu_mlp`: `0.57x`
- `concat_relu_mlp`: `0.20x`
- `permute_reshape_linear`: `0.74x`

CPU 仍只作为 smoke，可运行即可，不做主结论。

### CUDA 主口径

命令：

```bash
conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cuda \
  --profile bench \
  --workloads all \
  --warmup 5 \
  --iters 20
```

结果：

| workload | structured ms p50 | baseline ms p50 | speedup |
|---|---:|---:|---:|
| `relu_heavy_mlp` | `3.956` | `2.612` | `0.66x` |
| `residual_relu_mlp` | `2.423` | `1.596` | `0.66x` |
| `concat_relu_mlp` | `4.406` | `2.269` | `0.51x` |
| `permute_reshape_linear` | `1.335` | `1.478` | `1.11x` |

计数侧：

- `relu_heavy_mlp`: `{"RightMatmulLinearOperator": 8}`
- `residual_relu_mlp`: `{"RightMatmulLinearOperator": 4}`
- `concat_relu_mlp`: `{"RightMatmulLinearOperator": 6}`

结论：

- `SliceInputLinearOperator` dense fallback 已经归零
- `RightMatmulLinearOperator` 成为唯一剩余热点
- 这轮优化没有带来整体 ReLU latency 提升，`concat_relu_mlp` 甚至更慢

## 影响面

- 不改 solver public API
- 不改 benchmark schema
- 不改 `RightMatmulLinearOperator` 语义
- 把后续优化目标进一步收敛为“为 `RightMatmul` 设计满足 exact contract 的表示或证明”

## 验证

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr9_dag_linear_operator.py \
  tests/test_phase7a_pr11_shared_crown_bench.py \
  tests/test_phase7a_pr10_relu_barrier_structured.py::test_backprop_relu_step_preserves_operator_form_and_matches_dense_reference
```

结果：`11 passed in 0.80s`

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7a_shared_crown_path_attribution.py --device cpu --profile smoke --workloads all --warmup 1 --iters 1
```

结果：脚本正常输出 JSON，4 个 workload 均可运行。

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7a_shared_crown_path_attribution.py --device cuda --profile bench --workloads all --warmup 5 --iters 20
```

结果：见上方 CUDA 主口径表。
