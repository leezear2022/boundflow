# 2026-04-10：Phase 7A PR-14 为 `RightMatmul` 实现 ReLU 专用 pullback

## 摘要

本轮 PR-14 没有去修改 `RightMatmulLinearOperator.split_pos_neg()` 的 exact contract，而是只优化了 ReLU 专用路径：

- 为 `RightMatmulLinearOperator.relu_relax_pullback()` 提供专用实现
- 为 `SliceInputLinearOperator.relu_relax_pullback()` 提供对 base 的委托实现

目标是把 shared CROWN ReLU backward 从 `_split_pos_neg_dense(...)` 热路径上挪开，同时保持和现有 dense reference 的数值对齐。

## 代码改动

### 1. `RightMatmulLinearOperator.relu_relax_pullback()`

文件：`boundflow/runtime/linear_operator.py`

新实现不再调用：

- `self.split_pos_neg()`
- `_split_pos_neg_dense(self)`

而是直接：

- 广播并校验 `pos_slope / neg_slope / pos_bias / neg_bias`
- `dense = self.to_dense()`
- 用 `torch.where(dense >= 0, ..., ...)` 精确构造 pullback 后的系数矩阵
- 计算 `delta_b`
- 返回一个由正负 dense 部分组成的 `AddLinearOperator`

这条路径保持了与旧 split-based dense reference 的 exact 等价，但避免了旧的 `split_pos_neg_dense` 观测热点。

### 2. `SliceInputLinearOperator.relu_relax_pullback()`

`concat` 路径上还有一层 slice 包装，因此只改 `RightMatmul` 不够。

本轮增加：

- 把 `pos_slope / neg_slope / pos_bias / neg_bias` 先嵌回 `base.input_shape`
- 调用 `base.relu_relax_pullback(...)`
- 再把返回的 `A_out` slice 回当前输入轴

这样 `concat_relu_mlp` 上也能避开默认 split helper 重新递归进 `RightMatmul.split_pos_neg()`。

### 3. 默认绑定修正

PR-13 在文件底部把若干 operator 统一绑定到了 `_default_relu_relax_pullback`。

本轮把以下类从默认绑定名单中移除，保留它们自己的专用实现：

- `RightMatmulLinearOperator`
- `SliceInputLinearOperator`

## 测试

### 单测更新

文件：`tests/test_phase7a_pr10_relu_barrier_structured.py`

新增：

- `test_right_matmul_relu_relax_pullback_does_not_use_split_pos_neg_dense`
  - 通过 monkeypatch `_split_pos_neg_dense` 为 fail hook
  - 验证 `RightMatmul.relu_relax_pullback()` 不再依赖旧 split 路径
  - 同时保持和旧 dense reference 的 exact 一致

文件：`tests/test_phase7a_pr11_shared_crown_bench.py`

更新：

- `relu_heavy_mlp`、`residual_relu_mlp`、`concat_relu_mlp` 的 `split_pos_neg_dense_total == 0`
- `split_pos_neg_dense_by_op` 中不再出现 `RightMatmulLinearOperator`

## 验证

### Pytest

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr10_relu_barrier_structured.py \
  tests/test_phase7a_pr9_dag_linear_operator.py \
  tests/test_phase7a_pr11_shared_crown_bench.py
```

结果：`19 passed in 0.86s`

### CPU smoke

```bash
conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cpu --profile smoke --workloads all --warmup 1 --iters 1
```

结果：

| workload | speedup |
|---|---:|
| `relu_heavy_mlp` | `0.67x` |
| `residual_relu_mlp` | `0.71x` |
| `concat_relu_mlp` | `0.49x` |
| `permute_reshape_linear` | `0.73x` |

计数侧：4 个 workload 的 `split_pos_neg_dense_total` 都是 `0`。

### CUDA 主口径

```bash
conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cuda --profile bench --workloads all --warmup 5 --iters 20
```

结果：

| workload | structured ms p50 | baseline ms p50 | speedup |
|---|---:|---:|---:|
| `relu_heavy_mlp` | `4.037` | `2.607` | `0.65x` |
| `residual_relu_mlp` | `2.342` | `1.592` | `0.68x` |
| `concat_relu_mlp` | `4.256` | `2.255` | `0.53x` |
| `permute_reshape_linear` | `1.315` | `1.463` | `1.11x` |

计数侧：

- `relu_heavy_mlp`: `{}`
- `residual_relu_mlp`: `{}`
- `concat_relu_mlp`: `{}`

## 结论

- PR-14 已经把 ReLU 路径上的 `split_pos_neg_dense` 热点清零。
- 改善来自 `relu_relax_pullback()` 专用实现，不是 `RightMatmul` 获得了通用 structured sign split。
- 三个 ReLU workload 相比 PR-13 均有回升：
  - `0.54x -> 0.65x`
  - `0.57x -> 0.68x`
  - `0.46x -> 0.53x`
- 仍未超过 dense ReLU barrier，因此后续若继续优化，应优先减少 `relu_relax_pullback()` 内部 dense materialization 的成本，而不是试图重写 `split_pos_neg()` contract。
