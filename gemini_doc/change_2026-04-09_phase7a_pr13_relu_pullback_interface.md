# 2026-04-09：Phase 7A PR-13 引入 ReLU 专用 pullback 接口

## 摘要

本轮 PR-13 是一次保守重构，不做新的 `RightMatmul` 数学优化，也不以性能提升为目标。

核心变化是把 ReLU backward 从：

- 直接依赖 `state.A_*.split_pos_neg()`

改成：

- 调用新的 `LinearOperator.relu_relax_pullback(...)` 内部接口

这样后续可以针对 `RightMatmulLinearOperator` 单独实现 sound 的 ReLU pullback，而不用再强迫它满足当前 exact `split_pos_neg()` contract。

## 代码改动

### 1. 新增 `LinearOperator.relu_relax_pullback(...)`

文件：`boundflow/runtime/linear_operator.py`

在 `LinearOperator` 协议中新增：

```python
def relu_relax_pullback(
    self,
    *,
    pos_slope,
    neg_slope,
    pos_bias,
    neg_bias,
) -> tuple[LinearOperator, torch.Tensor]
```

接口语义固定为：

- 输入按当前 `input_shape` 广播
- 输入要求非负
- 返回 `(A_out, delta_b)`，其中：
  - `A_out.shape == self.shape`
  - `delta_b.shape == [B, K]`

### 2. 新增共享默认实现 `_relu_relax_pullback_via_split(...)`

同文件新增 helper：

- 先调用 `op.split_pos_neg()`
- 再复用现有 split-based ReLU 公式构造：
  - `delta_b = A_pos.contract_input(pos_bias) + A_neg.contract_input(neg_bias)`
  - `A_out = ScaledInputLinearOperator(A_pos, pos_slope) + ScaledInputLinearOperator(A_neg, neg_slope)`

这保证 PR-13 的默认行为与 PR-12 完全一致。

### 3. 所有现有 operator 默认接入新接口

默认接入的 operator 包括：

- `DenseLinearOperator`
- `RightMatmulLinearOperator`
- `ReshapeInputLinearOperator`
- `ReindexInputLinearOperator`
- `Conv2dLinearOperator`
- `AddLinearOperator`
- `SliceInputLinearOperator`
- `ScaledInputLinearOperator`
- `RepeatedRowLinearOperator`

本轮里，`RightMatmulLinearOperator.relu_relax_pullback()` 明确仍使用默认 helper，也就是继续通过 `split_pos_neg()` 的 dense fallback 路径实现。

### 4. `_backprop_relu_step()` 改为只调用新接口

文件：`boundflow/runtime/crown_ibp.py`

改动前：

- `_backprop_relu_step()` 显式调用 `state.A_u.split_pos_neg()` / `state.A_l.split_pos_neg()`

改动后：

- 上界：
  - `A_u_op, delta_b_u = state.A_u.relu_relax_pullback(...)`
- 下界：
  - `A_l_op, delta_b_l = state.A_l.relu_relax_pullback(...)`

`relu_pre_add_coeff_u/l` 的附加项逻辑保持不变，仍然叠加到 pullback 后的 `A_u_op` / `A_l_op` 上。

### 5. 顺手修复 `Conv2dLinearOperator.split_pos_neg()`

同文件里把一个明显错误的错位实现修正为结构化传递：

- `base_pos, base_neg = base.split_pos_neg()`
- 分别返回两个 `Conv2dLinearOperator(...)`

这不是 PR-13 的主目标，但属于同一块 runtime 代码里的明显正确性问题。

## 测试

### 1. ReLU backward 行为回归

文件：`tests/test_phase7a_pr10_relu_barrier_structured.py`

新增/扩展：

- `relu_relax_pullback()` 与旧 split-based 参考公式的全量等价测试
- 覆盖 operator 形态：
  - `DenseLinearOperator`
  - `ReshapeInputLinearOperator`
  - `SliceInputLinearOperator`
  - `AddLinearOperator`
  - `RightMatmulLinearOperator`
- `_backprop_relu_step()` 使用 `relu_relax_pullback()` 接口的调用测试

### 2. 既有结构/bench 回归

继续验证：

- `tests/test_phase7a_pr9_dag_linear_operator.py`
- `tests/test_phase7a_pr11_shared_crown_bench.py`

其中 bench 断言继续锁定：

- `SliceInputLinearOperator` 不重新出现在 dense hotspot 中
- `RightMatmulLinearOperator` 仍然是唯一剩余热点

## 结果

### 单测

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr10_relu_barrier_structured.py \
  tests/test_phase7a_pr9_dag_linear_operator.py \
  tests/test_phase7a_pr11_shared_crown_bench.py
```

结果：`18 passed in 0.88s`

### CPU smoke

```bash
conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cpu --profile smoke --workloads all --warmup 1 --iters 1
```

结果摘要：

- `relu_heavy_mlp`: `0.58x`
- `residual_relu_mlp`: `0.49x`
- `concat_relu_mlp`: `0.44x`
- `permute_reshape_linear`: `0.74x`

### CUDA 主口径

```bash
conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cuda --profile bench --workloads all --warmup 5 --iters 20
```

结果：

| workload | structured ms p50 | baseline ms p50 | speedup |
|---|---:|---:|---:|
| `relu_heavy_mlp` | `4.864` | `2.628` | `0.54x` |
| `residual_relu_mlp` | `2.802` | `1.589` | `0.57x` |
| `concat_relu_mlp` | `4.915` | `2.259` | `0.46x` |
| `permute_reshape_linear` | `1.335` | `1.492` | `1.12x` |

计数侧保持：

- `relu_heavy_mlp`: `{"RightMatmulLinearOperator": 8}`
- `residual_relu_mlp`: `{"RightMatmulLinearOperator": 4}`
- `concat_relu_mlp`: `{"RightMatmulLinearOperator": 6}`

## 结论

- PR-13 已经把 ReLU caller 与 exact `split_pos_neg()` contract 解耦。
- 当前默认 `relu_relax_pullback()` 仍复用旧 split-based 行为，因此数值行为保持不变。
- `RightMatmulLinearOperator` 仍然是唯一剩余的 dense hotspot；这轮没有也不应该假装解决它。
- 下一步应在不改 `split_pos_neg()` 语义的前提下，为 `RightMatmulLinearOperator.relu_relax_pullback()` 单独设计 sound 实现。
