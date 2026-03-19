# 2026-03-18：Phase 7A PR-4——Conv2dLinearOperator 的 exact lazy row-norm 归约

## 背景与动机

Phase 7A PR-3 已经把 `Conv2dLinearOperator` 接到高维 `NCHW` contract 和 plain CROWN-IBP 的链式 CNN 子集上，但它的三个 row norm 入口：

- `row_abs_sum()`
- `row_l2_norm()`
- `row_abs_max()`

仍然是直接 `self.to_dense()` 后再对 `[B,K,I]` 扁平 dense matrix 做归约。

这条路径在语义上是正确的，但把 Conv operator 的结构信息全都提前摊平成大矩阵，和 PR-3 想保留的 operator 边界相冲突。因此 PR-4 的目标很收敛：

- 不改任何 public API。
- 不扩 solver 语义。
- 不改变 bound 数值。
- 只把 `Conv2dLinearOperator` 的 row norm 归约改成 exact lazy 路径。

这里的 “lazy” 是指：
- 不再经由 `Conv2dLinearOperator.to_dense()` 先构 `[B,K,I]`。
- 改为沿 operator 结构递归 materialize `[B*K,C,H,W]` feature-map rows，再直接在 NCHW 上归约。

它不是零 materialization，也不承诺改善最坏时间/空间复杂度。

## 主要改动

### 1. 在 `linear_operator.py` 增加结构化 rows helper

更新：`boundflow/runtime/linear_operator.py`

新增两个私有 helper：

- `_materialize_feature_map_rows(op, expected_input_shape)`
  - 返回 `[B*K,C,H,W]`
  - 只服务 rank-3 输入形状 `(C,H,W)`
  - 对不同 operator 分支做结构化处理
- `_reduce_feature_map_rows(rows, batch, spec_dim, reduce)`
  - 将 `[B*K,C,H,W]` 直接归约成 `[B,K]`
  - 支持 `l1` / `l2` / `linf`

这两个 helper 没有进入 `LinearOperator` protocol，保持为 runtime 内部实现细节。

### 2. `DenseLinearOperator` / `ReshapeInputLinearOperator` / `Conv2dLinearOperator` 的 rows materialization 切分

更新：`boundflow/runtime/linear_operator.py`

`_materialize_feature_map_rows(...)` 的规则是：

- `DenseLinearOperator`
  - 直接把现有系数 reshape 成 `[B*K,C,H,W]`
  - 不引入额外 flatten/expand
- `ReshapeInputLinearOperator`
  - 先尽量递归 materialize `base`
  - 再根据 `input_shape` 做 view/reshape
- `Conv2dLinearOperator`
  - 先递归 materialize `base` 的 output-side rows
  - 再走 `F.conv_transpose2d(...)`
  - `output_padding` 复用现有 `_conv_transpose_output_padding(...)`
- 其他未知 operator
  - 允许 fallback 到 `op.to_dense().reshape(B*K,C,H,W)`

因此 PR-4 重点优化的是 Conv path 本身，而不是一次性把所有 operator 的 norm 计算都改成结构化版本。

### 3. `Conv2dLinearOperator.row_abs_*` 不再依赖 `self.to_dense()`

更新：`boundflow/runtime/linear_operator.py`

`Conv2dLinearOperator` 的三个归约入口现在都改成：

1. `rows = _materialize_feature_map_rows(self, expected_input_shape=self.input_shape)`
2. `_reduce_feature_map_rows(...)`

具体保证：

- `row_abs_sum()` 与 `self.to_dense().abs().sum(dim=2)` 完全同值
- `row_l2_norm()` 与 `torch.linalg.vector_norm(self.to_dense(), ord=2, dim=2)` 完全同值
- `row_abs_max()` 与 `self.to_dense().abs().amax(dim=2)` 完全同值

`Conv2dLinearOperator.to_dense()` 本身没有改，仍保留为 debug/reference path。

## 测试

新增：`tests/test_phase7a_pr4_conv_lazy_norms.py`

覆盖：

- 单层 conv operator 的三个 row norm 与 dense reference 完全一致
- 嵌套 conv operator 的三个 row norm 与 dense reference 完全一致
- `monkeypatch` 掉 `Conv2dLinearOperator.to_dense` 后，三个 row norm 仍然可以正确工作
- `LpBallPerturbation.concretize_affine(center, conv_op, b)` 在 `p in {inf,2,1}` 下与 dense reference 完全一致

## 影响面

- 不改 `LinearOperator` protocol
- 不改 `PerturbationSet.concretize_affine(...)` 签名
- 不改 `CROWN` / `alpha` / `alpha-beta` / `BaB` 的公开语义
- 不改 `Conv2dLinearOperator.to_dense()` 行为

这次 PR 的本质是把 Conv operator 的 exact row norm 路径与 `to_dense()` 解耦，而不是扩新能力。

## 验证

已执行：

- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr4_conv_lazy_norms.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr3_conv_linear_operator.py tests/test_phase7a_pr3_highdim_concretize.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase7a_pr2_linear_operator_backward_state.py tests/test_phase6b_crown_ibp_mlp.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py`

结果：

- PR-4 新增测试：`6 passed`
- 回归测试：`34 passed`

## 后续建议

- 如果后面还要继续优化 Conv operator 的 norm 计算，下一步应单独评估：
  - 是否值得把最终 `[B*K,C,H,W]` rows 的 materialization 再往分块/流式归约推进
  - 是否要给 `RightMatmulLinearOperator` 也做类似 exact lazy norm

这两件事都不建议和 PR-4 混在一起。
