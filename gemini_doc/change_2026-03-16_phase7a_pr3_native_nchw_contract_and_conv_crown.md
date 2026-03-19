# 2026-03-16：Phase 7A PR-3——原生 NCHW contract 与 Conv-ready CROWN-IBP

## 背景与动机

Phase 7A PR-1 把输入边界 `concretize_affine(...)` 抽象成可接受 `LinearOperator`，PR-2 又把链式 MLP 的 linear backward `A_u/A_l` 状态 operator 化。但当时 runtime 的公开 contract 仍然默认“输入是扁平 `[B, I]`”，这会让后续 Conv 扩展继续在接口层反复做 flatten/unflatten 适配。

因此这次 PR-3 选择把 contract 往前推进一层：
- `PerturbationSet.concretize_affine(...)` 原生接受高维输入 `center:[B,*input_shape]`。
- `LinearOperator` 原生携带 `input_shape`，不再把高维输入只当成内部细节。
- plain CROWN-IBP 从链式 MLP 扩到链式 CNN 子集 `{conv2d,relu,flatten,linear}`。
- `alpha_crown.py`、`alpha_beta_crown.py` 明确保持 MLP-only，不跟着这次一起扩 solver 语义。

这一步更接近 vendored `auto_LiRPA` 的方向：公开层承认高维/NCHW 输入，内部仍允许 dense matrix path 和结构化 conv path 并存，而不是一步强推完整 patches 生态。

## 主要改动

### 1. 升级 `LinearOperator` 的高维输入 contract

更新：`boundflow/runtime/linear_operator.py`

- `LinearOperator` 协议新增：
  - `input_shape`
  - `input_numel`
  - `spec_dim`
  - `contract_input(...)`
  - `reshape_input(...)`
  - `conv2d_right(...)`
- `DenseLinearOperator` 现在同时携带：
  - `coeffs:[B,K,*input_shape]` 或 `[B,K,I]`
  - `input_shape`
- 保留 `matmul_right(...)`，但明确只用于扁平输入轴的 linear 段。
- 新增 `ReshapeInputLinearOperator`
  - 用于在 flat `<->` `NCHW` 的逻辑输入形状之间切换，不改变数值线性形式。
- 新增 `Conv2dLinearOperator`
  - 表示 `A @ Conv2d(x)` 的 lazy operator。
  - `center_term(...)` / `contract_input(...)` 直接接受 NCHW 输入。
  - `to_dense()` 允许通过 `conv_transpose2d` materialize 成扁平 `[B,K,I]`，主要用于 debug 与 row norm 计算。

### 2. `PerturbationSet.concretize_affine(...)` 现在原生接受高维 `center`

更新：`boundflow/runtime/perturbation.py`

- `LpBallPerturbation.concretize_affine(...)` 从只接受 `center:[B,I]` 扩成接受 `center:[B,*input_shape]`。
- 对 tensor `A`，现在支持：
  - `[B,K,I]`
  - `[B,K,*input_shape]`
- 对 operator `A`，会用 `A.input_shape` 与 `center.shape[1:]` 做显式校验。
- `Linf/L2/L1` 的 deviation 逻辑不变；只是把“输入是否高维”的责任前移到 operator contract。

### 3. `crown_ibp.py` 扩到 chain CNN 子集

更新：`boundflow/runtime/crown_ibp.py`

- forward trace `_forward_ibp_trace_mlp(...)` 新支持：
  - `conv2d`
  - `flatten(start_dim=1, end_dim=-1)`
- 对 `InputPerturbationState + conv2d`：
  - 先走 `bounding_box(...)`
  - 再用 `IntervalDomain.affine_transformer(..., op="conv2d")`
  - 这对 `Linf` 是自然路径；对 `L2/L1` 是 sound 的 box downgrade。
- backward 新增：
  - `_backprop_flatten_step(...)`
  - `_backprop_conv2d_step(...)`
- backward 规则变成：
  - `linear`：继续走 PR-2 的 `matmul_right(...)`
  - `flatten`：用 `reshape_input(...)` 恢复 pre-shape
  - `conv2d`：用 `conv2d_right(...)` 生成 `Conv2dLinearOperator`，并显式加上 broadcast bias 项
  - `relu`：对高维 pre-bound 临时 flatten 成 `[B,I]` 做 dense barrier，处理完后恢复原始 `input_shape`
- `get_crown_ibp_mlp_stats(...)` 虽然名字未改，但已接受链式 `{conv2d,relu,flatten,linear}` 图；它现在更准确地表示 plain chain CROWN-IBP 的支持性检查。

### 4. 明确划出 solver 边界

更新：
- `boundflow/runtime/alpha_crown.py`
- `boundflow/runtime/alpha_beta_crown.py`

- 若图中出现 `conv2d` 或 `flatten`，现在直接 fail-fast：
  - `alpha-crown conv graphs not yet supported; PR3 only extends plain CROWN-IBP`
  - `alpha-beta-crown conv graphs not yet supported; PR3 only extends plain CROWN-IBP`

这能避免因为 forward trace 现在支持 Conv，而让 alpha/beta/BaB 路径在更深处以不明确方式失败。

### 5. 新增 PR-3 专项测试

新增：
- `tests/test_phase7a_pr3_highdim_concretize.py`
- `tests/test_phase7a_pr3_conv_linear_operator.py`
- `tests/test_phase7a_pr3_crown_ibp_cnn.py`

覆盖点：
- 高维 `center` 与高维 tensor/operator `A` 的 `concretize_affine(...)` 等价性。
- `Conv2dLinearOperator.center_term(...)` / `contract_input(...)` / `to_dense()` 与显式 dense 参考的一致性。
- 链式 CNN `conv2d -> relu -> conv2d -> relu -> flatten -> linear` 在 `Linf/L2` 下的 soundness。
- batched `linear_spec_C:[B,S,O]` 在 CNN 链上的 serial/batched 一致性。
- `get_crown_ibp_mlp_stats(...)` 支持 chain CNN，并继续拒绝 skip-like 非链式图。

## 影响面

- 公开 runtime contract 升级为原生支持高维输入，但当前承诺范围只到：
  - rank-2 flat 输入
  - rank-4 `NCHW` conv2d 输入
- 不改 IR、planner、CLI、artifact schema。
- 不引入任意 rank 泛化，也不引入完整 `Patches` 风格结构化张量体系。
- `alpha_crown.py`、`alpha_beta_crown.py`、`bab.py` 仍然是 MLP-only；PR-3 没有把 solver 语义一起扩到 CNN。

## 验证

已执行：

- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr3_highdim_concretize.py tests/test_phase7a_pr3_conv_linear_operator.py tests/test_phase7a_pr3_crown_ibp_cnn.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr2_linear_operator_backward_state.py tests/test_phase7a_linear_operator_concretize.py tests/test_phase6b_crown_ibp_mlp.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py`

结果：

- PR-3 新增测试：`10 passed`
- 回归测试：`34 passed`

## 后续建议

- PR-4 可以继续往两个方向推进，但不要混在同一批里：
  - 把 alpha/beta/BaB 的 CNN 语义单独展开。
  - 把当前 `Conv2dLinearOperator` 的 `row_abs_*` 从 `to_dense()` 近路进一步替换成真正 lazy 的归约实现。
