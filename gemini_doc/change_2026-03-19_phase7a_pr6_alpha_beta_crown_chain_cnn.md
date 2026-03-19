# 2026-03-19 Phase 7A PR-6：将 alpha-beta-CROWN 从 MLP 扩到 chain CNN

## 背景

PR-5 已经把 plain CROWN-IBP 和 alpha-CROWN 扩到链式 CNN 子集 `{conv2d,relu,flatten,linear}`，但 `alpha_beta_crown.py` 仍然停在 MLP-only：

- `run_alpha_beta_crown_mlp(...)` 对 `conv2d/flatten` 显式 fail-fast
- conv `relu_split_state` 仍然不能进入 `_forward_ibp_trace_mlp(...)`
- `BetaState` 和 split/beta 编码只支持 `[H]` / `[B,H]`
- first-layer infeasible detector 只支持 `input -> linear -> relu`
- `branch_choices` 只支持 rank-2 ReLU pre bound

PR-6 的目标是把 alpha-beta oracle 扩到 chain CNN，但明确不打开 conv BaB。

## 本次修改

### 1. 新增共享 ReLU shape/broadcast helper

新增：

- `boundflow/runtime/relu_shape_utils.py`

职责：

- `shape_numel(shape)`
- `relu_input_shapes(relu_pre)`
- `coerce_relu_param_shape(...)`
- `broadcast_relu_split_like_pre(...)`

这样 `alpha_crown.py`、`alpha_beta_crown.py`、`crown_ibp.py` 共享同一套高维 ReLU shape 归一逻辑，不再各自维护一套私有 helper。

### 2. conv split 真正进入 forward trace

更新：

- `boundflow/runtime/crown_ibp.py`

主要变化：

- `_apply_relu_split(...)` 从 rank-2 专用改成 rank-agnostic
- `relu_split_state` 现在支持：
  - shared：`[*S]`、`[I]`、`[1,*S]`、`[1,I]`
  - batch-specific：`[B,*S]`、`[B,I]`
- 内部统一归一到 flat `[B,I]`
- `_forward_ibp_trace_mlp(...)` 不再因为 conv split 直接 fail-fast

这一步使得 alpha-beta-CROWN 可以在 conv 图上真正携带 split 语义跑 forward trace。

### 3. alpha-CROWN 改用共享 helper，但不扩大 solver 边界

更新：

- `boundflow/runtime/alpha_crown.py`

主要变化：

- 删除本地 `_shape_numel/_relu_input_shapes/_coerce_shared_alpha_shape`
- 统一改用 `relu_shape_utils.py`

这里是内部复用重构，不改变 PR-5 已有的 alpha-CROWN 行为，只是避免 PR-6 再复制一套高维 shape 逻辑。

### 4. alpha-beta-CROWN 扩到 chain CNN

更新：

- `boundflow/runtime/alpha_beta_crown.py`

主要变化：

- 去掉 conv 图 fail-fast，`run_alpha_beta_crown_mlp(...)` 现在支持 chain CNN 子集 `{conv2d,relu,flatten,linear}`
- `AlphaState/BetaState` 初始化从“隐藏维度 `H`”改成基于 `relu_pre` 的逻辑 shape `[*S]`
- `per_batch_params=False` 时：
  - conv alpha/beta 存成 shared `[*S]`
- `per_batch_params=True` 时：
  - conv alpha/beta 存成 `[B,*S]`
- `_beta_to_relu_pre_add_coeff(...)` 现在支持高维 beta + 高维 split，统一输出 flat `[B,I]`
- `_branch_choices_from_relu_pre(...)` 现在支持任意 rank 的 ReLU pre bound，并继续返回 `(relu_input_name, flat_idx)`
- best-of 逻辑补齐高维参数：
  - per-batch 参数用 broadcast mask 做逐 batch 更新
  - shared 参数保留整体 snapshot 更新语义

### 5. first-layer conv infeasible detector

更新：

- `boundflow/runtime/alpha_beta_crown.py`

主要变化：

- `check_first_layer_infeasible_split(...)` 新增支持 `input -> conv2d -> relu`
- `_collect_first_layer_split_halfspaces(...)` 现在同时支持：
  - direct-input `linear`
  - direct-input `conv2d`

对 conv 的做法：

- 对每个被 split 的 conv output unit 构造 one-hot output row
- 通过 `DenseLinearOperator(...).conv2d_right(...).to_dense()` 提取该 unit 对输入的 affine row `a`
- 常数项 `c` 使用对应 output channel 的 bias
- 再乘 split sign `s`

数学边界保持明确：

- 只对 direct-input `linear/conv2d -> relu` 生效
- deeper-than-first-layer 的 split 不进入 halfspace 证书
- detector 会返回 `ok (no first-layer split halfspaces)`，但完整 alpha-beta oracle 仍然可以继续跑

### 6. conv BaB 继续 fail-fast

更新：

- `boundflow/runtime/bab.py`

变更：

- conv 图的拒绝文案更新为：
  - `BaB conv graphs not yet supported; PR6 only extends alpha-beta-CROWN oracle`

这一步是为了避免 PR-6 打开 oracle 后，`solve_bab_mlp(...)` 被动误开闸。

## 语义结果

PR-6 完成后：

- `run_alpha_beta_crown_mlp(...)` 支持 chain CNN 子集 `{conv2d,relu,flatten,linear}`
- conv `relu_split_state` 支持 shared + batch-specific 高维形状
- conv `alpha/beta` 完全沿用 `per_batch_params` 语义
- `AlphaBetaCrownStats.branch_choices` 在 conv 图上继续返回 flat index
- first-layer direct-input conv split 可以被 infeasible detector 证伪
- deeper conv split 不误用 first-layer halfspace 证书
- `solve_bab_mlp(...)` 对 conv 图仍然保持 fail-fast

## 新增/更新测试

新增：

- `tests/test_phase7a_pr6_alpha_beta_crown_cnn.py`

覆盖：

- chain CNN 上 `run_alpha_beta_crown_mlp(...)` steps=0 可运行且 sound
- structured split 与 flat split 等价
- per-batch alpha/beta 参数与 warm-start
- conv beta 梯度非零且有限
- alpha-beta 优化不劣于 step=0
- `branch_choices` 返回 flat idx
- first-layer conv infeasible detector 可证矛盾
- deeper-than-first-layer conv split 不进入 halfspace 证书，但 oracle 仍可运行

同步更新：

- `tests/test_phase7a_pr5_alpha_crown_cnn.py`

把 PR-5 里“conv split_state 仍然不支持”的旧断言改成 PR-6 后的新行为检查。

## 验证

已执行：

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7a_pr6_alpha_beta_crown_cnn.py
conda run -n boundflow python -m pytest -q tests/test_phase7a_pr5_alpha_crown_cnn.py tests/test_phase7a_pr4_conv_lazy_norms.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py tests/test_phase6e_bab_mlp.py
```

结果：

- PR-6 专项：`6 passed`
- 定向回归：`30 passed`

## 非目标

本次 PR-6 仍然**不**做：

- conv BaB
- conv node batch / branch picking 落到 BaB 主循环
- skip/branch/general DAG
- deeper-than-first-layer infeasible split 证书
