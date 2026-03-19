# 2026-03-19：Phase 7A PR-5——将 alpha-CROWN 从 MLP 扩到 chain CNN

## 背景与动机

Phase 7A PR-3/PR-4 已经把 plain CROWN-IBP 和 `Conv2dLinearOperator` 扩到链式 CNN 子集 `{conv2d,relu,flatten,linear}`，但 `alpha_crown.py` 仍然停在 MLP-only：

- `run_alpha_crown_mlp(...)` 对含 `conv2d/flatten` 的图直接 fail-fast；
- `run_crown_ibp_mlp(..., relu_alpha=...)` 对 rank>2 的 ReLU pre-activation 直接拒绝；
- `alpha_beta_crown.py` 与 `bab.py` 还都默认更上层 solver 仍是 MLP-only。

因此 PR-5 的目标很收敛：

- 只扩 `alpha-CROWN` 到 chain CNN；
- 只支持逐元素共享 alpha，不引入 per-batch 参数化；
- 不把 `alpha-beta-CROWN`、`BaB`、conv `split_state` 一起带进来。

## 主要改动

### 1. `crown_ibp.py` 支持高维 ReLU alpha

更新：`boundflow/runtime/crown_ibp.py`

- `_broadcast_relu_alpha(...)` 从只支持 rank-2 扩到支持高维 ReLU pre bound。
- 允许的 alpha 形状：
  - scalar `[]`
  - 逻辑形状 `[*S]`
  - flatten 后的 `[I]`
  - batch-leading singleton `[1,*S]`
  - batch-leading singleton flat `[1,I]`
- 明确拒绝 batch-specific alpha：
  - `[B,*S]`
  - `[B,I]`

内部仍统一广播成 flat `[B,I]`，所以 backward 侧原有的 dense ReLU barrier 逻辑不需要改语义，只是把 conv ReLU 的 structured alpha 也接进来了。

### 2. `alpha_crown.py` 从静态维度推断改成 forward-trace 驱动

更新：`boundflow/runtime/alpha_crown.py`

- 删除原来的 `_relu_input_dims(module)` 线性层静态推断路径。
- 改为先跑一次 `_forward_ibp_trace_mlp(...)`，从 `relu_pre` 直接读取每个 ReLU 输入的逻辑 shape。
- `AlphaState.alpha_by_relu_input[name]` 现在既可以是 MLP 的 `[H]`，也可以是 conv ReLU 的 `[C,H,W]`。

这让 alpha 的形状来源与真实 forward trace 对齐，不再依赖“ReLU 前驱一定是 rank-2 linear”这个 MLP-only 假设。

### 3. `run_alpha_crown_mlp(...)` 改成 forward-trace reuse

更新：`boundflow/runtime/alpha_crown.py`

- 入口先执行一次 `_forward_ibp_trace_mlp(module, input_spec, relu_split_state=...)`
- 优化循环内改成调用 `run_crown_ibp_mlp_from_forward_trace(...)`
- 不再每一步都重跑完整 forward IBP

这在语义上不改变 alpha-CROWN 的目标函数，但把 PR-5 的 shape 初始化和优化循环统一到了同一份 forward trace 上，也避免了 conv 图上重复 forward 的额外开销。

### 4. warm-start 支持高维共享 alpha

更新：`boundflow/runtime/alpha_crown.py`

`_init_alpha_state(...)` 现在支持把 warm-start alpha 归一到逻辑 shape：

- scalar `[]`
- exact logical shape `[*S]`
- flatten `[I]`
- `[1,*S]`
- `[1,I]`

仍然明确拒绝：

- `[B,*S]`
- `[B,I]`

也就是说，PR-5 固定选择“跨 batch/spec 共享”的 alpha 语义，而不是 per-batch 独立 alpha。

### 5. conv `split_state` 继续保持禁用

更新：`boundflow/runtime/crown_ibp.py`

`_forward_ibp_trace_mlp(...)` 对 rank>2 的 `relu_split_state` 现在报更明确的错误：

- 说明当前只在 rank-2 pre-activation 上支持 split-state
- 同时明确写出 `conv split_state remains unsupported until alpha-beta/BaB PRs`

这让 PR-5 的边界更清楚：conv 图上只扩 alpha-CROWN，不扩 split/beta/BaB。

### 6. `alpha_beta_crown.py` / `bab.py` 收紧 conv 边界

更新：

- `boundflow/runtime/alpha_beta_crown.py`
- `boundflow/runtime/bab.py`

改动：

- `alpha-beta-CROWN` 对 conv 图继续 fail-fast，但文案改成：
  - `alpha-beta-CROWN remains MLP-only on conv graphs; PR5 extends alpha-CROWN only`
- `solve_bab_mlp(...)` 在入口处显式拒绝 `conv2d/flatten` 图：
  - `BaB conv graphs not yet supported; PR5 only extends alpha-CROWN`

这样即使 `run_alpha_crown_mlp(...)` 已经支持 conv 图，也不会误把上层 complete solver 路径被动打开。

## 新增测试

新增：`tests/test_phase7a_pr5_alpha_crown_cnn.py`

覆盖：

- chain CNN toy 上 alpha 优化能改进 lower bound，且 soundness 通过
- conv 图上的 warm-start 不劣于 cold-start
- structured alpha（例如 `[C,H,W]`）可反传，梯度非零且有限
- structured alpha 与 flat alpha 数值等价
- conv `relu_split_state` 继续明确不支持
- `alpha_beta_crown.py` / `bab.py` 对 conv 图继续 fail-fast

## 影响面

- `run_alpha_crown_mlp(...)` 现在支持链式 CNN 子集 `{conv2d,relu,flatten,linear}`
- `run_crown_ibp_mlp(..., relu_alpha=...)` 现在支持高维 shared alpha
- `AlphaState` 的值张量不再只限于 `[H]`，还可能是 `[C,H,W]`
- `alpha-beta-CROWN` 与 `BaB` 仍然保持 MLP-only
- conv `split_state` 仍未开放

这次 PR 的重点是把“更强的 incomplete bound”接到 chain CNN 上，而不是把 complete solver 一起带过去。

## 验证

已执行：

- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr5_alpha_crown_cnn.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py tests/test_phase6e_bab_mlp.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase7a_pr4_conv_lazy_norms.py`

结果：

- PR-5 新增测试：`6 passed`
- Phase 6 alpha/alpha-beta/BaB 回归：`14 passed`
- PR-3/PR-4 CNN 回归：`10 passed`

## 后续建议

- 下一步自然是 `alpha-beta-CROWN on chain CNN`
- 但建议把它单独做成下一 PR，不要和 `BaB on chain CNN` 混在一起
- 因为 conv `split_state`、beta 编码、branch picking 和 node batching 会一起引入新的 shape/语义边界
