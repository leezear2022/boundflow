# 2026-03-16：Phase 7A PR-2——线性 MLP backward `A` 状态 operator 化

## 背景与动机

Phase 7A PR-1 已经把输入边界 `concretize_affine(...)` 抽象成可接受 `LinearOperator`，但 CROWN/alpha-beta 主路径里的 backward `A_u/A_l` 仍然是全程显式 dense tensor。继续直接做 Conv 或一般图，会把“线性表达抽象”和“算子覆盖扩展”绑死在一起，返工风险高。

因此这次 PR-2 的目标收敛为：
- 仍然只服务于链式 MLP。
- 把 linear backward 改成 operator 路径。
- 保持 ReLU backward 仍然是显式 dense barrier。
- 不改 IR、planner、CLI、artifact schema。

## 主要改动

### 1. 扩展运行时内部 `LinearOperator`

更新：`boundflow/runtime/linear_operator.py`

- 在现有 `shape/device/dtype/center_term/row_norm/to_dense` 基础上，新增：
  - `contract_last_dim(vec)`：支持 `A @ bias` 这类最后一维收缩。
  - `matmul_right(rhs)`：支持 backward 里的 `A_next = A @ W`。
- 新增 `RightMatmulLinearOperator`：
  - 表示 `base @ rhs` 的 lazy 组合。
  - `center_term` 和 `contract_last_dim` 走解析式，不要求立刻 materialize。
  - `row_abs_sum` / `row_l2_norm` / `row_abs_max` 当前允许内部 `to_dense()`，保持实现范围可控。
  - 连续 `matmul_right(...)` 会融合成一个更短的组合链，避免无谓嵌套。

### 2. 收敛 `crown_ibp.py` 的 backward 逻辑

更新：`boundflow/runtime/crown_ibp.py`

- 新增共享内部状态 `AffineBackwardState`，统一管理：
  - `A_u`
  - `A_l`
  - `b_u`
  - `b_l`
- 新增共享 helper：
  - `_init_backward_state(...)`
  - `_backprop_linear_step(...)`
  - `_backprop_relu_step(...)`
  - `_run_crown_backward_from_trace(...)`
- `run_crown_ibp_mlp(...)` 和 `run_crown_ibp_mlp_from_forward_trace(...)` 现在共用同一套 backward 实现。

### 3. linear backward 现在真正走 operator 路径

更新：`boundflow/runtime/crown_ibp.py`

- 遇到 `linear` 时：
  - `b += A.contract_last_dim(bias)`
  - `A = A.matmul_right(W)`
- 不再直接对 raw tensor `A_u/A_l` 做 `matmul/einsum`。
- 最终输入 concretize 直接把 `LinearOperator` 传给 `concretize_affine(...)`，不再只是在最后包一层 `DenseLinearOperator`。

### 4. ReLU backward 明确保留为 dense barrier

更新：`boundflow/runtime/crown_ibp.py`

- 遇到 `relu` 时先 `to_dense()`，再沿用当前 sign-based 逻辑：
  - `torch.where(A >= 0, ...)`
  - `b += (A * beta).sum(dim=2)`
  - `A = A * alpha`
  - `relu_pre_add_coeff_{u,l}` 广播加法不变
- ReLU 处理结束后重新包成 `DenseLinearOperator`。

这一步不是要把 backward 一次性全部 lazy 化，而是先把“可以 lazy 的 linear 段”和“必须 dense 的 ReLU 段”切开。

### 5. 新增 PR-2 专项测试

新增：`tests/test_phase7a_pr2_linear_operator_backward_state.py`

覆盖：
- `RightMatmulLinearOperator` 与 dense 参考实现的数值等价。
- 连续 `matmul_right(...)` 的融合行为。
- 非法 `rhs/vec` 的 shape 与 dtype 防御。
- 纯线性链里 `run_crown_ibp_mlp(...)` 确实调用了 operator 的 `matmul_right(...)`。
- 含 `linear -> linear -> relu -> linear` 的混合链仍保持 soundness。

同步更新：`tests/test_phase7a_linear_operator_concretize.py`

- 将原先“必须是 `DenseLinearOperator`”的断言放宽成“必须是 `LinearOperator`”，以反映 PR-2 后主路径会把 lazy operator 直接传到输入边界。

## 影响面

- 不改 Python API、CLI、artifact schema、IR、planner。
- `alpha_crown.py` 与 `alpha_beta_crown.py` 不需要重写主算法；它们通过共享的 CROWN backward 路径自动继承 PR-2。
- 这一步不承诺立即带来性能收益；主要收益是把表达层切开，为 PR-3 的 Conv-ready operator 和 PR-4 的一般图扩展减少返工。

## 验证

已执行：

- `python -m pytest -q tests/test_phase7a_pr2_linear_operator_backward_state.py`
- `python -m pytest -q tests/test_phase7a_linear_operator_concretize.py`
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`
- `python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py`
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`
- `python -m pytest -q tests/test_phase6g_alpha_beta_multispec_batch.py`

## 后续建议

- PR-3：在当前 operator 边界明确的前提下，继续做 Conv-ready backward 表达。
- PR-4：把 chain-structured MLP 的限制往 skip/branch/general graph 推进，并补 sound gate 与调度扩展。
