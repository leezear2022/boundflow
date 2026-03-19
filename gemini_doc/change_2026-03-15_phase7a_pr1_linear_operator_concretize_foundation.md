# 变更记录：Phase 7A PR1 `LinearOperator` 输入侧基础抽象

**日期**: 2026-03-15
**类型**: runtime / 基础设施
**范围**: `boundflow/runtime/`、`tests/`、`docs/`

---

## 背景与动机

Phase 6 已经把 CROWN、alpha-CROWN、alpha-beta-CROWN 与 BaB 的最小闭环跑通，但输入处的 `concretize_affine(...)` 仍要求显式稠密 `A:[B,K,I]`。这在当前链式 MLP 子集上可行，但一旦继续往 Conv 或更大规模 backward 扩展，就会立刻碰到“线性形式必须显式 materialize”的结构性瓶颈。

因此，本次 PR 不直接扩 Conv 或一般图，而是先做一个安全且可立即合入的基础设施步骤：

- 保持现有 Phase 6 语义、测试口径与工件口径不变；
- 只把输入边界的 affine concretize 抽象成可接受 `LinearOperator`；
- 先用 `DenseLinearOperator` 作为第一个实现，确保行为与当前 tensor 路径严格等价。

---

## 主要改动

- 新增：`boundflow/runtime/linear_operator.py`
  - 定义运行时内部 `LinearOperator` 协议；
  - 提供 `DenseLinearOperator`；
  - 提供 `as_linear_operator(...)` 归一化帮助函数。
- 更新：`boundflow/runtime/perturbation.py`
  - `PerturbationSet.concretize_affine(...)` 改为接受 `torch.Tensor | LinearOperator`；
  - `LpBallPerturbation.concretize_affine(...)` 统一通过 operator API 计算 `center_term` 与各类行范数。
- 更新：`boundflow/runtime/crown_ibp.py`
  - 在最终输入 `concretize_affine(...)` 处显式包一层 `DenseLinearOperator`。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - first-layer infeasible helper 在内部构造 `A` 时也统一走 `DenseLinearOperator` 路径。
- 新增：`tests/test_phase7a_linear_operator_concretize.py`
  - 覆盖等价性、防御性检查、round-trip，以及主路径确实走了 operator 路径的 spy 测试。

---

## 影响面

- 不改变 `TaskKind`、planner、TVM executor、artifact schema。
- 不改变现有 Phase 6 算法语义与公开 CLI。
- 不把 `LinearOperator` 混入 IR 或 JSONL，它只存在于 runtime 内部。
- 当前 backward 中间态 `A_u/A_l` 仍然保持 dense tensor，本次不做 lazy operator 化。

---

## 验证方式

- `python -m pytest -q tests/test_phase7a_linear_operator_concretize.py`
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`
- `python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py`
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`
- `python -m pytest -q tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 后续建议

- PR2：在线性 MLP 子集里把 backward 的 `A` 状态逐步 operator 化，而不只是在输入边界包装。
- PR3：在 `LinearOperator` 抽象稳定后，扩 Conv-ready 的 backward 表达与 soundness gate。
- PR4：在表达层稳定后，再推进 skip/branch/general graph 的 sound gate 与调度扩展。
