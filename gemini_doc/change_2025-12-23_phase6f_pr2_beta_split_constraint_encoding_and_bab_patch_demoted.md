# 变更记录：Phase 6F PR-2（β/αβ-CROWN MLP）——β 真实 split-constraint encoding + 可证空域（非 pairwise）+ BaB 1D patch 降级

## 动机

Phase 6F PR-1 只把 β 做成“可微占位符（penalty）”，用于：

- 把 β 放进 autograd 图（梯度钉子能测）；
- 给 BaB 提供 `feasibility`/prune 的接口形态。

但 PR-1 仍然缺少 Phase 6F 的核心语义：**把 split 约束编码进 bound propagation**。这会导致两个工程问题：

1. BaB 的 “complete” 支点不稳：当 split 约束需要被真正用于收紧（例如 `x>=0` 这种子域约束）时，如果 oracle 仍在原始输入域上做 concretize，就会出现 leaf bound 过松甚至错误地返回 `unsafe` 的情况。
2. 空域识别过弱：PR-1 只能抓 “相反权重 pairwise” 的矛盾组合，无法覆盖 “需要多个约束线性组合才能推出矛盾” 的非平凡空域。

本 PR-2 目标是：把 β 从占位符升级成 **真实的 split-constraint Lagrangian encoding**，并把 BaB 的 1D Linf 输入域收缩补丁降级为可选开关（默认关闭），让 “complete 的锚” 回到 β 本身。

## 本次改动

### 1) β 真实 encoding：Lagrangian 注入到 CROWN-IBP backward

- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., relu_pre_add_coeff_l/relu_pre_add_coeff_u=...)`：允许在 ReLU backward 处对 **pre-activation** 的线性系数做“额外注入”（后续用于 β/其它约束的对偶变量）。

- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 移除 PR-1 的 `beta_penalty_scale` 占位符路径。
  - β 的 split-constraint encoding：
    - split 约束：`s * z >= 0`（`s∈{-1,+1}`）
    - 标准化为不等式：`g(z) = -s*z <= 0`
    - lower-bound 的对偶松弛：在目标中加入 `+ β*g(z) = -β*s*z`（`β>=0`）
    - 对应到实现：把 `-β*s` 作为“对 pre-activation 的附加线性系数”，注入到 `run_crown_ibp_mlp` 的 backward。
  - β 的投影：每步优化后对 β 做 `clamp_(0)`，保证 `β>=0`。

### 2) 空域识别升级：支持“非 pairwise”证书

- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - `*_convex_combo` infeasible 检测：对 first-layer split 约束构造半空间 `a_i·x + c_i >= 0`，搜索 simplex 上的 convex combination，使
    - `max_{x∈S} ( (Σ w_i a_i)·x + Σ w_i c_i ) < 0`
    - 作为 **可证明空域** 的证书（证书含权重与组合后的 `(a,c)`）。
  - 性能/稳定性：当当前迭代已经得到 `max<0` 的证书时直接 early-exit（避免无谓的优化迭代拖慢单测）。

### 3) BaB：1D Linf 输入域收缩补丁降级为可选开关

- 更新：`boundflow/runtime/bab.py`
  - `BabConfig.use_1d_linf_input_restriction_patch: bool = False`
  - 默认不再启用 `_restrict_input_spec_linf_1d_for_first_layer_splits(...)`，避免 “complete 依赖 patch”。
  - Phase 6E 的历史 toy 仍可通过显式打开该开关复现。

### 4) 测试（DoD）

- 更新：`tests/test_phase6e_bab_mlp.py`
  - 明确 Phase 6E 的 toy complete 仍使用 patch：`use_1d_linf_input_restriction_patch=True`（作为“历史演示路径”保留）。

- 更新：`tests/test_phase6f_alpha_beta_crown_pr1.py`
  - β 梯度钉子：直接对 `run_crown_ibp_mlp(..., relu_pre_add_coeff_l=...)` 做反传，验证 β 的梯度非空、finite、非零。
  - 空域证书：
    - 仍覆盖 2 条约束的矛盾；
    - 新增 3 条约束线性组合才能推出矛盾的 case（非 pairwise）。
    - 新增 2D L2 球的“三方向（0/120/240°）”干净证书用例（w=1/3 的凸组合给出显式矛盾）。
  - BaB 回归：在 1D toy 上证明 αβ-CROWN 可以在 **不启用 patch** 的前提下恢复 complete（α-only 会返回 `unsafe`，αβ 会 `proven`）。
  - best-of 回归：`steps>0` 的结果不劣于 `steps=0`（包含 step=0 的 best-of 语义）。

## 如何验证

```bash
python -m pytest -q \
  tests/test_phase6b_crown_ibp_mlp.py \
  tests/test_phase6d_alpha_crown_mlp.py \
  tests/test_phase6e_bab_mlp.py \
  tests/test_phase6f_alpha_beta_crown_pr1.py
```

## 已知限制（TODO）

- 当前 infeasible 证书只覆盖 “first-layer split constraints”的可证空域检测；更一般的多层约束编码与证书还需要继续推进。
- `run_alpha_beta_crown_mlp` 当前以 BaB 常用的 `objective="lower"` 为主线；其它 objective 的 β 方向与 tightness 口径可后续补齐。
