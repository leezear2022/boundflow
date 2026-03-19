# 变更记录：Phase 6F PR-1（β/αβ-CROWN MLP）——αβ oracle 闭环 + `feasible` 标志 + β 梯度钉子 + 非平凡空域

## 动机

Phase 6E 已明确 “sound≠complete”，并给出 Phase 6F 的计划：把 split 约束更系统地编码进 bound propagation（β/αβ-CROWN 风格）。
本 PR-1 先按“接口/可微/可剪枝优先”的顺序落地最小闭环：

- dense `BetaState` 形态落地（便于 autograd/batching，稀疏化留到 6G）；
- 新增 `run_alpha_beta_crown_mlp(...)` oracle，返回 `stats.feasible`，让 BaB 能将空域当成一等公民剪枝；
- 增加 β 梯度回归钉子（防 silent bug）；
- 增加一个“非平凡空域”用例（联合 split 约束导致矛盾），避免被误认为是 trivial merge check。

## 本次改动

- 新增：`boundflow/runtime/alpha_beta_crown.py`
  - `BetaState`：dense 存储 `beta_by_relu_input: Dict[str, Tensor[H]]`。
  - `run_alpha_beta_crown_mlp(...) -> (bounds, alpha_state, beta_state, stats)`：
    - 返回 `AlphaBetaCrownStats(feasible, reason, infeasible_witness, ...)`；
    - PR-1 先复用 `run_alpha_crown_mlp` 做 α 优化；
    - β 通过 conservative penalty（loosen bounds）进入 autograd 图，以保证梯度链路可测（真实 β encoding 将在后续 PR-2 落地）。
  - `_is_infeasible_split_first_layer_pairwise(...)`：best-effort 的 first-layer split 矛盾检测（用于非平凡空域回归）。

- 更新：`boundflow/runtime/bab.py`
  - `BabConfig.oracle: {"alpha","alpha_beta"}`：支持切换 bound oracle；
  - 当 `oracle="alpha_beta"` 时调用 `run_alpha_beta_crown_mlp`，若 `feasible=False` 则直接 prune 节点；
  - 节点 warm-start 透传 `warm_start_beta`（暂为接口形态，真实 β encoding 后续完善）。

- 新增测试：`tests/test_phase6f_alpha_beta_crown_pr1.py`
  - DoD-1（β 梯度钉子）：`loss=-lb.mean()` 反传，断言 `beta.grad` 非空、finite、非零。
  - DoD-2（非平凡空域）：2D 输入、first-layer 两个 neuron 权重互为相反数，split 两者 active 导致联合矛盾，断言 `stats.feasible=False` 且有 witness。

## 如何验证

```bash
python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py
```

## 已知限制（TODO）

- PR-1 的 β 仍是 “可微占位符（penalty）”，真实的 split constraint encoding（β-CROWN/αβ-CROWN 核心）需要在 Phase 6F PR-2 落地。
- infeasible 检测目前是 first-layer 的 pairwise 规则（用于打掉非平凡空域回归）；更一般的空域识别需要后续更系统的编码/优化。

## 语义护栏（PR-1 必须守住）

- `stats.feasibility` 采用二态（`"unknown"|"infeasible"`）：
  - `"infeasible"` 仅在 **可证明空域** 时返回（可安全 prune）；
  - `"unknown"` 表示未发现矛盾（不代表可行，不能用于剪枝）。
- PR-1 的 β 只用于“可微链路与接口形态”，并且 **只允许放松 bounds**（下界只会更小、上界只会更大），避免破坏 soundness。
