# 变更记录：Phase 6F 计划补强——complete 锚点 + BetaState 存储策略调整 + 非平凡空域 DoD

## 动机

Phase 6F（β/αβ-CROWN MLP）是从 “sound 但不保证 complete” 走向 “面向 complete verification 的 split constraint encoding” 的关键阶段。为减少后续实现返工、并让文档口径更 reviewer-proof，本次对 6E/6F 文档做三处补强：

1) 在 6E 文档中加入 “β-CROWN + BaB（ReLU splitting）soundness & completeness” 的权威锚点（Theorem 3.3）；
2) 调整 Phase 6F 的 `BetaState` 建议：先 dense（或 dense+mask）保证 autograd/并行友好，稀疏化延后到 6G；
3) 将 Phase 6F 的 infeasible split DoD 从“平凡冲突”提升为“非平凡空域”（更能体现 β encoding 的价值）。

## 本次改动

- 更新：`gemini_doc/change_2025-12-23_phase6e_bab_mlp_split_state_and_driver.md`
  - 在 “complete 口径” 处加入 Theorem 3.3 的对照锚点链接。

- 更新：`gemini_doc/phase6f_beta_crown_mlp_plan.md`
  - `BetaState` 建议改为 Phase 6F 先 dense（`Dict[str, Tensor[H]]`）并从 split_state 派生 mask；稀疏化延后至 6G。
  - DoD-2 由“同 neuron active/inactive 冲突”改为“非平凡空域”测试描述，避免被 reviewer 认为是 merge check 就能解决的 trivial case。

## 如何验证

- 本次为计划/文档变更，无需运行时验证；代码路径不变。

