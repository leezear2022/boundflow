# 变更记录：Phase 6E 文档口径补强——明确 sound≠complete + 给出 Phase 6F（β/αβ-CROWN MLP）落地计划

## 动机

Phase 6E 已实现 split state + α-CROWN bound oracle + BaB driver 的最小闭环，并包含 1D Linf toy 的 complete 演示补丁。
为了避免论文/评审/合作者误解，需要在文档中明确：

- 一般情形下该实现是 **sound** 但不保证 **complete**；
- toy complete 只对特定补丁成立；
- 通用 complete 需要进入 Phase 6F（β/αβ-CROWN 风格的 split constraint encoding 或更强约束求解）。

## 本次改动

- 更新：`gemini_doc/change_2025-12-23_phase6e_bab_mlp_split_state_and_driver.md`
  - 增加 “关于 complete 的口径” 小节，明确 sound≠complete，并将 1D Linf patch 定位为 demo-only。

- 新增：`gemini_doc/phase6f_beta_crown_mlp_plan.md`
  - 给出 Phase 6F 的 PR 级落地清单：数据结构（BetaState 稀疏存储）、API 签名、DoD、必须过的测试集合，以及明确“不做”的范围（避免返工/膨胀）。

## 如何验证

- 本次为文档变更，无需额外运行时验证；代码路径不变。

