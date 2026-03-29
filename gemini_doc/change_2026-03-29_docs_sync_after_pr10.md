# 变更记录：PR-10 后文档同步与发布准备

**日期**: 2026-03-29
**类型**: docs
**范围**: 根文档、统一参考、LLM 交接、下一步计划

---

## 动机

2026-03-28 新增的文档整合与交接文档已经基本成型，但其中部分“当前前沿 / 下一步计划”仍停留在 PR-10 实施前的状态。现在 PR-10 与 shared CROWN layout-only support 已完成，需要把这些文档同步到当前事实，再统一提交与推送。

## 主要改动

- 更新：`CLAUDE.md`
  - 当前前沿改为“已完成 structured ReLU backward + layout-only shared CROWN”
  - 下一步计划改指向 `gemini_doc/next_plan_after_phase7a_pr10.md`
- 更新：`AGENTS.md`
  - “当前计划”索引改为 `gemini_doc/next_plan_after_phase7a_pr10.md`
- 更新：`README.md`
  - 在系统特性中补充 shared CROWN 的结构化 ReLU/DAG backward 与 layout-only op 支持
- 更新：`docs/reference.md`
  - runtime/linear operator 描述同步到 structured ReLU 与 layout-only support 的现状
- 更新：`gemini_doc/README.md`
  - “当前前沿 / 下一步”索引同步到 PR-10 之后
- 更新：`gemini_doc/llm_briefing_boundflow.md`
  - 顶部更新时间刷新到 2026-03-29
  - Phase 7A 当前前沿同步加入 PR-10 与 layout-only support
  - 已知限制中的 ReLU barrier 表述改为“剩余 sign-split dense 点”
- 新增：`gemini_doc/next_plan_after_phase7a_pr10.md`
  - 记录 PR-10 后的下一步：benchmark、继续消除 `split_pos_neg()` dense 点、补性能/结构测试

## 影响面

- 不影响代码和测试语义。
- 根文档、统一参考、LLM 交接文档、Agent 指引之间的口径重新对齐。
- 后续 PR 若引用“当前计划”，可以直接落到 PR-10 之后的计划文档，而不是继续指向历史计划。
