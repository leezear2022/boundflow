# 变更记录：新增研发脉络总览文档

**日期**: 2026-03-15
**类型**: 文档 / 导航
**范围**: `gemini_doc/`、`docs/`

---

## 背景与动机

仓库里的阶段总结、总账、全流程文档和设计评审已经比较完整，但它们分别回答的是不同问题：

- `docs/change_log.md` 负责记录时间顺序上的修改总账；
- `gemini_doc/phase*_summary.md` 负责按阶段做收官总结；
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 更偏 claim 到 AE 的全流程；
- `gemini_doc/phase6_review_three_axis_stage_pipeline.md` 更偏设计评审与落地约束。

现有材料并不缺，但缺少一篇面向“项目演化主线”的总整理文档，帮助接手者快速回答：

1. BoundFlow 想做什么；
2. 这条线是如何从 Phase 0 推到 Phase 6 的；
3. 不同阶段主要改动了哪些代码模块；
4. 现有记录分别在哪里；
5. 现在做到哪，下一步自然要做什么。

---

## 新增文档内容概览

新增文档：`gemini_doc/project_evolution_overview.md`

该文档的定位不是替代已有阶段总结，而是作为：

- 入口层：第一次进入仓库时先读它；
- 导航层：告诉读者接下来该跳到哪些文档；
- 演化视角层：把 Phase 0~6 的主线串成一条连续研发脉络。

文档内容覆盖：

- BoundFlow 的系统定位与论文叙事；
- 从前端、IR、planner、runtime、artifact 到 alpha-beta/BaB/E2E 的主线演进；
- 各阶段的核心代码落点与测试入口；
- 现有总账、阶段总结、全流程文档、设计评审的分工；
- 对当前仓库状态的客观判断，以及下一步自然路线。

---

## 为什么现有文档还不够

现有文档的粒度主要有两种：

- 过细：如 `docs/change_log.md` 和大量 `change_YYYY-MM-DD_*.md`，适合追溯某次修改，但不适合作为接手总入口；
- 过专：如 Phase 总结、AE 文档、设计评审，适合回答单个阶段或单个主题，但不直接承担“研发演化主线总整理”的角色。

因此，新文档的价值不是新增事实，而是把已有事实重新组织成一篇“高层入口 + 明确导航 + 可接手”的总览。

---

## 本次更新的文件列表

- 新增：`gemini_doc/project_evolution_overview.md`
- 新增：`gemini_doc/change_2026-03-15_add_project_evolution_overview.md`
- 更新：`gemini_doc/README.md`
- 更新：`docs/change_log.md`

---

## 验证方式

本次仅涉及文档整理，不涉及代码语义改动。验证方式以文档一致性为主：

- 检查 `gemini_doc/project_evolution_overview.md` 中引用的路径均存在；
- 检查 `gemini_doc/README.md` 已纳入新文档并补充“研发演化/接手视角”阅读路径；
- 检查 `docs/change_log.md` 已追加本次修改总账；
- 确认新文档不替代 `phase*_summary.md`，而是作为入口层与演化视角层存在。

---

## 后续建议

- 若后续新增 Phase 7 或更高阶段总结，可在 `project_evolution_overview.md` 中同步扩展“阶段演进总览”与“下一步自然路线”。
- 若 `boundflow_full_pipeline_from_claims_to_ae.md` 继续演进，建议保持它偏 claim/AE，而把“研发演化史”继续收敛在 `project_evolution_overview.md`。
- 若以后文档继续增长，建议把 `gemini_doc/README.md` 保持为轻索引，把“如何接手项目”统一引流到本次新增文档。
