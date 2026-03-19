# 变更记录：新增 BoundFlow 工作流 skill

**日期**: 2026-03-15
**类型**: skill / 工作流固化
**范围**: `~/.codex/skills/`、`gemini_doc/`、`docs/`

---

## 背景与动机

仓库内已经有 [gemini_doc/llm_collaboration_workflow.md](/home/lee/Codes/boundflow/gemini_doc/llm_collaboration_workflow.md) 这份“与大模型协作工作流”文档，但它目前只是文档，还不是一个可被 Codex 直接触发和复用的 skill。

为了让这套长期使用的工作流真正变成可调用能力，本次把它固化成一个本机 skill：

- 名称：`boundflow-workflow`
- 位置：`/home/lee/.codex/skills/boundflow-workflow/SKILL.md`

这样以后在 BoundFlow 仓库内提到这个 skill，或者任务明显符合这套 PR-by-PR 协作模式时，就可以直接按固定流程执行，而不是每次重复解释。

---

## 主要改动

- 新增 skill：`/home/lee/.codex/skills/boundflow-workflow/SKILL.md`
  - 定义触发范围：BoundFlow 仓库内的 phase/PR 迭代、runtime/planner/backend 变更、文档驱动开发任务。
  - 固化执行顺序：先看 `AGENTS.md` 和 `gemini_doc/llm_collaboration_workflow.md`，再做 DoD、实现最小闭环、跑定向测试、写 `gemini_doc/change_*.md`、追加 `docs/change_log.md`。
  - 固化仓库约束：中文交流、conda 环境 `boundflow`、每次改动必须有变更记录和总账。
- 新增：`gemini_doc/change_2026-03-15_add_boundflow_workflow_skill.md`
- 更新：`docs/change_log.md`

---

## 为什么不只保留文档

只保留文档的问题是：

- 工作流存在，但不会被工具层自动触发；
- 每次都要重新解释“先计划、后 DoD、再实现、再写记录”的流程；
- 很难把这套仓库内经验沉淀成一个可复用的操作单元。

把它做成 skill 之后，文档继续作为详细参考，skill 则负责把最核心的流程、约束和触发条件压缩成一个可执行入口。

---

## 本次更新的文件列表

- 新增：`/home/lee/.codex/skills/boundflow-workflow/SKILL.md`
- 新增：`gemini_doc/change_2026-03-15_add_boundflow_workflow_skill.md`
- 更新：`docs/change_log.md`

---

## 验证方式

- 检查 `~/.codex/skills/boundflow-workflow/SKILL.md` 已创建；
- 检查 skill 描述明确包含 BoundFlow 仓库、PR-by-PR 工作流、DoD、pytest、变更记录和 `docs/change_log.md`；
- 检查 skill 未重复复制整份 workflow 文档，而是把详细内容继续指向仓库内已有文档；
- 检查仓库侧已写本次变更记录，并追加总账。

---

## 后续建议

- 如果你后面希望这个 skill 更“自动化”，可以再补一个 `references/` 或 `scripts/` 子目录，放入常用命令模板或 phase 选择参考。
- 如果你希望它在 Codex UI 里显示得更完整，可以后续再补 `agents/openai.yaml`，但当前 `SKILL.md` 已足够可用。
