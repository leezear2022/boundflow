# 变更记录：刷新 BoundFlow 工作流 skill

**日期**: 2026-03-25
**类型**: skill / 工作流整理
**范围**: `~/.codex/skills/`、`gemini_doc/`、`docs/`

---

## 背景与动机

仓库里已经有 [gemini_doc/llm_collaboration_workflow.md](/home/lee/Codes/boundflow/gemini_doc/llm_collaboration_workflow.md) 这份工作流模板，也已经有一个早期版本的本机 skill：

- `/home/lee/.codex/skills/boundflow-workflow/SKILL.md`

但旧版 skill 更像“简版入口”，还没有把文档里的几项关键动作收紧进去，例如：

- 用户输入模板
- 先写 DoD 再实现
- 失败时先分流 `contract/pipeline` 与 `shape/dtype/数值语义`
- 回合结束时要交代“已验证/残余风险/下一步 PR”

因此本次不是新建第二个 skill，而是把现有 `boundflow-workflow` 直接升级为更贴近仓库真实协作方式的版本。

---

## 主要改动

- 更新 skill：`/home/lee/.codex/skills/boundflow-workflow/SKILL.md`
  - 把描述字段收紧成“何时触发”，避免把执行流程塞进 frontmatter。
  - 明确“先读什么”：
    - `AGENTS.md`
    - `gemini_doc/llm_collaboration_workflow.md`
    - 阶段文档 / `project_evolution_overview.md`
    - `docs/change_log.md` 与最近 `gemini_doc/change_*.md`
    - 相关实现与测试
  - 固化仓库硬约束：
    - 中文交流
    - conda 环境 `boundflow`
    - 每次改动都要写 `gemini_doc/change_*.md`
    - 每次改动都要追加 `docs/change_log.md`
    - 改动要收敛成小步、可验证闭环
  - 固化默认执行顺序：
    - 先确认 phase/PR 目标
    - 先写 DoD 再实现
    - 先做最小可验证闭环
    - 先定向验证再扩回归
    - 失败时先归类问题类型
    - 再写变更记录和总账
    - 收尾必须说明“做了什么/验证了什么/风险/下一步”
  - 补上用户输入模板，方便把 `llm_collaboration_workflow.md` 里的“目标/DoD/约束/输出”压缩成 skill 中的固定动作。

- 更新：`gemini_doc/README.md`
  - 在“研发协作流程”入口里补充本机 skill 路径，形成“仓库文档 + 本机 skill”双入口。

- 新增：`gemini_doc/change_2026-03-25_refresh_boundflow_workflow_skill.md`

---

## 为什么要保留“文档 + skill”双层结构

只保留 skill 的问题是：

- skill 适合短入口，不适合承载完整背景；
- 一旦流程需要扩展，完整 rationale 仍要落在仓库文档里。

只保留文档的问题是：

- 工具层不容易直接触发；
- 每次都要重新解释同样的 PR-by-PR 习惯动作。

因此更合适的结构是：

- `gemini_doc/llm_collaboration_workflow.md` 继续做长版模板；
- `boundflow-workflow` skill 做短版可执行入口；
- `gemini_doc/README.md` 负责在仓库内给出发现路径。

---

## 本次涉及的文件

- 更新：`/home/lee/.codex/skills/boundflow-workflow/SKILL.md`
- 更新：`gemini_doc/README.md`
- 新增：`gemini_doc/change_2026-03-25_refresh_boundflow_workflow_skill.md`
- 更新：`docs/change_log.md`

---

## 验证方式

- 检查 `~/.codex/skills/boundflow-workflow/SKILL.md` 仍存在且内容已更新；
- 检查 skill 已覆盖：
  - 触发条件
  - 仓库硬约束
  - 默认执行顺序
  - 用户输入模板
  - 完成标准
- 检查 `gemini_doc/README.md` 已补上 skill 路径；
- 检查仓库侧已写本次变更记录，并追加总账。

---

## 后续建议

- 如果后面希望这个 skill 再进一步“可操作化”，可以增加一个 `references/` 子目录，收纳：
  - 常用 pytest/bench 命令模板
  - phase/PR 入口索引
  - change 文档命名建议
- 如果后面发现这套流程在 Phase 7A 之后又有显著变化，可以继续按“刷新 skill + 仓库文档长版模板”的方式同步维护。
