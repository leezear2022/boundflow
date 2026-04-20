# 变更记录：文档整合与 LLM 交接文档

**日期**: 2026-03-28
**类型**: docs
**范围**: CLAUDE.md, AGENTS.md, GEMINI.md, README.md, docs/reference.md, gemini_doc/

---

## 动机

CLAUDE.md、AGENTS.md、README.md 三个根文件内容严重重叠（环境设置、代码规范、模块描述、测试命令等均有重复），且 README.md 停留在早期状态未反映 Phase 6/7A 的进展。需要：

1. 消除重复，建立单一权威参考
2. 更新 README.md 反映项目真实现状
3. 生成 LLM 交接文档方便与其他大模型协作

## 主要改动

### 新建文件

- **`docs/reference.md`**: 统一技术参考手册，合并三文件所有共性内容（项目概述、架构、模块组织、环境设置、常用命令、关键概念、开发规范、文档体系、可扩展性、阶段演进表）
- **`gemini_doc/llm_briefing_boundflow.md`**: LLM 交接文档，"问题-背景-解决方案-现状"格式，替代已过时的 `docs/llm_handoff_summary.md`（后者停留在 Phase 4B）

### 重写文件

- **`CLAUDE.md`**: 瘦身至 ~40 行，仅保留 Claude Code 专属约定 + 指向 reference.md
- **`AGENTS.md`**: 瘦身至 ~45 行，仅保留 Agent 专属约定 + 指向 reference.md
- **`GEMINI.md`**: 从空文件更新为包含约定和指针
- **`README.md`**: 完全重写，反映 Phase 7A 现状（方法族、模型支持、系统特性、阶段进度表、快速开始、最小示例）

### 编辑文件

- **`gemini_doc/README.md`**: 新增"E. LLM 交接视角"阅读路径；新增 reference.md 和 llm_briefing 到关键文档列表；更新现状从 Phase 5 到 Phase 7A
- **`docs/llm_handoff_summary.md`**: 顶部添加废弃提示，指向新文档

## 设计决策

- **reference.md 放在 `docs/`**: 按约定，`docs/` 存放稳定面向用户的文档；统一参考是长期稳定文档
- **llm_briefing 放在 `gemini_doc/`**: 按约定，`gemini_doc/` 存放生成的过程文档；交接文档会随项目演进更新
- **保留四个根文件**: 每个服务不同消费者（Claude Code / Codex / Gemini / 人类），不能合并
- **不删除旧 handoff**: 按维护规则不移动/删除历史文档，只加废弃标记

## 影响面

- 不影响任何代码或测试
- 所有内部文档引用链保持完整
- 新文档覆盖了原三文件的全部实质内容（逐条验证）
