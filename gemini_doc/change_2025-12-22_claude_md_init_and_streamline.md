# Change Log: CLAUDE.md 初始化与精简

**日期**: 2025-12-22
**类型**: 文档
**范围**: 仓库根目录配置

---

## 背景与动机

为了让未来的 Claude Code 实例能够快速上手 BoundFlow 仓库，需要创建一个 CLAUDE.md 文件作为入口指引。该文件应该：

1. 明确约定（中文交流、conda 环境、文档存放位置）
2. 提供最常用的命令和工作流
3. 概览架构，避免过度详细
4. 指向关键文档索引

---

## 主要改动

### 1. 创建 CLAUDE.md

新增 `CLAUDE.md` 文件，内容包括：

- **约定部分**（新增）：
  - 全程用中文交流
  - Conda 环境: `boundflow`
  - 文档存放在 `gemini_doc/` 目录
  - 可新建子文件夹分类
  - 每次修改都要记录 change log

- **项目概述**：BoundFlow 的核心目标和定位

- **环境设置**：安装命令、环境变量说明

- **常用命令**：
  - 测试命令（pytest）
  - 基准测试命令（artifact pipeline、消融实验）

- **架构概览**：
  - 简化的流程图
  - 核心模块一句话说明（ir、planner、backends、frontends、domains、runtime）
  - 第三方依赖说明

- **关键概念**：
  - Task 抽象
  - Storage Planning
  - Benchmark JSONL Schema

- **开发规范**：
  - 代码风格
  - 测试规范
  - 提交规范

- **重要文档**：
  - 入口索引（gemini_doc/README.md、AGENTS.md、docs/phase5_done.md）
  - 核心文档（schema、claims、appendix、workflow、memo）

- **可扩展性**：未来扩展方向

### 2. 精简与中文化

初始版本较为冗长（214 行），经过精简后压缩至约 140 行：

- 删除了每个文件的详细描述，改为模块级别的概要
- 删除了重复的 Building 部分
- 简化了环境变量说明
- 将所有章节标题和内容改为中文
- 保留必要的英文术语（Task、Planner、JSONL 等）

### 3. 注册 gemini_doc/README.md 导引

在"重要文档"部分：

- 新增"入口索引"子部分，将 `gemini_doc/README.md` 作为第一项
- 重新组织文档列表，分为"入口索引"和"核心文档"两类
- 添加了 `docs/phase5_done.md` 作为 Phase 5 完成声明的入口

---

## 验证方式

- 阅读 `CLAUDE.md`，确认：
  - 约定部分清晰明确
  - 常用命令完整可用
  - 架构概览简洁但足够
  - 文档索引准确

- 确认 `gemini_doc/README.md` 已被正确引用

---

## 影响范围

- 新增文件：`CLAUDE.md`
- 新增文件：`gemini_doc/change_2025-12-22_claude_md_init_and_streamline.md`（本文档）
- 未来 Claude Code 实例将以此文件作为入口指引

---

## 后续工作

- 随着项目演进，持续更新 CLAUDE.md 的"常用命令"和"重要文档"部分
- 保持文件简洁，避免过度详细（详细内容应放在被引用的文档中）
- 确保约定部分与实际工作流保持一致
