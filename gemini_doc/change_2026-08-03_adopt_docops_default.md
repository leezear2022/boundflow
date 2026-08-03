# 变更记录：仓库默认启用 DocOps Logic

> 日期：2026-08-03
> 范围：仓库协作与审计流程

- 初始化 `.docops/s.md`、`c.yaml`、`p.yaml`、`k.jsonl`、`ev.jsonl`；
- `AGENTS.md` 规定每次代码、文档、schema、计划或流程修改后记录 `ch`；
- 测试、构建、审计或显式延期后记录 `va`；
- 交接前执行 `dol lint --soft`；
- RVIR closure 以 PR #4、stage `s01` 回填 change/validation 事件。

本变更只增加低 token 的符号审计层，不替代 `gemini_doc` 变更文档、Git commit、测试或
artifact。
