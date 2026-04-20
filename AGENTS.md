# Repository Guidelines

本文件为 AI Agent（Codex、Claude Code、Gemini 等）提供仓库级工作约定。

## Agent 工作约定

- **全程用中文交流**
- **Conda 环境**: `boundflow`
- **文档存放**: 生成的文档放在 `gemini_doc/` 中
- **每次修改都写变更记录**: `gemini_doc/change_YYYY-MM-DD_*.md` + 追加 `docs/change_log.md`
- **PR 采用回合式协作**: 见 [gemini_doc/llm_collaboration_workflow.md](gemini_doc/llm_collaboration_workflow.md)

## 项目参考

项目完整的技术参考（架构、模块、环境设置、常用命令、开发规范、阶段演进等）请见：

- **[docs/reference.md](docs/reference.md)**: 统一技术参考手册
- **[gemini_doc/README.md](gemini_doc/README.md)**: 文档索引与阅读路径
- **[gemini_doc/llm_briefing_boundflow.md](gemini_doc/llm_briefing_boundflow.md)**: LLM 交接文档

## Agent 专属补充

### PR 规范

- PR 应包含：摘要、关联 issue（如有）、验证命令（`pytest`/`rebuild_tvm.sh`）、性能/覆盖影响
- 截图仅用于用户面向输出的变更

### 安全提醒

- 不提交凭证或本地路径
- `env.sh` 是机器特定配置的正确位置
- 修改 `PYTHONPATH`/`TVM_HOME` 后验证 `tests/test_env.py`

### 关键文档速查

| 用途 | 文档 |
|------|------|
| 当前计划 | `gemini_doc/next_plan_after_phase7a_pr10.md` |
| LLM 协作工作流 | `gemini_doc/llm_collaboration_workflow.md` |
| JSONL schema | `docs/bench_jsonl_schema.md` |
| 研发脉络 | `gemini_doc/project_evolution_overview.md` |
| Phase 6 总结 | `gemini_doc/phase6_summary.md` |
