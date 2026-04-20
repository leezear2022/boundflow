# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供项目指引。

## Claude Code 专属约定

- **全程用中文交流**（除非用户明确要求英文）
- **Conda 环境**: `boundflow`
- **文档存放**: 所有生成的文档、变更记录、备忘录存放在 `gemini_doc/` 目录中
- **文档组织**: 可在 `gemini_doc/` 下新建子文件夹分类管理
- **每次修改都要记录**: 写 change log 到 `gemini_doc/change_YYYY-MM-DD_*.md`，并追加 `docs/change_log.md`

## 项目参考

项目完整的技术参考（架构、模块、环境、命令、开发规范、阶段演进等）请见：

- **[docs/reference.md](docs/reference.md)**: 统一技术参考手册
- **[gemini_doc/README.md](gemini_doc/README.md)**: gemini_doc 目录导引与阅读路径
- **[gemini_doc/llm_briefing_boundflow.md](gemini_doc/llm_briefing_boundflow.md)**: LLM 交接文档（问题-背景-解决方案-现状）

## 当前状态速查

- **当前开发前沿**: Phase 7A（已完成 structured ReLU backward + layout-only shared CROWN）
- **下一步计划**: [gemini_doc/next_plan_after_phase7a_pr10.md](gemini_doc/next_plan_after_phase7a_pr10.md)（benchmark + 继续消除 sign-split dense 点）
- **研发脉络总览**: [gemini_doc/project_evolution_overview.md](gemini_doc/project_evolution_overview.md)
- **Phase 6 总结**: [gemini_doc/phase6_summary.md](gemini_doc/phase6_summary.md)

## 快速命令

```bash
conda activate boundflow && source env.sh   # 激活环境
pytest tests/test_env.py                     # 环境验证
pytest tests                                 # 全部测试
bash scripts/rebuild_tvm.sh                  # TVM 增量重编译（需重启 Python）
```
