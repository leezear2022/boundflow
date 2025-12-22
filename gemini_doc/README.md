# gemini_doc 导引（BoundFlow 工程文档索引）

本目录用于存放“由大模型协助生成/维护”的工程文档与变更记录（changelog-style notes），目标是：

- 让每次 PR/阶段推进都有可审计的文字记录；
- 让别人（或未来的你）能快速定位：某个决策/某个口径/某个脚本是“为什么这样做”；
- 让论文/AE 的证据链（claim → 命令 → 产物 → 字段）能闭环。

> 约定：每次工程改动都应在 `gemini_doc/` 新增一份 `change_YYYY-MM-DD_*.md` 记录，并在 `docs/change_log.md` 追加一条总账。

---

## 1) “我应该从哪读起？”

按目的给三条阅读路径：

### A. 论文/AE 视角（最推荐）

1. `docs/phase5_done.md`（Phase 5 完成声明：复现入口、产物结构、DoD、边界）
2. `gemini_doc/artifact_claims_phase5d.md`（claims→文件/字段→命令 的证据映射）
3. `gemini_doc/artifact_appendix_phase5d.md`（AE 操作说明：怎么跑、怎么验证、怎么看结果）
4. `docs/bench_jsonl_schema.md`（JSONL schema=1.0：字段与计时口径）

### B. 全流程总览（从 claims 到工程到 AE）

- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
- `gemini_doc/boundflow_full_pipeline_director_view.md`（指挥视角的系统路线）

### C. 研发协作流程（人与大模型怎么配合）

- `gemini_doc/llm_collaboration_workflow.md`（输入计划→修正测试→总结→下一步计划）

---

## 2) 本目录文件分类

### 2.1 关键交付文档（“长期有效”）

- `gemini_doc/artifact_claims_phase5d.md`：Phase 5D artifact claims（证据链/口径映射）
- `gemini_doc/artifact_appendix_phase5d.md`：Phase 5D artifact appendix（复现说明）
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`：全流程总览（从研究主张到工程到 AE）
- `gemini_doc/boundflow_full_pipeline_director_view.md`：指挥视角的工程主线
- `gemini_doc/tvm_backend_optimization_memo.md`：TVM/Relax 后端优化备忘
- `gemini_doc/llm_collaboration_workflow.md`：与大模型协作工作流模板

### 2.2 变更记录（`change_YYYY-MM-DD_*.md`）

这些文件按时间记录“当时做了什么/为什么做/怎么验证”，适合：

- 回溯某个接口/口径的由来
- 追踪阶段推进（Phase 4 → Phase 5）

常见命名模式：

- `change_YYYY-MM-DD_phase5*_pr*_*.md`：按阶段/PR 编号
- `change_YYYY-MM-DD_*_memo.md`：备忘或总结

---

## 3) Phase 5（现状）的一句话索引

如果你只想知道“Phase 5 到底做完了什么”：

- 完成声明：`docs/phase5_done.md`
- 口径冻结：`docs/bench_jsonl_schema.md`（`schema_version=1.0`）
- 一键产线：`scripts/run_phase5d_artifact.py`（产 `results.jsonl/table_main.csv/figures/MANIFEST/CLAIMS/APPENDIX`）
- 证据链：`gemini_doc/artifact_claims_phase5d.md`

---

## 4) 维护规则（防止目录继续膨胀失控）

1. **不要移动/改名历史 `change_*.md`**（避免破坏已有引用）。
2. 新增文档时优先选择：
   - `docs/`：面向用户/读者的稳定说明（安装、schema、完成声明）
   - `gemini_doc/`：面向研发/演进的过程记录（变更记录、备忘、决策）
3. 任何影响口径的变更都要同时更新：
   - `docs/bench_jsonl_schema.md`
   - 对应的 contract tests / postprocess tests
4. 运行产物目录 `artifacts/`、`out/` 不进入 git（已在 `.gitignore` 忽略）。

