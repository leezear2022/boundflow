# 变更记录：新增 Phase 0 总结文档（phase0_summary.md）

## 动机

仓库总账中最早一条记录以 `Phase 0/1` 合并描述，且此前已补齐 Phase 1/2/3/4/5/6 的总结文档。为保持“每个 phase 都有独立总结”的文档体系一致性，本次补齐 Phase 0 总结：

- 聚焦工程止血（editable install / 包结构清理 / 最小 smoke）；
- IR 加固部分仍由 `gemini_doc/phase1_summary.md` 承担，避免重复叙事。

## 本次改动

- 新增：`gemini_doc/phase0_summary.md`
  - 总结 Phase 0 的目标/完成定义、关键工程改动落点与验证命令。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase0_summary.md` 加入“关键交付文档”索引。

## 备注

本次为文档增量，不涉及 runtime 语义变更。

