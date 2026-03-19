# 变更记录：更新 `boundflow_full_pipeline_from_claims_to_ae.md`（v2.0，对齐 Phase 0~6）

## 动机

`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 之前停留在 “Phase 5 完成（schema 1.0）” 视角，并把 Phase 6（αβ-CROWN + BaB + E2E 工件链）仍写成 TODO。随着 Phase 6（含 6H AE 交付形态）已收官，需要把“从 claims 到 AE”的总导览更新为覆盖 Phase 0~6 的现状，避免新读者按旧文档走错入口/口径。

## 本次改动

- 重写：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 版本升级为 v2.0，明确两条可复现主线：
    - Phase 5：interval IBP + TVM（bench JSONL `schema_version=1.0`，入口 `docs/phase5_done.md`）
    - Phase 6：αβ oracle + BaB（E2E JSON `schema_version=phase6h_e2e_v2`，入口 `gemini_doc/ae_readme_phase6h.md`）
  - 用两张 Mermaid 流水线图对齐两套工件链路（Phase5 artifact runner vs Phase6H E2E runner）。
  - 更新“从哪开始读”导航：直接链接到 `gemini_doc/phase0_summary.md` ~ `gemini_doc/phase6_summary.md`。
  - 固化 Phase 6 边界：明确当前 CROWN/αβ/BaB 仅覆盖链式 MLP（Linear+ReLU），未实现 conv。

## 验证

- 文档变更（无额外运行时验证）。
