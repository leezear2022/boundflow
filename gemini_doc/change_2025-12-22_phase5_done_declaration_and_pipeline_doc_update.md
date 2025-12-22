# 变更记录：Phase 5 完成声明 + 全流程文档修订

## 动机

在当前仓库状态下，Phase 5 的“工程收口/可复现产线/论文消融证据链”已具备：

- `schema_version=1.0` 口径冻结（bench JSONL + postprocess + contract tests）；
- artifact runner 一键产出 `results.jsonl / table_main.csv / figures / MANIFEST / CLAIMS / APPENDIX`；
- baseline（auto_LiRPA）已纳入证据链，且通过 baseline 外提预计算/去重避免浪费算力；
- TVM 侧 compile/run 拆分、compile stats、call_tir/fusion、memory planning 对照可解释。

因此需要一份“Phase 5 完成声明”（面向论文/AE），并同步修订全流程文档，避免与最新实现不一致（例如 runner/workload 支持等）。

## 本次改动

### 1) 新增 Phase 5 完成声明

- 新增：`docs/phase5_done.md`
  - Phase 5 覆盖范围
  - 复现入口（quick/full）
  - 输出目录结构（runner 产物）
  - 最终 DoD（机械收尾项）
  - 已知限制与 Phase 6 边界

### 2) 修订全流程总览文档（从 claims 到 AE）

- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 修正 Non-goals 中关于 workload/runner 的旧描述
  - 更新 Phase 5 部分为“已收口产线”
  - 在收口入口中新增 `docs/phase5_done.md` 链接

## 备注

- `artifacts/` 与 `out/` 为运行产物目录，已加入 `.gitignore`，不进入 git；复现应由 runner/bench 重新生成。

