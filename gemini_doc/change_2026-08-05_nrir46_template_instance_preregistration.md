# 2026-08-05 NRIR46 Template/Instance 预注册

## 起因

NRIR45 已把 whole trace 从 NRIR44 的约 44.1 秒降到约 31.3 秒，但 final 仍 9/9 unknown。对三个
Phase-B shard 的独立拆分显示，剩余成本不是 packed-plan compile 或 ranking，而是约 10.8 秒 floor
和两条各约 9.9 秒 packed queue。

低开销 diagnostic repeat0 进一步定位：60 个 child 的 prepared compile/execute 分别约
5.30/5.66 秒，per-child refinement 合计约 10.98 秒；optimizer execute 约 1.16 秒。因此下一步先处理
per-child compiler ownership，不泛化为“继续优化 CROWN”。

## 预注册决策

- 新分支：`feat/intermediate-refinement-template-instance-v1`；
- stacked base：`a2d8f96`，依赖 PR #56 外部审计批准与合并；
- 唯一变量：把静态 `PlanTemplate/TaskTemplate/ScheduleTemplate` 与动态
  `PlanInstance/InstanceSchedule` 分开；
- dynamic target ledger、split、source lineage、objective 与 bounds 继续逐 child exact-owned；
- 不改 policy/pass/cap、selected-CROWN math、optimizer、branch、31/depth4、floor、deadline 或 workload；
- Phase 0 先验证 static-shareable ceiling，Phase A 两条 queue 过 exact/ownership/timing 后才允许
  Phase B whole query。

## Claim 边界

当前只是预注册和诊断，没有实现、artifact 或新性能 claim。即使完全消除已测 5.30 秒 compile，约
31.3 秒 trace 的理论下界仍约 26 秒，因此本路线不是 10x、公平竞品或 ASPLOS-ready 终点。

## 文档

- 计划：`gemini_doc/BOUNDFLOW_INTERMEDIATE_REFINEMENT_TEMPLATE_INSTANCE_V1_PLAN_2026_08_05.md`
- changelog：
  `gemini_doc/BOUNDFLOW_INTERMEDIATE_REFINEMENT_TEMPLATE_INSTANCE_V1_CHANGELOG_2026_08_05.md`

