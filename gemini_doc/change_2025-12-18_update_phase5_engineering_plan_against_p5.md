# 变更记录：根据 `docs/p5.md` 更新 Phase 5 工程计划文档

## 背景

你在 `docs/p5.md` 里对 Phase 5 计划做了明显强化（更偏科研/系统化）：强调 pass pipeline、PlanBundle、以及 cost model/ablation harness。

## 本次调整

- 更新 `gemini_doc/phase5_engineering_plan.md`
  - 增加 “Phase 5 pass 插槽” 小节（property/domain/partition/cache/layout/objective）
  - 将 Phase 5 分阶段扩展为 `5A–5F`，新增 `5F`（objective/search/ablation）
  - 明确 TVM 后端 Phase 5 的 baseline：**Relax VM + Relax op（工程侧不手写 TE/TIR）**；并把 `call_tir` mixed-module 作为进阶可选路线
  - PR 拆分增加 `PR#8` 对应 `5F`

