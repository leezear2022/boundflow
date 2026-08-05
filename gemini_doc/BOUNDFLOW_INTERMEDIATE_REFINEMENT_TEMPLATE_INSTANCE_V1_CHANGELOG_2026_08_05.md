---
status: preregistered
updated: 2026-08-05T02:44:32Z
type: changelog
topic: boundflow
slug: intermediate-refinement-template-instance-v1
stage: s01
---

# BoundFlow Intermediate Refinement Template/Instance v1 Changelog

## Summary

- NRIR46 已完成开工前 residual attribution 与预注册；当前仅有文档/诊断，没有实现或正式性能结果。

## Changes

- 冻结 integration base=`main@6cd229a`、NRIR45 formal/payload hashes 与用户 review 豁免边界；
- 将剩余 trace 拆为 floor、packed slice、plan compile、rank 与 trace 外证据校验；
- 冻结 `PlanTemplate/ScheduleTemplate + PlanInstance/InstanceSchedule` first-class compiler IR；
- 明确 dynamic target ledger 不能跨 child 共享，NRIR46 不改变数值 batching、policy 或 budget；
- 冻结 Phase 0 ceiling、Phase A per-clause 和 Phase B whole-query 三层门禁。

## Validation

- Phase-B raw shards：floor action median=`10.818262 s`，packed slice median=`9.932808 s`，packed-plan
  compile median=`0.146457 s`，rank median=`0.024966 s`；
- diagnostic repeat0：trace=`30.826307 s`，60 child prepared compile=`5.300590 s`、execute=
  `5.659414 s`、per-child total=`10.975123 s`、optimizer execute=`1.156098 s`；
- 以上是路线诊断，不是 formal claim；必须由实现后的 three-repeat artifact 独立验证。

## Decisions

- 下一单变量是 static/dynamic compiler IR ownership，不重开 NRIR43 CPU scorer batching；
- Template 只共享静态图/策略/拓扑，Instance 继续拥有 exact split/source/objective/targets；
- PR #56 已由 executor deterministic gates 自检后合入；static-shareable ceiling 不足即 NO-GO；
- 即使 NRIR46 消除全部已测 compile，约 31.3 秒也只降至理论约 26 秒，不能声称 10x 或 ASPLOS-ready。

## Follow-Ups

- 已重定基到 `main@6cd229a`；先实现 Phase 0 细粒度 attribution，再按门禁决定是否编码 IR；
- `nrir45-20260805` exchange 保留 ready-for-audit 历史状态，不伪造外部批准。

## Links

- plan: `gemini_doc/BOUNDFLOW_INTERMEDIATE_REFINEMENT_TEMPLATE_INSTANCE_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
