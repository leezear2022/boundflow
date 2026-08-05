---
status: validated-no-go
updated: 2026-08-05T11:10:00Z
type: changelog
topic: boundflow
slug: intermediate-refinement-template-instance-v1
stage: s01
---

# BoundFlow Intermediate Refinement Template/Instance v1 Changelog

## Summary

- NRIR46 已完成三 fresh process Phase 0 compiler ownership attribution；strict static-shareable
  median=`1.071197 s < 1.5 s`，按预注册门禁以 `VALIDATED-NO-GO` 关闭，Phase A/B 未启动。

## Changes

- 冻结 integration base=`main@6cd229a`、NRIR45 formal/payload hashes 与用户 review 豁免边界；
- 将剩余 trace 拆为 floor、packed slice、plan compile、rank 与 trace 外证据校验；
- 冻结 `PlanTemplate/ScheduleTemplate + PlanInstance/InstanceSchedule` first-class compiler IR；
- 明确 dynamic target ledger 不能跨 child 共享，NRIR46 不改变数值 batching、policy 或 budget；
- 冻结 Phase 0 ceiling、Phase A per-clause 和 Phase B whole-query 三层门禁。
- 新增独立 Phase 0 generate/replay/worker runner、两条归因测试与 digest-locked artifact；未修改
  frozen NRIR45 production 实现。

## Validation

- Phase-B raw shards：floor action median=`10.818262 s`，packed slice median=`9.932808 s`，packed-plan
  compile median=`0.146457 s`，rank median=`0.024966 s`；
- diagnostic repeat0：trace=`30.826307 s`，60 child prepared compile=`5.300590 s`、execute=
  `5.659414 s`、per-child total=`10.975123 s`、optimizer execute=`1.156098 s`；
- Phase 0 compile total=`5.356892/5.366369/5.452290 s`；strict static topology=
  `1.071197/1.062492/1.071704 s`，median=`1.071197 s`；ownership-convertible ceiling=
  `2.097255/2.102134/2.109857 s`，median=`2.102134 s`；
- observed/semantic target selection 每轮=`124/60`，冗余=`64`，估计耗时=
  `1.026058/1.039642/1.038153 s`；60 个 target ledger 全部动态互异；
- formal hash=`712ce359501a010a197797909ab71fb127ebda43329dd3a7a8e21b6dbb4cf846`；
  replay 与同步外层重哈希篡改拒绝通过，`performance_claimed=false`；
- targeted `2 passed`、全量 `986 passed, 37 skipped`、Black/mypy/Pylint `10.00/10` 通过。

## Decisions

- 下一单变量是 static/dynamic compiler IR ownership，不重开 NRIR43 CPU scorer batching；
- Template 只共享静态图/策略/拓扑，Instance 继续拥有 exact split/source/objective/targets；
- PR #56 已由 executor deterministic gates 自检后合入；static-shareable ceiling 不足即 NO-GO；
- strict static-shareable ceiling 实测不足，故不实现 Template/Instance，不事后放宽门槛；
- 即使 NRIR46 消除全部已测 compile，约 31.3 秒也只降至理论约 26 秒，不能声称 10x 或 ASPLOS-ready。

## Follow-Ups

- NRIR46 已关闭；下一独立变量是只消除 compile/validate 中 64 次冗余 target reselection 的
  single-pass target admission receipt，须另行预注册并保留 full replay 重算；
- `nrir45-20260805` exchange 保留 ready-for-audit 历史状态，不伪造外部批准。

## Links

- plan: `gemini_doc/BOUNDFLOW_INTERMEDIATE_REFINEMENT_TEMPLATE_INSTANCE_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
