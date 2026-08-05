---
status: preregistered
updated: 2026-08-05T12:39:58Z
type: changelog
topic: boundflow
slug: top2-production-execution-cost-attribution-v1
stage: s01
---

# BoundFlow Top-2 Production Execution Cost Attribution v1 Changelog

## Summary

- NRIR-48 top-2 production execution cost attribution 已预注册；当前只有测量协议、分类与路线门禁，
  没有 runner、artifact、dominant category 或新性能 claim。

## Changes

- 基线冻结为 `main@1e44949` 的 NRIR-45 default production route；NRIR-47 candidate 保持禁用；
- 冻结七个互斥顶层类别和 child-refinement-execute 内部五个诊断子类；
- 冻结 three-fresh-process paired control/profile、exact semantics、`<=1%` closure 与 `<=1.05`
  instrumentation perturbation 门禁；
- 冻结 dominant category 的跨 clause/repeat share、稳定性与 pooled-MAD 门禁。

## Validation

- preregistration only；已知 `5.300590/5.659414/1.156098 s` 来自单次旧 diagnostic，不作为本轮
  正式结论；
- 当前不声称 selected-CROWN、optimizer、queue 或 Python dispatch 是 dominant。

## Decisions

- 本轮只做 attribution，不实现优化；
- 顶层类别必须互斥，内部 inclusive 时间不得重复计入 total；
- 若 residual dominant，先增加计时点；若 child execute dominant 但内部无 `>=30%` 子类，继续细分；
- 只有两条 clause 三轮一致且稳定的 dominant category 才能成为 NRIR-49 单变量来源。

## Follow-Ups

- 实现 additive runner、replay/tamper 与 focused test；
- 运行正式 Phase 0 后按预注册规则选择或拒绝 NRIR-49。

## Links

- plan: `gemini_doc/BOUNDFLOW_TOP2_PRODUCTION_EXECUTION_COST_ATTRIBUTION_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
