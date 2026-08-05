---
status: active
updated: 2026-08-05T00:59:47Z
type: changelog
topic: boundflow
slug: prepared-intermediate-refinement-capsule-v1
stage: s01
---

# BoundFlow Prepared Intermediate Refinement Capsule v1 Changelog

## Summary

- NRIR-45 已预注册；当前只有只读诊断与 ceiling probe，无代码、artifact 或正式性能结论。

## Changes

- 冻结 base=`main@b6eb697` 与 NRIR-44 Phase A/B source hashes；
- 唯一变量为 intermediate refinement 的 prepare-once validation ownership；
- 冻结 Phase A per-clause 与 Phase B global correctness/work/timing 门禁；
- 明确首次完整验证、artifact full replay 和 stale/mutation fail-closed 不能删除。

## Validation

- cProfile 发现单 queue 246 次 `_select_targets` 中 186 次来自重复 Program validation；
- 每 exact object 验证一次的 ceiling probe：clause 3 trace 约 `12.85→9.761678 s`，31-node/
  worst lower 保持；正式 three-repeat 尚未开始。

## Decisions

- NRIR-44 已把 floor 降至约 8.6 秒；下一最大可控成本是 top-2 queue 的 child refinement；
- NRIR-43 scorer batching CPU NO-GO 不重开；NRIR-45 不改算法、预算或 policy；
- Phase A ratio 门禁 `<=0.80`，Phase B trace/measured 分别 `<=40/50 s`。

## Follow-Ups

- 先实现 typed capsule 和专用负向测试；
- Phase A 全过后才实现 projected-floor global composition。

## Links

- plan: `gemini_doc/BOUNDFLOW_PREPARED_INTERMEDIATE_REFINEMENT_CAPSULE_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
