---
status: active
updated: 2026-08-04T23:28:03Z
type: changelog
topic: boundflow
slug: cross-axis-verification-batch-schedule-v1
stage: s01
---

# BoundFlow Cross-Axis Verification Batch Schedule v1 Changelog

## Summary

- NRIR-43 已预注册，状态为 `preregistered`；尚无代码、artifact 或性能结论。

## Changes

- 冻结基线为 `main@34ca6c6`，直接前序为 NRIR-42 / PR #53；
- 唯一变量固定为跨 clause/node/candidate 的 ready-work batch Schedule；
- 冻结 typed ragged segment ownership、Phase A/B 顺序、exact semantics、launch 与 timing 门禁；
- 明确不得同时修改 policy、optimizer/refinement、queue、budget/deadline、dtype 或 workload。

## Validation

- 预注册文档完成后执行 `git diff --check`、DocOps validate 与 lint；
- 代码和实验 validation 尚未开始。

## Decisions

- 当前 57–58 秒 whole query 中 floor 约 22 秒，两个顺序 production slices 合计约 33 秒；
- 因此下一大杠杆是把两条独立 queue 的同 round work 联合发射，而不是继续削 scorer validation 常数；
- 端到端门禁设为每轮 `<=45 s` 且 median ratio `<=0.80`，未过即 NO-GO。

## Follow-Ups

- 先实现 typed scorer ragged pack 与 sibling-pair exact parity；
- Phase A 全过后才实现/运行 two-clause ready-set coordinator formal。

## Links

- plan: `gemini_doc/BOUNDFLOW_CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
