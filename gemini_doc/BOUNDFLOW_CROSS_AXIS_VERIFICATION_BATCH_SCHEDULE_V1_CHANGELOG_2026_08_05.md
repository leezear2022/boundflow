---
status: completed
updated: 2026-08-05T00:05:57Z
type: changelog
topic: boundflow
slug: cross-axis-verification-batch-schedule-v1
stage: s01
---

# BoundFlow Cross-Axis Verification Batch Schedule v1 Changelog

## Summary

- NRIR-43 Phase A 已完成并以 `VALIDATED-NO-GO` 关闭；Phase B 按门禁未启动。

## Changes

- 冻结基线为 `main@34ca6c6`，直接前序为 NRIR-42 / PR #53；
- 唯一变量固定为跨 clause/node/candidate 的 ready-work batch Schedule；
- 冻结 typed ragged segment ownership、Phase A/B 顺序、exact semantics、launch 与 timing 门禁；
- 明确不得同时修改 policy、optimizer/refinement、queue、budget/deadline、dtype 或 workload。
- 新增 typed ragged Plan/Instance/Task/Schedule/Trace、联合 scorer runtime、additive production queue、
  Phase-A generate/replay 与专用测试；NRIR-42 frozen 文件未改。

## Validation

- 6/6 old/new 组的 queue/branch/score/child-bound/state/refinement exact；每条 scorer launch `31→16`；
- clause 2/3 median ratio=`1.051134/1.044573`，两条 timing gate 均失败；
- formal replay 与 synchronized outer-rehash launch/segment/objective-owner tamper 通过；formal hash=
  `692b9e273661fce9f12129e134550547afa4023361e2a79d751c437c92f30390`；
- targeted `10 passed`，全量 `968 passed, 37 skipped`，Black/mypy/Pylint `10.00/10` 通过。

## Decisions

- 当前 57–58 秒 whole query 中 floor 约 22 秒，两个顺序 production slices 合计约 33 秒；
- 因此下一大杠杆是把两条独立 queue 的同 round work 联合发射，而不是继续削 scorer validation 常数；
- 端到端门禁设为每轮 `<=45 s` 且 median ratio `<=0.80`，未过即 NO-GO。
- Phase A 已证明更大的 CPU domain batch 使墙钟退化约 4–5%；按预注册停止，不运行 Phase B。

## Follow-Ups

- 下一步为 NRIR-44 Root-Projection Floor Schedule，消除 ranking consumer 不需要的九条深层 floor queue；
- 保留本轮负结果，不把 launch reduction 当作性能代理。

## Links

- plan: `gemini_doc/BOUNDFLOW_CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
