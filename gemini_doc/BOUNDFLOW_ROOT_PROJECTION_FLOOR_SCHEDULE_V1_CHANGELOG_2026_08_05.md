---
status: active
updated: 2026-08-05T00:11:36Z
type: changelog
topic: boundflow
slug: root-projection-floor-schedule-v1
stage: s01
---

# BoundFlow Root-Projection Floor Schedule v1 Changelog

## Summary

- NRIR-44 已预注册，状态为 `preregistered`；尚无代码、artifact 或正式性能结论。

## Changes

- 冻结 integration base=`main@d9d76da` 与 NRIR-42 source formal hash；
- 唯一变量为 floor objective-query 的 consumer-driven root projection：`9×n31d4 → 9×n1d0`；
- 冻结 Phase A/B 顺序、soundness 边界、root/rank/selection exact、work elimination 与 timing 门禁；
- 明确一般 complete verifier 不自动启用该 specialization。

## Validation

- 路线冻结前只读 probe：9 条 root queries 合计 `0.789371 s`，9/9 root results exact；
- 正式 three-repeat validation 尚未开始。

## Decisions

- NRIR-43 已证明扩大 CPU domain batch 无效；
- 当前 floor 约 21.8 秒，其中九条 deep objective queries 约 13.88 秒，而 ranking 只消费 root；
- Phase A 门禁设为 floor `<=11 s` 且 ratio `<=0.50`，Phase B whole `<=48 s` 且 ratio `<=0.82`。

## Follow-Ups

- 先实现 typed consumer/liveness IR 与 additive projected floor；
- Phase A 全过后才接 NRIR-42 production runtime。

## Links

- plan: `gemini_doc/BOUNDFLOW_ROOT_PROJECTION_FLOOR_SCHEDULE_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
