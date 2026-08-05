---
status: validated-reduced
updated: 2026-08-05T01:04:00Z
type: changelog
topic: boundflow
slug: root-projection-floor-schedule-v1
stage: s01
---

# BoundFlow Root-Projection Floor Schedule v1 Changelog

## Summary

- NRIR-44 Phase A/B 已按预注册门禁完成，以固定 ResNet2B property 0 CPU8
  ranking-floor + production admission `VALIDATED-REDUCED` 关闭。

## Changes

- 冻结 integration base=`main@d9d76da` 与 NRIR-42 source formal hash；
- 唯一变量为 floor objective-query 的 consumer-driven root projection：`9×n31d4 → 9×n1d0`；
- 冻结 Phase A/B 顺序、soundness 边界、root/rank/selection exact、work elimination 与 timing 门禁；
- 明确一般 complete verifier 不自动启用该 specialization。
- 新增 consumer-owned Plan/Instance/7-task Task/Schedule/Trace、additive projected floor 与 NRIR-42
  global composition；frozen NRIR-31/42/43 文件不改；
- Phase A objective evaluations `279→9`；Phase B 继续执行原 top-2 两条 31-node production queues。

## Validation

- Phase A old/projected median=`24.235039/9.876515 s`，ratio=`0.407530`，三轮最大 projected
  `10.740998 s`；root/rank/selected 语义 exact；formal hash=`ecb553d8…ff0fe`；
- Phase B floor=`8.538814/8.622447/8.648849 s`，whole=
  `43.571040/44.144990/44.095736 s`，相对 NRIR-42 whole median ratio=`0.764254`；
- Phase B 每轮 `[31,31]` nodes、worst lower exact；formal payload hash=`2f22d44f…7272d9`；
- replay、typed reconstruction、同步重哈希 budget/consumer/deadline tamper、targeted `11 passed`、
  全量 `979 passed, 37 skipped` 与静态门禁通过。

## Decisions

- NRIR-43 已证明扩大 CPU domain batch 无效；
- 当前 floor 约 21.8 秒，其中九条 deep objective queries 约 13.88 秒，而 ranking 只消费 root；
- Phase A 门禁设为 floor `<=11 s` 且 ratio `<=0.50`，Phase B whole `<=48 s` 且 ratio `<=0.82`。
- 两阶段门禁均通过；结论仍是 internal fixed-workload admission，不能写作公平竞品 speedup。

## Follow-Ups

- 完成全量回归和发布；
- 下一步先对剩余约 35 秒 top-2 production queue 做新一轮单变量归因，不回退已否决的 CPU
  cross-axis scorer batching，也不把当前结果外推为 ASPLOS-ready。

## Links

- plan: `gemini_doc/BOUNDFLOW_ROOT_PROJECTION_FLOOR_SCHEDULE_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
