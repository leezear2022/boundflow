---
status: completed-no-go
updated: 2026-08-04T15:14:00Z
type: changelog
topic: boundflow
slug: objective-ancestral-child-budget-pareto-v1
stage: s01
---

# Objective-Ancestral Child Budget Pareto v1 Changelog

## Summary

- NRIR-33 已以 `VALIDATED-NO-GO` 关闭。five-cap 全部只提交 7 nodes，预注册 90% retention 规则
  只能选择 cap128；降低 child target cap 无法转化为搜索 coverage。

## Changes

- 用 DocOps 建立 standalone plan/changelog，冻结 candidate set、pilot order、selection rule、formal
  repeats 和 full-query upgrade boundary。
- 新增 additive child-budget Policy/Calibration/Decision/Plan IR。candidate cap、pilot order、root
  tolerance、retention rule、calibration rows/evidence 与 winner 均进入 stable hash；同步 winner tamper
  被拒绝。
- 新增 thin runtime wrapper，以结构化 Plan 协议复用 frozen NRIR-32 queue engine；selected cap 精确
  lower 为 child refinement policy，NRIR-32 source/artifact 零修改。
- 新增 five-cap fresh-process pilot generator/replay、完整 shard/log/manifest 与 5 个 tests。

## Validation

- cap `8/16/32/64/128` 的 accepted nodes 均为 `7`，max depth 均为 2；worst active lower 分别为
  `-173.078613/-162.253326/-148.134460/-126.962929/-104.765411`。
- root-global worst active lower=`-200.465393`；cap128 gain=`+95.699982`。90% retention winner=
  cap128，selected retention=`1.0`；较小 cap 没有 coverage 收益。
- artifact replay PASS；focused `5 passed`；pilot hash=
  `db9b406eebebad0c1c4d6f39e8088667935f10e3d54f38cb848dce792dd757eb`。
- 全量 `851 passed, 37 skipped`；Black、mypy、Pylint `10.00/10` 与 diff gate 通过。

## Decisions

- 不直接修改 NRIR-32 固定 cap128 Plan/runtime，否则其 source-bound replay 会失效；NRIR-33 使用
  additive Plan IR 与 thin wrapper 复用已验证 queue engine。
- pilot 可决定 winner，但不能成为正式 repeated claim。
- cap128 的 7-node 结果已有 NRIR-32 三 fresh repeats 支撑；本轮 winner 等于 reference，且所有
  candidate 单次均未超过 7，所以预注册 formal activation gate 已失败，不重复消耗三轮实验来制造
  同一负结论。
- timing 只定位瓶颈，不声明 performance；没有 property、GPU、competitor、multi-workload 或
  ASPLOS-ready claim。

## Follow-Ups

- 下一路线固定为 objective-ancestral sibling packed refinement/evaluation + parametric evaluator，
  保持 child cap128、root source、31/depth4 和 60 秒 deadline；目标是消除 serial per-child evaluator
  开销并严格增加 committed nodes。

## Links

- plan: `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PARETO_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
