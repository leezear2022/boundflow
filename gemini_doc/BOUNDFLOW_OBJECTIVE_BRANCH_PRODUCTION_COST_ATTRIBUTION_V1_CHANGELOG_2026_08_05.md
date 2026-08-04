---
status: completed
updated: 2026-08-04T21:30:14Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_OBJECTIVE_BRANCH_PRODUCTION_COST_ATTRIBUTION_V1
stage: s01
---

# Objective Branch Production Cost Attribution v1

## Summary

- NRIR-41 已完成：same-prefix frontier 与 scoring wall-time 两个因果门禁均成立，自动选择下一路线
  `optimize_scorer_ownership`；不撤销 NRIR-40 production NO-GO。

## Changes

- main 同步到 PR #51 merge `9befc51`，建立
  `feat/objective-branch-production-cost-attribution-v1`。
- 预注册 frozen-prefix reconstruction、三 fresh paired wall runs、独立 cProfile diagnostic 与
  `frontier_order_retained`/`scoring_cost_dominant` 两个方向门禁。
- 新增 attribution Plan/6-task TaskModule/Schedule、16 条 prefix IR、12 条 wall IR、8 条 profile phase
  IR 与 causal Decision；NRIR-39/40 frozen 文件零修改。
- 新增 three-process paired runner、counterbalanced order、profile worker、formal/manifest/shards/logs 与
  frozen artifact contract/tamper tests。

## Validation

- clauses 2/3 同节点 prefix improvements 全正；31-node 仍为 `+2.043362/+5.641768`。
- widest/objective queue median=`10.515292/18.387675 s` 与 `10.619606/18.591097 s`，ratio=
  `1.748660/1.750639`；MAD 远小于 policy 差值。
- cProfile branch-program share=`21.9371%/21.9139%`；branch program 31 calls、candidate enumeration
  341 calls，暴露 validation/ownership 重复。
- artifact replay 与 formal/manifest 同步重哈希 prefix tamper 通过预期；focused `4 passed`、
  predecessor-inclusive `12 passed`、全量 `948 passed, 37 skipped`；Black、mypy、Pylint `10.00/10`
  通过。formal hash=
  `fe67b77197905a8a4d7f92ad5eac686892243dfb0e7d7b7c7434861aaa794834`。

## Decisions

- 不修改 NRIR-39/40 frozen 文件；profiled 时间不进入 unprofiled median；所有 timing 仅为内部归因，
  `performance_claimed=false`。
- 两项预注册门禁都成立，下一阶段只能以 candidate enumeration/validation 的 scorer ownership 与复用
  为单变量；不得同时改 candidate policy 或 search/deadline 常数。

## Follow-Ups

- 建立 NRIR-42 scorer ownership/reuse 计划；先冻结 exact selected candidate/score/child-lower parity，
  再要求 341 enumeration calls 显著下降并回到 whole-query production 门禁。

## Links

- plan: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_PRODUCTION_COST_ATTRIBUTION_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
- implementation commit: `991c75e`
- publication: GitHub PR #52（base `main`，head
  `feat/objective-branch-production-cost-attribution-v1`）
