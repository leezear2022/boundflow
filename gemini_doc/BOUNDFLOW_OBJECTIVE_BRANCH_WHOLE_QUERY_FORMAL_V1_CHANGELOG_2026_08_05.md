---
status: completed
updated: 2026-08-04T21:16:26Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_OBJECTIVE_BRANCH_WHOLE_QUERY_FORMAL_V1
stage: s01
---

# Objective Branch Whole Query Formal v1 Changelog

## Summary

- NRIR-40 已完成：objective branch 在真实 single-global-deadline whole query 中保持 typed correctness，
  但覆盖量和 frontier tightness 均未过预注册 production gate，以 `VALIDATED-NO-GO` 关闭。

## Changes

- main 同步到 `331086d`，建立 `feat/objective-branch-whole-query-formal-v1`。
- 预注册 three fresh processes、global 60 秒、31/depth4/`+1.0`/whole≤70s production-admission 门禁。
- 新增 raw objective-branch shared production queue 与 multi-clause composition；branch scoring 在真实
  slice/global monotonic deadline 内执行，不再额外重放 widest control。
- 新增 three-process generate/replay runner、worker/formal/manifest schema、逐轮 shard/log 与冻结 artifact；
  replay 交叉绑定 shard 和 formal 内嵌结果。
- 新增冻结 artifact 合约与同步重哈希 branch-coverage tamper 测试。

## Validation

- 三轮 floor elapsed=`[21.636507,22.057062,22.088135] s`，rank/selected 均为
  `[2,3,4,5,0,8,6,7,1]`/`[2,3]`，correctness gate 全过。
- accepted nodes=`[[29,23],[29,21],[29,21]]`，groups=`[[14,11],[14,10],[14,10]]`；branch execution
  counts 与 accepted nodes 一致，每轮 cache=`1 miss`，无 partial sibling commit。
- clauses 2/3 worst-active lower 为 `[-48.315041,-43.299690]`、
  `[-48.315041,-44.731468]`、`[-48.315041,-44.731468]`，相对 frozen widest 不升反降；final 仍
  9/9 unresolved。
- whole cooperative elapsed=`[63.357098,63.161128,62.485366] s`；原样 replay 通过，同步修改
  formal+shard 并重算 worker/formal/manifest hashes 的重复 branch-node tamper 仍 fail closed。
- focused `8 passed`、predecessor-inclusive `55 passed`、全量 `944 passed, 37 skipped`；Black、mypy、
  Pylint `10.00/10` 通过。formal hash=
  `d69b56d4d82ad5bf8d30883258c15a39e5a45f1fac9dbc8eb35e91fda9f6a492`。

## Decisions

- scoring 必须计入 query 时间；不允许用 NRIR-39 logical fixed-budget clock 或计时外 widest anchor 美化结果。
- correctness 成立但 production 的 coverage/tightness gate 三轮均失败，按预注册冻结为
  objective-branch global-budget `VALIDATED-NO-GO`；不调 threshold/top-k/slice/node/cap 追结果。

## Follow-Ups

- 保留 NRIR-39 fixed-budget tightness 作为机制证据，但冻结当前 objective-branch policy 的 production
  admission。若继续，只先做 scoring/queue wall-time 与 frontier-order 因果归因，再预注册新的单变量；
  不直接开展 policy sweep。

## Links

- plan: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_WHOLE_QUERY_FORMAL_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
