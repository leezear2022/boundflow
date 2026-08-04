---
status: completed
updated: 2026-08-05T07:18:00+08:00
type: changelog
topic: boundflow
slug: BOUNDFLOW_OBJECTIVE_BRANCH_SCORER_OWNERSHIP_V1
stage: s01
---

# Objective Branch Scorer Ownership v1

## Summary

- NRIR-42 启动：依据 NRIR-41 自动 Decision，只改变 scorer ownership/validation reuse；目标是在 exact
  semantics 下把 31-node enumeration calls 从 341 降到 31，并复测 global-budget production gate。

## Changes

- main 同步到 PR #52 merge `355e80b`，建立 `feat/objective-branch-scorer-ownership-v1`。
- 预注册 capsule exact parity、enumeration `31 compile/0 execute`、new/old median `<=0.75` 与有条件
  whole-query formal 门禁。
- 新增 `ValidatedBranchProgramCapsuleIR`、Plan-owned candidate scorer Task/Schedule、prevalidated runtime、
  additive production queue 与 multi-clause global composition；historical NRIR-39/40 文件未改。
- replay 会把 JSON capsule 重新构造为 typed Plan/Task/Schedule，并重算 candidate/score/selection/token；
  同步 token、score、call-count、deadline tamper 均 fail closed。

## Validation

- Phase A：clauses 2/3 median ratio=`0.706888/0.698486`，enumeration `341→31`，六组 exact parity；
  formal hash=`0d310c2ffc96844648a83f9921bc7f353ec8425986bccb36f75e6d1cd2b25b58`。
- Phase B：三轮 accepted nodes 均 `[31,31]`，whole=`57.175184/57.697757/58.114412 s`，formal hash=
  `7274e834b3bf08a9e138fa3284b70222620cf3c571395331e1a87ed5fee7d759`。
- targeted `10 passed`；全量 `958 passed, 37 skipped, 7 warnings`；Black/mypy clean、Pylint 10.00/10；
  两个 artifact replay 均通过。

## Decisions

- 不修改 historical scorer 或 NRIR-39/40 frozen artifact；不得把删除 fail-closed validation 当作优化，
  新 capsule 必须以更便宜的 immutable token 保持相同拒绝能力。
- Phase A/B 自动 gate 均通过，NRIR-42 关闭为 `VALIDATED-REDUCED`；恢复 NRIR-40 在固定 ResNet2B
  property 0、CPU8、global-60s 范围内的 production admission，但不撤销 final unknown。

## Follow-Ups

- 下一阶段只允许新增 cross-clause/node/candidate batching/schedule ownership 并以相同语义作配对；
  不得声明 GPU、multi-workload、property closure、competitor speedup 或 ASPLOS-ready。

## Links

- plan: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SCORER_OWNERSHIP_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
