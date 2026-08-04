---
status: completed
updated: 2026-08-04T03:15:59Z
type: plan
topic: boundflow
slug: prepared-production-fast-path-v1
stage: s01
---

# Prepared Production Fast Path v1 Plan

## Goal

- 把 NRIR-15 定位的重复静态验证、compiler/hash 和 selected-native 双执行移出 steady-state，
  建立一个仍由 optimizer Plan/Task/Schedule 驱动、但不构造 audit hash chain 的 prepared
  production root-query path。
- 分离一次性 preparation 与 repeated execution；任何 performance 数字必须同时披露 cold setup、
  warm steady-state、audit 对照和 exact semantic equivalence。

## Scope

- v1 只支持固定 module/input/objective/threshold/policy/intermediate semantics 的 root-bound
  conjunction；不支持 child split、dynamic queue 或跨 semantic scope 复用。
- preparation 完整运行 program/compiler validation 并冻结 exact program hashes/scope；production
  每次仍逐个消费 Optimizer Task/Schedule action，但不生成逐 action tensor hash/trace，也不再执行
  selected-native validation stack。
- audit path 与 NRIR-10—15 schema/hash/默认行为保持不变；production trace 必须明确
  `performance_claimed=false`，直到多 workload/设备/竞品协议关闭。

## Tasks

1. [x] 分解 fixed ResNet clause-0 audit wall time，确认 compile/validate/hash/re-execution 占比。
2. [x] 实现 exact `NativePreparedOptimizerProgram` 与无 audit hash-chain 的 Schedule-driven executor。
3. [x] 实现 prepared root conjunction preparation/execution、sound unsafe candidate replay 与 typed trace。
4. [x] 生成 local/external audit vs prepared production 的 3-group CPU artifact；分别记录 prepare、
   candidate、bound、aggregation 与 complete-query wall time。
5. [x] replay、tamper、focused/full regression、Black/Mypy/Pylint/DocOps 全关闭。

## Validation

- production lower/upper、selected alpha/beta state 与 audit optimizer 必须逐张量一致；九子句状态
  必须仍为 6 verified / 3 unknown，candidate unsafe 仍需 concrete replay。
- objective/input/intermediate source/scope/program identity 任一漂移必须 fail closed。
- performance comparison 至少 3 个轮换 group；cold setup 不得摊入或藏在 warm latency 之外。

## Rollback

- prepared path 为独立 opt-in；删除新 capsule/query/runner 后，NRIR-15 audit path 与 artifact
  保持不变。

## Links

- predecessor: `gemini_doc/BOUNDFLOW_END_TO_END_TIGHTNESS_PERFORMANCE_BASELINE_V1_PLAN_2026_08_04.md`
- changelog: `gemini_doc/BOUNDFLOW_PREPARED_PRODUCTION_FAST_PATH_V1_CHANGELOG_2026_08_04.md`
- artifact: `artifacts/prepared-production-fast-path/vnncomp21-resnet2b-prop0-cpu-v1/`
