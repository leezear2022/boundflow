---
status: completed
updated: 2026-08-04T04:05:15Z
type: changelog
topic: boundflow
slug: hard-clause-objective-branching-v1
stage: s01
---

# Hard-Clause Objective Branching v1 Changelog

## Summary

- NRIR-17 已完成。objective-bound-impact branching 在同一 7-node/depth-2/25-step optimizer
  预算下，对 clauses `0/2/4` 的 worst leaf 相对 widest 分别改善
  `+0.120752/+0.071564/+0.057901`，但三者仍全部为 unknown。

## Changes

- 新增 objective branch Plan/Task/Schedule IR：deterministic shortlist、inactive/active child
  materialization、child bound evaluation、worst-child reduction 与 argmax selection 均为 first-class
  action；objective/split/selected-state/scope/policy/candidate 全进入 stable hash。
- 新增 batched score executor 与 optimized queue opt-in。每个节点使用 parent-selected alpha/beta
  state 批量估计候选两侧 lower，按 worst-child、mean-child、stable identity 选择；默认 widest
  路径与 NRIR-15/16 schema/replay 不变。
- 新增 explicit external-constraint forward refinement opt-in；它逐 ReLU 相交 external 与当前
  split-constrained local interval 后继续传播。fixed ResNet 探针影响仅约 `1e-5`，没有包装成
  material tightness claim。
- float32 packed selected-native equivalence gate 调整为 `atol=rtol=1e-5`；正式工件观测到 lower/
  upper max diff=`1.52588e-5/4.88758e-6`，仍远低于 serialized trace ceiling `2e-3`。
- 新增 formal generate/replay runner、1.20 MB frozen artifact、digest/claim/selection/schedule tamper
  tests。

## Validation

- same-budget widest→objective worst leaf：clause 0 `-0.440550→-0.319799`，clause 2
  `-0.498173→-0.426609`，clause 4 `-0.562577→-0.504676`；三组首分支均选择 `31:17`。
- objective 三树 terminal leaves 仍全部为负，因此 property status 明确保留 `unknown`；未把最佳
  leaf 或单一路径 improvement 冒充 complete proof。
- artifact fresh replay evidence hash=
  `1193bee8817e4acc9ec33f8ddadc00a671d0ac3c9411f14f62978eb5ab1a95bd`。
- focused runtime/IR/artifact/tamper `16 passed`；全量 `707 passed, 37 skipped`；新增/相关模块
  Black check、targeted Mypy、Pylint 10.00/10 与 diff check 通过；NRIR-16 冻结工件 replay 仍为
  `e14fcd62b322c0bc60d45c726cf94a7aa6cfb8d7aa3212662d08996db169b6b2`。

## Decisions

- objective branching mechanism 与 fixed-budget tightness evidence 以 `VALIDATED-REDUCED` 关闭；
  它显著优于 widest，但不足以关闭 fixed property。
- 单次 audit wall time 约 `20.38–21.73 s`，仅为 runner 诊断，不是 performance claim。下一主线
  必须转向多 workload/设备/竞品 E2E 协议与更强 bound mechanism，不能只继续堆深同一小树。

## Links

- plan: `gemini_doc/BOUNDFLOW_HARD_CLAUSE_OBJECTIVE_BRANCHING_V1_PLAN_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_PREPARED_PRODUCTION_FAST_PATH_V1_CHANGELOG_2026_08_04.md`
- artifact: `artifacts/hard-clause-objective-branching/vnncomp21-resnet2b-prop0-cpu-v1/`
