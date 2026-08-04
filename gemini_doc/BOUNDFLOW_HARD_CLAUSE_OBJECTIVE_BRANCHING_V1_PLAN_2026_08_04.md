---
status: completed
updated: 2026-08-04T04:05:15Z
type: plan
topic: boundflow
slug: hard-clause-objective-branching-v1
stage: s01
---

# Hard-Clause Objective Branching v1 Plan

## Goal

- 面向 fixed ResNet 尚未证明的 clauses `0/2/4`，把 ReLU branching 从与性质无关的 widest
  interval heuristic 升级为 objective-bound-impact selection。
- 先以 batched strong-branch estimate 证明分支选择确实改善最差子域下界，再把 policy、候选
  shortlist、score 与选择结果冻结成 typed/replayable evidence；不得以单一路径改善冒充完整证明。

## Baseline

- NRIR-15 external-adaptive root 仍为 6/9 verified，hard-clause lower 分别约
  `-0.5357/-0.5260/-0.5963`。
- clause 0 的现有 widest tree（7 nodes、depth 2）最差叶 lower 约 `-0.5168`；首个 split 为
  ReLU input `31` neuron `93`。
- batched 48-candidate fixed-state probe 只需约 `14.7 ms`，选出的 `31:17` 估计最差子域
  lower=`-0.4309`；实际 optimizer child lower=`-0.4210/-0.4239`。这只是校准探针，尚不是
  frozen claim。

## Scope

- v1 以每个 ReLU 的 deterministic top-width shortlist 为候选集合，用同一 objective、当前 split
  state 和 parent-selected alpha/beta state 批量计算 inactive/active 子域 lower；按 worst-child、
  mean-child、稳定 identity 顺序选分支。
- 原 widest runtime/schema/artifact 默认行为不变；objective branching 采用独立 policy/trace，
  score batch 必须由显式 Plan/Task/Schedule IR 驱动或在门禁中明确仍缺失。
- 完整性质状态仍只能由 closed queue + concrete unsafe replay 推导；bounded-tree 改善只能记作
  tightness evidence。

## Tasks

1. [x] 复现 widest depth-2 tree，并量化最差叶 improvement。
2. [x] 完成 objective-aware batched strong-branch feasibility probe。
3. [x] 实现 typed objective branching policy、score records 与 fail-closed validation。
4. [x] 将 score evaluation 编译为 first-class Plan/Task/Schedule，并接入 optimized queue。
5. [x] 对 clauses `0/2/4` 执行 frozen depth/node sweep，报告 closed/unknown、最差叶 lower、
   node/time/memory，而非只报告最佳路径。
6. [x] replay、tamper、focused/full regression、Black/Mypy/Pylint/DocOps 全关闭。

## Acceptance

- objective policy 在 deterministic shortlist 上必须选择最大 worst-child lower；任一 objective、
  split、candidate、score 或 schedule 漂移必须 fail closed。
- clause 0 首层必须独立复现实测 improvement；三 hard clauses 至少不能弱于 widest 对照。
- 若有限预算仍不能证明，必须给出 frontier deficit 分布和下一紧界机制，不能升级 ASPLOS claim。

## Links

- predecessor: `gemini_doc/BOUNDFLOW_PREPARED_PRODUCTION_FAST_PATH_V1_CHANGELOG_2026_08_04.md`
- changelog: `gemini_doc/BOUNDFLOW_HARD_CLAUSE_OBJECTIVE_BRANCHING_V1_CHANGELOG_2026_08_04.md`
- artifact: `artifacts/hard-clause-objective-branching/vnncomp21-resnet2b-prop0-cpu-v1/`
