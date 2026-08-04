---
status: active
updated: 2026-08-04T06:28:14Z
type: plan
topic: boundflow
slug: per-child-objective-refinement-v1
stage: s01
---

# BoundFlow Per-Child Objective Refinement V1 Plan

## Goal

- 将 NRIR-20 的 root-global objective-directed intermediate refinement 接入 native optimized
  ReLU-split BaB：每个被评估 child 必须依据自身 exact split state 重新执行 forward IBP、
  clause-sensitive influence extraction、target selection、selected CROWN 与 propagation。
- 让每个 node 的 refinement Plan/Task/Schedule、split identity 和 semantic execution trace 成为
  queue trace 的一等组成，并在固定 ResNet hard clauses 上验证同一 tree budget 的 frontier
  tightness 是否优于 root-global reuse。

## Scope

- 新增可选 per-child policy；启用时禁止同时传 root override，provenance 必须是
  `native_refined`，且只准入 single-clause objective-directed policy。
- 每个 node 独立编译 refinement IR；packed optimizer batch 只在 child-specific refined bounds
  拼接后执行。parent α/β 只作 monotonic warm initialization；parent refined bounds 不作为 child
  exact state。
- queue/evaluation trace 冻结 node split hash、refinement Plan/Task/Schedule hash、去 timing 的
  semantic trace hash、initial/final intermediate hash和 target count。
- 不包含 CUDA、latency/speedup、无限树、竞品公平性能或 ASPLOS-ready claim。

## Tasks

1. 建立 per-child refinement trace schema 和 queue/evaluation linkage。
2. 实现 node split→single-domain refinement compile/execute，并把多个 node 的 refined bounds
   精确拼成 optimizer domain batch。
3. 修正 per-child warm-state scope：从实际 parent node 的 bounds/state 构造 warm batch，禁止用
   child override 冒充 parent semantics。
4. 增加 admission、split lineage、parent warm-only、tamper、packed/serial 与数值 soundness测试。
5. 生成 ResNet clauses 0/1、7-node/depth-2、same-policy root-global/per-child artifact；冻结
   frontier lower、IR/trace hashes、source digest 与 semantic replay。
6. 更新 claims/status/memo/index，运行全量测试和 DocOps 门禁并发布 PR。

## Validation

- 所有 per-child trace 必须与 evaluation 一一对应；Plan split hash 必须等于 queue node split
  hash；child initial intermediate hash 不得伪装成 parent hash。
- root-global 和 per-child 的 root bound 必须一致；per-child 每个 refined bound 必须相对该
  node local split-forward bounds 单调收紧，并保持 selected-CROWN soundness。
- packed/serial 必须有相同 logical queue signature、node split、per-node refinement semantic IR
  与 bounds；允许 batch-owned optimizer trace ID 不同，但不得隐藏数值或 lineage 漂移。
- fixed artifact 以 worst frontier lower 为主；若 per-child 未严格改善，按 no-go 关闭该策略并
  转向更强 intermediate method，不扩大结论。
- 运行 focused/full pytest、Black、targeted Mypy、Pylint、fresh generate/replay、digest/tamper、
  `git diff --check` 和 DocOps validate/lint。

## Rollback

- per-child policy 为默认关闭的独立入口；删除该参数、trace 与执行路径即可恢复 NRIR-20
  root-global 行为，旧 queue/refinement payload 通过条件序列化保持 hash 兼容。

## Links

- changelog: `gemini_doc/BOUNDFLOW_PER_CHILD_OBJECTIVE_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_INTERMEDIATE_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
