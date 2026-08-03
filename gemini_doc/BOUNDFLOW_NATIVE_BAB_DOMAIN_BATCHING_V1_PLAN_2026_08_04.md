---
status: completed
updated: 2026-08-03T22:04:06Z
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_BAB_DOMAIN_BATCHING_V1
stage: s01
---

# Native BaB Domain Batching v1 Plan

## Goal

- 将固定 ResNet 原始 VNNLIB input box 确定性二分为 8 个不同 leaf domains；用 Plan 的
  domain BatchCandidate 选择 packed-4，实际执行 2 个 native child stacks，和 8 次 serial
  same-policy domain execution 对齐并恢复 per-domain results/state lineage。

## Scope

- v1 只实现 input-box domain batching，不实现 ReLU split/β optimization 或完整 BaB queue。
- 每个 leaf domain 独立重算 IBP intermediate bounds/state；parent state 明确标记
  `WARM_START_ONLY` 且不作为 child exact execution input。
- source BatchCandidate 同时有 full-domain 与 size-4；query-time max domain size 驱动 source
  PlanInstance/Schedule query slices。
- 每个 Schedule slice 编译/执行一个 batched-domain child Bound/Plan/Task/Schedule stack，结果按
  domain/query order 恢复。
- CPU correctness/ownership only；不声称 prune、TTVerify、latency、memory、CUDA 或 speedup。

## Tasks

1. 新增 deterministic domain-batch Plan variants，保持 spec/sample 轴独立。
2. 新增 typed domain query/state lineage、source Schedule→child stack binding 与 execution trace。
3. 实现 InputSpec/interval/relu/C 的 domain-axis slicing和 domain-axis result aggregation。
4. 实现 serial same-policy domain reference 与 parent-state non-consumption gate。
5. toy different boxes 与 fixed ResNet 8-leaf artifact/replay/tamper tests。

## Validation

- packed 2 child vs serial 8 child；8/8 domain IDs/parents/boxes/state hashes/results 恢复。
- packed/serial lower/upper allclose；parent state never consumed as exact child state。
- full-domain vs packed-4 source Plan/Schedule identity不同，默认 NRIR-1—7 artifact 不变。
- 全量 pytest、Black/Mypy/Pylint/diff/DocOps 全过。

## Rollback

- additive domain runtime；不修改冻结 NRIR-7 property-query API/artifact。

## Completion boundary

- 只升级 different-domain formation/state validity/packing/restore 为 VALIDATED-REDUCED。
- 8 vs 2 child 仍是机制计数；完整 BaB queue/branch/β/prune 和公平 performance 继续 pending。
- Closure validation：fixed ResNet generate/replay 通过；聚焦 `19 passed`；全量
  `559 passed, 37 skipped`；Black/Mypy/Pylint/diff/DocOps 门禁通过。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_BAB_DOMAIN_BATCHING_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
