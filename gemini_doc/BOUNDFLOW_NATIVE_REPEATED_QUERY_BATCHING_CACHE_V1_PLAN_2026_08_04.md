---
status: validated-reduced
updated: 2026-08-04T07:00:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_REPEATED_QUERY_BATCHING_CACHE_V1
stage: s01
---

# Native Repeated-Query Batching and Cache v1 Plan

## Goal

- 把一个真实 ResNet property 的 9 个不同 linear objectives 建模为 9 条独立 verification
  queries，形成 3-query physical spec batches，执行后按 query lineage 恢复结果，并以 9 次
  serial same-policy execution 作为 correctness baseline。

## Scope

- query contract 显式包含 query ID、objective tensor digest/range 与 workload/state identity。
- compatibility 仅允许共享 model/input domain/intermediate-bound state、dtype/device、representation
  policy 的 query 进入同一 physical batch。
- packed path 复用 NRIR-6 joint compiler 的 exact ranges/child stacks，但新增 query→range 映射、
  per-query split/result digest 与 observable cache hit/miss。
- compile cache key 必须包含 workload identity、state identity、query order/content、policy/budget/
  batch size；不得用进程对象 ID、模糊 shape-only key 或未验证父状态复用。
- v1 只做同一 input domain 上的不同 property queries；BaB domain/parent-child state validity 是
  下一独立门禁。

## Tasks

1. 新增 typed repeated-query spec/layout/binding/execution trace 与 fail-closed verifier。
2. 新增 deterministic in-process compilation cache，首次 miss 编译，完全相同 stream hit；query
   内容、顺序、state/workload/policy 任一变化必须 miss。
3. packed executor 使用 9-spec joint compilation + spec-size-3 slices，实际执行 3 个 child stacks，
   并按 9 条 query 的 exact offsets 恢复结果。
4. serial reference 对 9 条 query 分别使用 source-selected representation/storage policy 编译/执行；
   和 packed 逐 query 比较 lower/upper。
5. toy query stream 测 lineage/cache/tamper；fixed ResNet 生成 replay-grade artifact。

## Validation

- packed 3 child vs serial 9 child；9/9 query IDs、ranges、result digests 与顺序完整恢复。
- cache first miss/second hit；objective/state/order tamper 改变 key或在 runtime fail closed。
- packed/serial/external 均 allclose、sign 9/9；不报告 timing/speedup。
- NRIR-1—6 replay、全量 pytest、Black/Mypy/Pylint/diff/DocOps 全过。
- 实测：新旧 native/Plan/Task/Schedule 聚焦 `121 passed`；全量
  `540 passed, 37 skipped`；Black/Mypy clean、Pylint 10.00/10、diff check 通过。

## Rollback

- 新 runtime 为 additive；NRIR-6 single-query joint compiler API/artifact 保持不变。

## Completion boundary

- 只升级 real repeated-query formation、physical spec packing、cache/lineage ownership 为
  `VALIDATED-REDUCED`。
- 不将 3 vs 9 次 child execution 直接写成 speedup；必须另有公平重复 timing 与成熟 batched
  baseline。domain/BaB state validity 与 CUDA 仍 pending。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_REPEATED_QUERY_BATCHING_CACHE_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
