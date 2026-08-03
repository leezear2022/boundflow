---
status: validated-reduced
updated: 2026-08-04T05:45:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_REAL_NETWORK_SLICED_BATCH_EXECUTION_V1
stage: s01
---

# Native Real-Network Sliced Batch Execution v1 Plan

## Goal

- 关闭“BatchCandidate 只改变 metadata/hash，未改变执行”的缺口：让 query-time spec batch
  上限选择不同 PlanInstance，并由 source Schedule 的精确 objective ranges 驱动多个 native
  child Bound/Plan/Task/Schedule stack，最后按 spec 轴聚合结果。

## Scope

- v1 只实现 `spec` 轴；`domain` 与 `sample` 是独立维度，显式留待后续，禁止把三轴混为一谈。
- 同一 source BoundModule/PlanTemplate 包含 full-query 与固定宽度 spec-sliced
  BatchCandidate；`PlanSelectionContext.max_spec_batch_size` 决定准入。
- full path 保持单 child；sliced path 为每个连续 spec range 编译独立静态 child stack，实际执行
  后沿 spec 维拼接 lower/upper。
- source controller storage 仍是完整 query ledger；本阶段不实现物理 sliced allocator，也不作
  memory、latency、CUDA、OOM、Pareto 或 speedup claim。
- NRIR-4 dense/structured 两条路径必须继续回归，但 representation × batch 联合执行不在 v1
  closure 内，作为下一联合规划门禁。

## Tasks

1. 为 query-time selection context 增加三个互不混淆的可选 batch 上限，并将非默认上限纳入
   PlanInstance provenance/hash；默认路径不得改变历史 artifact identity。
2. 扩展 `QueryBatchSlice` 的可选 `[start_index, stop_index)`，使 reduced spec BatchDecision
   lower 为连续、无重叠、完整覆盖 objective 的 source `BatchLoopAction(axis="spec")`。
3. 新增 full/spec-sliced Plan variant compiler；保持 storage cost/peak 不变并写入
   `no_memory_claim`、`batch_policy_cost_not_benchmarked` 风险标签。
4. 编译每个 selected range 的 native child Bound/Plan/Task/Schedule stack，冻结 source
   Schedule→binding trace→child hashes 的一一映射。
5. 顺序执行 child stacks，验证原 objective digest，按 spec 轴聚合 lower/upper，并记录 child
   query/task trace 与结果 digest。
6. 在固定 VNN-COMP 2021 ResNet2B prop0 上生成可重放 artifact，并加入同步重哈希后的范围、
   query lineage、gate、execution trace 篡改拒绝测试。

## Validation

- toy residual：full 2 specs 与 2×1 sliced lower/upper bitwise equal；编译 hash deterministic；
  objective、Schedule range、binding range、selection provenance tamper 均 fail closed。
- frozen ResNet：full 9-spec path 为 1 个 21-op child；`max_spec_batch_size=3` 选择 3 个
  21-op child，范围恰为 `[0,3)`、`[3,6)`、`[6,9)`，共 63 Task/Launch。
- 两策略共享 source Bound/PlanTemplate；PlanInstance/Schedule 不同。full/sliced lower max diff
  `1.9073486328125e-06`；full/external 为 `7.152557373046875e-07`；sliced/external 为
  `1.9073486328125e-06`；三者 allclose，sign 9/9。
- artifact `generate` 与 fresh semantic `replay` 必须都 exit 0；manifest/payload digest、8 个
  ownership/correctness gates 与 claim boundary 全部 fail closed。
- 新旧 native/Plan/Task/Schedule 聚焦 `89 passed`；全量 `508 passed, 37 skipped`；Black、
  Mypy、Pylint 10.00/10 与 diff check 通过。DocOps validate/lint 在发布前记录于 changelog。

## Rollback

- Keep the merged NRIR-1—4 single-query execution path unchanged and make the
  sliced path additive.

## Acceptance boundary

- 本阶段可升级的唯一结论是：真实网络的 spec BatchDecision 已具有实际 compiler/runtime
  ownership，状态为 correctness/integration `VALIDATED-REDUCED`。
- 不得把顺序执行三个 child 写成吞吐优化；不允许用 estimated payload 或完整 controller
  ledger 推导物理 memory reduction。
- 下一联合门禁是 representation × batch policy composition；其后才可在可用设备上执行已冻结
  NRIR-3 protocol，并决定是否重开系统性能主张。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_SLICED_BATCH_EXECUTION_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
