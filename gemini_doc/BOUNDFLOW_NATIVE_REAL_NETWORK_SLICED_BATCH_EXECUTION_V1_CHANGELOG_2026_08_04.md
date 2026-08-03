---
status: completed
updated: 2026-08-04T05:45:00+08:00
type: changelog
topic: boundflow
slug: BOUNDFLOW_NATIVE_REAL_NETWORK_SLICED_BATCH_EXECUTION_V1
stage: s01
---

# Native Real-Network Sliced Batch Execution v1 Changelog

## Summary

- NRIR-5 已把真实 ResNet 的 spec-axis BatchCandidate 从 metadata 变成实际 source Schedule
  objective ranges、三个 native child compiler stacks 与结果聚合，并以
  correctness/integration `VALIDATED-REDUCED` 关闭。

## Changes

- Moved the latest DocOps-generated blank templates from `docs/planning/` to
  `gemini_doc/` to preserve the repository documentation contract.
- `PlanSelectionContext` 新增互相独立的 domain/spec/sample batch 上限；非默认上限进入
  PlanInstance provenance/identity，默认 context 保持历史 hash 兼容。
- `QueryBatchSlice` 新增可选半开区间；spec loop verifier 要求范围连续、不重叠、完整覆盖且
  width 不超过 selected BatchCandidate。domain legacy loop 的 dump/hash 保持不变。
- 新增 `spec_batch_plan_variants.py`，从 full-query 派生 spec-sliced candidate，明确复用原
  storage peak 和 latency placeholder，不声称 memory/performance。
- 新增 `native_sliced_batch_integration.py`：source Plan/Schedule 选择、每个 range 的 child
  Bound/Plan/Task/Schedule 编译、binding trace、真实执行、spec aggregation 与 execution trace。
- 新增 toy 正反向/篡改测试，以及真实 ResNet artifact generate/replay runner、manifest 和冻结
  artifact contract/tamper tests。

## Validation

- 新旧 native/Plan/Task/Schedule 聚焦：`89 passed`，2 个 NVML unavailable warnings。
- Mypy 对新增 planner/runtime/runner 首轮 clean；一次把 runner 文件与导入它的 test 同时作为
  path 参数时触发 mypy 的 duplicate-module invocation，属于调用方式问题，收尾改用 package
  module/分组命令复核。
- artifact generate/replay 均 exit 0，8 个 gates 全 true。
- frozen ResNet：full=1 child，sliced=3 children；spec ranges `0:3/3:6/6:9`；63/63
  child Task/Launch；full/sliced lower max diff `1.9073486328125e-06`，external sign 9/9。
- 全量：`508 passed, 37 skipped, 7 warnings in 91.68s`；37 个 skip 与 NRIR-4 基线一致，
  均为 CUDA/TVM 环境边界。
- Black clean；Mypy 两组命令 clean；Pylint `10.00/10`；`git diff --check` 通过。
- NRIR-1—4 frozen artifact/Plan/Task/Schedule regression 均包含在 89-test 聚焦集合中。
- DocOps change `ev001143`、validation pass `ev001144` 已记录；`dol validate` 与
  `dol lint --soft` 均通过。

## Decisions

- Batch metadata or a Schedule loop alone is insufficient; the selected slice
  sizes must affect actual Task execution and result/query accounting.
- v1 只关闭 spec-axis correctness/ownership；domain/sample 和 representation × batch joint
  execution 继续 fail closed 为 pending。
- source controller storage 在 full/sliced 两条路径均保留完整 ledger，不能从 slice payload
  推导物理峰值下降。

## Follow-Ups

- 发布并合并 NRIR-5。
- 下一代码切片实现 representation × batch policy composition，验证 dense/structured ×
  full/sliced 四组合共享 source semantics 且各自拥有可审计 execution IR。
- CUDA device 可用时严格执行已冻结 NRIR-3 protocol；无设备时不伪造性能结论。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_SLICED_BATCH_EXECUTION_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
