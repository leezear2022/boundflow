---
status: completed
updated: 2026-08-04T03:48:56+08:00
type: changelog
topic: boundflow
slug: BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1
stage: s01
---

# Native Real-Network Memory Plans v1 变更记录

## Summary

- NRIR-2 在固定真实 ResNet native Bound graph 上实现两个 storage plan、预算驱动的
  PlanInstance/Schedule 切换与运行时 last-use release。
- 所有 correctness/ownership 门禁已通过；性能、CUDA allocator、OOM rescue、structured
  materialization 与 sliced batching 仍未声明完成。

## Changes

- `boundflow/planner/storage_plan_variants.py`：从一个 dense baseline 派生 retain-all 与
  lifetime-reuse candidate；后者用 aligned interval allocation 对不重叠 lifetime 做 arena alias。
- `boundflow/runtime/storage_plan_runtime.py`：Task 前验证 resident input，Task 后按 selected
  binding 释放中间值，记录 planned/observed peak、逐 Task release 与 stable trace hash。
- `boundflow/runtime/task_ir_executor.py`：增加默认关闭的 storage runtime hook；旧路径行为与
  Task trace schema 不变。
- `boundflow/runtime/native_verifier_ir_integration.py`：新增 memory compile/execute 入口；同一
  template 接受 query-time memory budget 并返回 storage trace。
- `scripts/run_native_real_network_memory_plans_artifact.py`：固定 real ResNet 双计划 generate/
  replay，验证 parent artifact digest、五层 IR identity、budget switch、alias/release 与语义。
- `tests/test_native_memory_plans.py`：small residual 的预算切换、数值一致、低预算 fail-closed 与
  trace 篡改测试。
- `tests/test_native_real_network_memory_artifact.py`：冻结 artifact contract、精确数字与同步
  semantic tamper rejection。
- 新增 `artifacts/native-real-network-memory-plans/vnncomp21-resnet2b-prop0-cpu-v1/`。

## Validation

- real artifact generate/replay：8/8 gate PASS。
- NRIR-1 原 artifact replay：PASS，五层 hash 不变。
- focused contract/artifact tests：`7 passed`；全量回归：`473 passed, 37 skipped`，37 个
  skip 均为既有 CUDA/环境边界。
- Black、`git diff --check`、Mypy 5 files clean、Pylint 5 files `10.00/10`。

## Decisions

- 不将已有 Plan representation metadata 当作真实 structured execution。审计发现 structured
  runtime 目前依赖另一份 rewritten Bound module；Schedule `MaterializeAction` reference path
  只追踪动作而不转换 runtime value。
- NRIR-2 v1 因此只关闭真实 storage axis：计划选择改变 arena、生命周期和 runtime residency。
- `0.001 ms` 只用于 deterministic policy ordering，artifact 明确标记
  `policy_cost_not_benchmarked`；不得作为 latency 证据。
- 逻辑 arena 从 `1,860,912` 降到 `442,656` bytes 不等于 CUDA allocator peak。需要 fresh
  device measurement 后才能写 memory reduction/performance claim。

## Follow-Ups

- 首选下一门禁：在 fresh CUDA protocol 上测 retain/reuse 的
  `torch.cuda.max_memory_allocated/reserved`、latency 与 baseline OOM rescue；若当前设备不可用，
  先冻结 runner/protocol，不伪造结果。
- representation 路线必须先让 selected representation 进入 Task/backend semantic binding，
  并让 materialization 对 runtime value 真正生效，再加入真实 ResNet template。
- batch 路线必须让 BatchLoop slice 驱动 objective/spec tensor slicing 与结果重组，不能只改变
  Plan metadata。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1_PLAN_2026_08_04.md`
- artifact: `artifacts/native-real-network-memory-plans/vnncomp21-resnet2b-prop0-cpu-v1/`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
