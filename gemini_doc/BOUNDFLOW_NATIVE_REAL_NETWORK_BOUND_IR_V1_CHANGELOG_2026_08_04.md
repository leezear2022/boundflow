---
status: completed
updated: 2026-08-04T03:22:00+08:00
type: changelog
topic: boundflow
slug: native-real-network-bound-ir-v1
stage: s01
---

# Native Real-Network Bound IR v1 变更记录

## Summary

- 完成 NRIR-1：固定 VNN-COMP ResNet2B 的 initial-CROWN backward 主计算不再由一个
  external opaque op 包装，而是生成并执行 21 个 native Bound/Task regions。
- 五层 IR identity 绑定 model/objective 和 external intermediate-bound payload；CPU semantic
  replay 对 αβ-CROWN final lower max diff `7.152557373046875e-07`、sign `9/9`。
- 结论范围为 correctness/compiler ownership `VALIDATED-REDUCED`；没有性能升级。

## Changes

- `boundflow/runtime/abcrown_adapter.py`：portable external intermediate-bound serialize/
  deserialize、safe-load、digest/tamper validation 与 process-independent binding。
- `scripts/replay_pr14_abcrown_initial_crown.py`：payload v2 落盘 6 组 external bounds 及逐 tensor
  identity，旧 artifact 保持不变。
- `boundflow/frontends/plain_crown_bound_ir.py`：external aggregate digest 进入 ReLU relaxation
  输出 state version，使同形状但不同内容的 oracle 改变 Bound IR hash。
- `boundflow/runtime/native_verifier_ir_integration.py`：native Bound→PlanTemplate/Instance→Task→
  Schedule compiler、cross-layer verifier 与 reference execution。
- `scripts/run_native_real_network_ir_artifact.py`：固定输入准入、generate/replay、五层 hash、
  topology/action/semantic gate；`scripts/fetch_native_real_network_ir_inputs.py` 提供 pinned input。
- 新增 portable payload 与 native residual compiler tests；真实 artifact 位于
  `artifacts/native-real-network-ir/vnncomp21-resnet2b-prop0-cpu-v1/`。

## Validation

- Source capture：6 bounds，aggregate SHA
  `d51615b04dfb205afd67d2c21680ece4ca92f693157da1e32c7f8202a8e08cf1`。
- Real topology：17 Primal ops → 21 Bound ops → 21 Tasks → 21 Schedule launches；Bound/Task
  external-call count 均为 0。
- 五层 hash：
  - Bound `16e27f318f43be8df7571e16c0dd84657462dcc964bb360a553dc142961080fb`；
  - PlanTemplate `3b5b7e4b5ae4a78f3e97c83554726885b3d23adf7bc8173675d0ba87df539ed1`；
  - PlanInstance `5c11675627407ce5e5326f44b0c6ea77c2fb9b85c7dfccc4270612d045384506`；
  - Task `299bbd6416bc15518b57590365f62614c20e308d43506651a7ed3a1f1c001692`；
  - Schedule `a8c90b32b73d69ae0cabf26c17310f43d7c68efb22bfb2bd84fae22f351126a8`。
- Focused tests：`22 passed`，含同步重算内部 digest 后仍拒绝的负向探针；artifact
  generate/replay 均输出 `status=ok`。
- 全量回归：`468 passed, 37 skipped`；37 个 skip 均为既有 CUDA/环境边界。
- Mypy 6 files clean；Pylint 6 files `10.00/10`；Black 与 `git diff --check` 通过。

## Decisions

- NRIR-1 证明 real-network main CROWN backward 已进入一等 IR，不再用 typed external wrapper
  冒充 native compiler coverage。
- external intermediate bounds 仍是 semantic dependency；因此不得描述成完整 native
  αβ-CROWN verifier，也不得用本 CPU artifact 形成 acceleration claim。
- 当前 Plan 仍只有 1 个 dense storage、1 个 full-query batch、0 materialization；不能据此
  撤销 P0 memory No-Go。

## Follow-Ups

- 2026-08-04 更新：NRIR-2 已完成 storage-axis 双计划、预算 decision switch 与 runtime
  last-use release，详见 memory-plans changelog。它没有声称 representation/materialization 或
  sliced batch 已完成。
- 下一门禁是 fresh CUDA physical-memory/OOM protocol；若 device 不可用，则先实现 selected
  representation→Task/backend semantics 与 runtime materialization bridge。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_PLAN_2026_08_04.md`
- prior gate: `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_CHANGELOG_2026_08_04.md`
- artifact: `artifacts/native-real-network-ir/vnncomp21-resnet2b-prop0-cpu-v1/`
- PR: `https://github.com/leezear2022/boundflow/pull/12`
