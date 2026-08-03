---
status: completed
updated: 2026-08-04T03:22:00+08:00
type: plan
topic: boundflow
slug: native-real-network-bound-ir-v1
stage: s01
---

# Native Real-Network Bound IR v1 计划

## Goal

- 把一个固定 VNN-COMP ResNet-2B initial-CROWN 查询的主 backward 计算从单个
  `EXTERNAL_VERIFIER_CALL` 拆成 BoundFlow 自己拥有的多 region Bound/Plan/Task/Schedule IR。
- 用同一 αβ-CROWN external intermediate bounds 与 final lower 作为语义 oracle，先关闭
  correctness、IR identity 和 artifact replay，不提前产生性能主张。

## Scope

- 固定输入：VNN-COMP 2021 `resnet_2b.onnx`、prop0 VNNLIB、αβ-CROWN commit 与逐 ReLU
  intermediate bounds；所有文件和 tensor 都必须有 SHA-256 identity。
- 支持真实 17-op topology：6 Conv、6 ReLU、2 Add、1 Flatten、2 Linear。
- native lowering 覆盖 plain-CROWN backward；external verifier 仍提供 6 组 forward
  preactivation bounds，这一依赖必须显式披露。
- CPU reference correctness only；本切片不增加 storage/materialization alternatives、GPU
  backend 或 latency/memory claim。

## Tasks

- [x] 将 6 组 external intermediate bounds 序列化为 `weights_only=True` 可加载的 portable
  payload，并验证 ordinal/name/shape/dtype/tensor/aggregate digest。
- [x] 让 external intermediate-bound aggregate digest 进入 ReLU relaxation state version，
  从 Bound IR 开始贯穿五层 hash。
- [x] 新增 native plain-CROWN compiler，逐 Bound op 建 region，并 lower PlanInstance、Task IR
  与 Schedule IR；拒绝任何 external-call op/task。
- [x] 在固定 ResNet 上执行 21 个 native Bound ops / 21 Tasks / 21 launches，并与 external
  final lower 比较。
- [x] 生成 manifest + portable payload artifact，提供 pinned VNN-COMP input fetch 与 semantic
  replay CLI。
- [x] 跑全量 regression、Black/Mypy/Pylint/diff-check 与 DocOps lint，完成本切片交付。

## Validation

- Acceptance gates：
  - Primal topology 恰为 17 ops；native Bound graph 恰为 21 ops；
  - Bound/Task `EXTERNAL_VERIFIER_CALL` 均为 0；Task 与 Schedule launch 均为 21；
  - Bound/PlanTemplate/PlanInstance/Task/Schedule 五层 hash fresh replay 完全一致；
  - native lower 对 external lower `allclose(atol=rtol=2e-4)` 且 sign 9/9；
  - source/payload/model 任一 identity、顺序、shape 或 tensor 内容改变时 fail closed；
  - `performance_claimed=false`，manifest 保留单 storage/batch、0 materialization 限制。
- Focused validation：`22 passed`；全量 `468 passed, 37 skipped`；真实 generate/replay 均
  `status=ok`，max diff
  `7.152557373046875e-07`，sign `9/9`。
- Mypy 6 files clean；Pylint 6 files `10.00/10`；Black 与 `git diff --check` 通过。
- artifact：`artifacts/native-real-network-ir/vnncomp21-resnet2b-prop0-cpu-v1/`。

固定输入与 replay：

```bash
python scripts/fetch_native_real_network_ir_inputs.py \
  --output-dir /tmp/boundflow-vnncomp2021-native-ir
python scripts/run_native_real_network_ir_artifact.py replay \
  --model /tmp/boundflow-vnncomp2021-native-ir/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx \
  --artifact-dir artifacts/native-real-network-ir/vnncomp21-resnet2b-prop0-cpu-v1
```

## Rollback

- 本切片只新增 native correctness compiler/runner，并给 external capture payload 增加 portable
  字段；旧 artifact 不改。需要撤回时可删除新增 compiler、runner、tests 和 v1 artifact，保留
  既有 RVIR external-call 路径。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_CHANGELOG_2026_08_04.md`
- prior gate: `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_PLAN_2026_08_04.md`
- artifact: `artifacts/native-real-network-ir/vnncomp21-resnet2b-prop0-cpu-v1/`
- PR: `https://github.com/leezear2022/boundflow/pull/12`
