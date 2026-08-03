---
status: validated-reduced
updated: 2026-08-04T06:25:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_REPRESENTATION_BATCH_COMPOSITION_V1
stage: s01
---

# Native Representation Batch Composition v1 Plan

## Goal

- 在同一真实 ResNet source BoundModule/PlanTemplate 内联合 representation 与 spec-batch 两个
  Plan 轴，使 memory budget × query-time spec limit 选择 dense/structured × full/sliced 四组合，
  并让每个组合的实际 child execution IR 与 source decision 一致。

## Scope

- 复用 NRIR-4 的 source representation binder 与 NRIR-5 的 exact spec ranges/aggregation。
- source template 先加入全局 dense/structured-affine policy，再加入 full/spec-size-3 batch
  candidate；storage compatibility 必须传播到两个 batch candidate。
- source selector 由 budget 决定 dense/retain 或 structured/reuse，由 max spec size 决定
  full/sliced；不得为四组合写硬编码 execution branch。
- child compilation 必须显式继承 source selected storage/representation policy，不能依赖 child
  shape 变化后重新打分“碰巧选中”同一 policy。
- v1 仍为 CPU correctness/ownership；structured storage dense-equivalent、child 顺序执行，
  不声称 memory/latency/CUDA/OOM/Pareto/speedup。

## Tasks

1. 为 `PlanSelectionContext` 增加可选 required storage candidate，并在 generic selector、
   PlanInstance provenance/hash 与 tamper verifier 中 fail closed；默认路径保持历史 artifact hash。
2. 允许 representation compiler 接收显式 selection context，供 joint compiler 传播 source policy。
3. 新增 joint source compiler：representation variants → spec-batch variants → single selector →
   source Schedule → representation binding → exact child representation compilations。
4. 新增联合 binding/execution trace，逐 slice 绑定 source representation policy、batch range、child
   source/execution五层 IR 与结果聚合。
5. toy residual 跑 dense/structured × full/sliced 四组合与 policy/range/hash tamper tests。
6. 固定 ResNet 生成四组合 artifact/replay，验证 source identity、policy cross-product、NRIR-4
   transitions、NRIR-5 ranges/63 child launches 与 external semantics。

## Validation

- 四组合共享 source Bound/PlanTemplate，四个 source PlanInstance/Schedule identity 均按 decision
  区分；同 representation 的 full/sliced child policy必须一致。
- dense full/sliced、structured full/sliced 的 lower/upper 均与 frozen external oracle allclose、
  sign 9/9；联合路径间比较不得放宽 NRIR-4/5 的 `2e-4/2e-4` 语义门限。
- NRIR-1—5 artifacts replay 不变；全量 pytest、Black、Mypy、Pylint、diff、DocOps 门禁全过。
- 实测：新旧 native/Plan/Task/Schedule 聚焦 `103 passed`；全量
  `522 passed, 37 skipped`；Black/Mypy clean、Pylint 10.00/10、diff check 通过。

## Rollback

- 联合 compiler/runtime 作为 additive module；NRIR-4 representation-only 与 NRIR-5
  spec-slice-only public APIs、默认 selector context 和冻结 artifacts 不改。

## Completion boundary

- 只允许把“两个 Plan 轴已在同一 source template 中联合选择并驱动 execution”升级为
  `VALIDATED-REDUCED`。
- 下一门禁是跨 query/domain 的真实 repeated-query batching + cache/accounting；在真实物理
  baseline 与 frozen CUDA protocol 通过前，ASPLOS performance 状态继续 NO-GO。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_REPRESENTATION_BATCH_COMPOSITION_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
