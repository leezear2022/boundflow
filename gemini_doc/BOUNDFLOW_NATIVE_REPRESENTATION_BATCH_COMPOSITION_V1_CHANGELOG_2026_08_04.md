---
status: completed
updated: 2026-08-04T06:25:00+08:00
type: changelog
topic: boundflow
slug: BOUNDFLOW_NATIVE_REPRESENTATION_BATCH_COMPOSITION_V1
stage: s01
---

# Native Representation Batch Composition v1 Changelog

## Summary

- NRIR-6 在 NRIR-5 合并后启动，目标是消除 storage/representation/spec-batch 三者只独立成立、
  尚未联合执行的缺口。

## Changes

- 用 DocOps `doc new --dir gemini_doc` 直接创建符合仓库约定的 plan/changelog，避免上一阶段的
  路径迁移。
- `PlanSelectionContext.required_storage_candidate_id` 新增 generic policy propagation；非默认
  约束进入 PlanInstance provenance/hash，tamper 时由 Plan verifier 拒绝，默认 artifact identity
  不变。
- representation compiler 接受可选 selection context；joint compiler 在同一 source template
  依次加入 representation/storage 与 spec-batch variants，由单次 selector 得到四组合之一。
- source selected storage/representation 显式传播到每个 child；child shape 改变后不得自行重选。
- 新增 joint binding/execution trace、真实 child aggregation、ResNet artifact generate/replay 与
  range/policy/query/gate/claim tamper tests。

## Validation

- toy residual dense/structured × full/sliced 四组合 lower/upper bitwise equal；四个 source
  PlanInstance/Schedule identity distinct，child policy 继承 source。
- fixed ResNet 四组合 child op/task/launch 为 `21/63/49/147`；structured source 保留 28
  transition 与 49-op execution ownership；9 个 gates 全 true。
- 对 external lower max diff：dense-full `7.152557373046875e-07`、dense-sliced
  `1.9073486328125e-06`、structured-full `9.5367431640625e-07`、structured-sliced
  `1.6689300537109375e-06`；均 allclose、sign 9/9。
- artifact generate/replay exit 0；聚焦 `103 passed`；全量
  `522 passed, 37 skipped, 7 warnings in 107.65s`。
- Black/Mypy clean、Pylint 10.00/10、`git diff --check` 通过。
- DocOps change `ev001198`、validation pass `ev001199` 已记录；`dol validate` 与
  `dol lint --soft` 均通过。

## Decisions

- source policy propagation 必须是显式 selector contract；child shape 改变后重新打分不构成
  可审计的联合 policy ownership。
- 本阶段继续禁止 performance/memory claim。

## Follow-Ups

- 发布并合并 NRIR-6。
- 下一切片实现跨 query/domain 的真实 repeated-query batching、plan/cache reuse 与 per-query
  lineage/accounting；在公平 physical baseline 前继续禁止 performance claim。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_REPRESENTATION_BATCH_COMPOSITION_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
