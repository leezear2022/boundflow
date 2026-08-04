---
status: completed
updated: 2026-08-04T00:16:51Z
type: changelog
topic: boundflow
slug: native-optimized-relu-split-bab-v1
stage: s01
---

# Native Optimized ReLU-Split BaB v1 Changelog

## Summary

- NRIR-12 在 PR #22 合并后启动；目标是将 fixed-step optimizer control IR 接入 native
  ReLU-split BaB queue 的逐节点求界。

## Changes

- 使用 DocOps standalone plan/changelog 冻结 parent initialization-only、selected-state native
  re-execution、correctness-only 与独立 NRIR-9 rollback 边界。
- 新增 optimized queue evaluation/stack/trace 与非序列化 execution result；逐节点绑定 parent/selected
  state、warm kind、optimizer IR/trace hash、native compiler hash、gradient 与 re-execution diff。
- 实现 parent per-node selected states → batch-layout warm state 的 scope 重建；NRIR-10 classifier 必须
  判为 monotonic refinement，parent exact state 永不作为 child exact input。
- 每个 node batch 执行 NRIR-11 optimizer Schedule，selected batch state 随后进入 NRIR-10 native
  compiler；结果按 node slice 为独立 scoped state，供下一层 warm start。
- 新增 packed/serial state tensor comparator，区分稳定 scope fields 与 batch-layout-dependent
  intermediate hash；不以 exact state hash 掩盖数值布局差异。
- 新增 fixed ResNet artifact/replay 与 parent/stack/native/numeric/claim tamper tests。

## Validation

- toy complete queue：15 nodes，packed/serial 5/15 stacks；logical queue、bounds 与 selected state hash
  全部一致；child beta gradients 非零。
- fixed ResNet：7 nodes/3 expands/4 frontier，packed/serial 3/7 stacks；lower/upper max diff=
  `1.220703125e-04/1.8310546875e-04`；alpha/beta tensor max diff=
  `4.172325134277344e-07/7.450580596923828e-09`。
- packed alpha gradients L1=`180.2803225927055/338.43710628151894/630.9921717792749`；active child
  beta gradients L1=`20.047863006591797/42.26581954956055`；selected-state native diff 全为 0。
- artifact generate/replay：
  `e813826c8fe74161505ab2379b37fa67247fd40c3bd0cb8f82b77880ce403787`。
- 聚焦 `18 passed`；全量 `630 passed, 37 skipped, 7 warnings in 176.39s`；Black/Mypy clean、
  Pylint `10.00/10`、diff check 通过。

## Decisions

- 不把 parent 的 batch-scoped state 直接复用为 child exact state；先按 child batch layout 重组 parent
  split/alpha/beta 并重建 source scope，再由 NRIR-10 classifier 判定 monotonic refinement。
- 不修改 NRIR-9 plain-CROWN trace schema；NRIR-12 使用独立 optimized trace，避免历史 artifact hash
  漂移。

## Follow-Ups

- 完成 toy 与 fixed ResNet artifact 后，才判断是否进入 complete termination/property verdict；本阶段
  不启动性能路线。
- 下一阶段实现三态 termination/verdict：verified 必须 frontier 闭合且所有 leaves sound-pruned；
  unsafe 必须有 concrete witness；任何 node/depth/timeout budget 均为 unknown。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_OPTIMIZED_RELU_SPLIT_BAB_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
