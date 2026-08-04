---
status: completed
updated: 2026-08-04T18:05:00Z
type: changelog
topic: boundflow
slug: sibling-packed-objective-ancestral-evaluator-v1
stage: s01
---

# Sibling-Packed Objective-Ancestral Evaluator v1 Changelog

## Summary

- NRIR-34 启动。NRIR-33 five-cap 证明 child target cap 不影响 7-node coverage；本轮先分解真实
  sibling phase，再把同 parent 两个 child 的 optimizer/native evaluator 合并为一个 batch。

## Changes

- 用 DocOps 冻结 first-pair serial→packed profiling、semantics gate、formal repeats 与 full-query
  upgrade boundary。
- 新增 `objective_ancestral_sibling_pack.py`，把 source/evaluator objective projection、same-parent
  `(-1,+1)` group、child refinement lineage、packed optimizer/native identity 与 atomic commit 编译为
  一等 Plan/Task/Schedule/Group IR。
- 新增 sibling-packed queue runtime：每个 child 仍独立执行 cap128 ancestral refinement；同一 parent
  的两个 child 只共享 optimizer/native compiler execution；deadline-crossing complete pair 整体丢弃。
- complete-query adapter 独立成 additive 模块，保留 ascending original ordinal、search、sound verdict、
  unresolved/pending accounting 与 60 秒全局 cooperative deadline。
- 新增 first-pair profiler、三 fresh-repeat formal runner、9-clause integration runner，以及正向、依赖、
  projection、group-commit、deadline、artifact tamper tests。

## Validation

- feasibility artifact hash：`7bece7f04459df37dad115622fe3bab5bc16145a4b82190ab003950317117ce9`；
  serial/packed child=`13.291550/7.018038 s`，bounds exact，optimizer/native group=`2→1`。
- formal 三轮全部 PASS：serial=`[7,7,7]`、packed=`[15,15,15]`，minimum node gain=`+8`；
  formal hash=`9678f9624abd547b76326ad2a1b916c3944d14fc96b2fbe0e81cf61849a777b4`。
- common 7 nodes lower/upper max diff=`7.62939453125e-06`；split、branch、final refinement bounds exact；
  alpha/beta max diff=`1.0728836e-04/8.9406967e-08`。
- full-query integration：sound `unknown`，completed clause `[0]`，unresolved `[0]`，pending `[1..8]`；
  evidence hash=`dcd0dc89fa7e4eb503e8a8b29438e16d215da10e66cd045cc76eb19a30037bf5`。
- profile/formal/full-query 与 frozen NRIR-32/33 replay 全部 PASS；focused `11 passed`；全量
  `862 passed, 37 skipped, 7 warnings`。最终 style/DocOps 门禁在发布前执行。

## Decisions

- 优先复用 `native_optimized_relu_split_bab_runtime._evaluate_optimized_node_batch` 已有 packed
  evaluator；NRIR-34 只新增 typed native-root admission、sibling-group ownership 和 deadline commit。
- 不把 native root 伪装成 external seed。
- 允许正式主张同算法、单 hard clause、cooperative deadline 下的 committed-node coverage improvement；
  不把约 `64.5—66.2 s` 的原子组完成 wall time 说成 60 秒硬实时或 wall-clock speedup。
- 9-clause 回接暴露的是 query-level budget monopolization：第一条 hard clause 消耗全局预算。NRIR-35
  应做跨 clause shared root/parametric evaluator + anytime allocation，禁止给每 clause 各配 60 秒。

## Follow-Ups

- NRIR-35：先 profile 九条 root/source/optimizer 的共享边界，再实现 original-ordinal-safe cross-clause
  objective batching 与 anytime budget；门禁必须在同一 60 秒全局 deadline 内严格增加 completed
  original clauses，且保持 clause 0 的 sound accepted frontier。

## Links

- plan: `gemini_doc/BOUNDFLOW_SIBLING_PACKED_OBJECTIVE_ANCESTRAL_EVALUATOR_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
