---
status: completed
updated: 2026-08-04T13:34:00Z
type: changelog
topic: boundflow
slug: typed-hard-clause-escalation-v1
stage: s01
---

# Typed Hard-Clause Escalation v1 Changelog

## Summary

- NRIR-30 已启动。NRIR-29 证明 MNIST 在 31 nodes 已饱和、OVAL 的最后 clause 和 ResNet 全部
  clauses 对纯节点扩展无响应；本轮把 stronger native bounds 作为 unresolved-clause guarded stage
  纳入 compiler/runtime，而不是继续单轴增加同类节点。

## Changes

- 新建 standalone plan/changelog；冻结 baseline `7/2 local-forward`、shared native refinement
  `1 pass × 128 targets/ReLU × chunk32`、unresolved-only escalation `31/4` 和 60 秒 whole-query
  deadline。
- 新增 `hard_clause_escalation.py` IR：exact query Plan、baseline-derived Decision、8-task guarded
  TaskModule 和 sequential Schedule；baseline/refinement/projection/escalation/aggregate/emit 均有
  一等 ownership 和 hash。
- 新增 runtime：使用同一 monotonic whole deadline，baseline verified/unsafe 不重跑；hard clauses
  双射投影后消费 `native_refined` bounds，aggregate 恢复 original ordinals。deadline 后 proof 被
  丢弃并保留 baseline，不允许同步修改 digest 后升级 verdict。
- 新增 9-worker artifact runner 和 7 个测试；runner 绑定 NRIR-29 predecessor hash、重放 real
  source→program IR、compiler template/cache/instance、refinement trace、control/actions 与 final verdict。

## Validation

- 历史 NRIR-19 只作路线先验：同类 native refinement 曾把 OVAL21 从 unknown 变 verified，并将
  MNIST unresolved 从 3 个降为 1 个；它使用 audit query 且不是本轮证据。本轮必须重新通过
  parametric production path、hard-clause projection、三 fresh repeats 与 artifact replay。
- 正式结果三次完全一致：MNIST baseline `[0,1,2,4,5,6]`、admit `[3,7,8]`、final `[0..7]`；
  ResNet baseline `[]`、admit `[0..8]`、final `[]`；OVAL baseline `[0..7]`、admit `[8]`、final
  `[0..8]` 且 query status=`verified`。9/9 都 `fallback=none`。
- median/p90 whole-stage execution：MNIST `2.974/3.003 s`、ResNet `20.146/20.223 s`、OVAL
  `2.208/2.213 s`；这些值只用于 deadline accounting，`performance_claimed=false`。
- artifact fresh replay、NRIR-28/29 historical replay、focused `14 passed`、Black、targeted Mypy、
  Pylint `10.00/10` 与 diff gate 通过；evidence hash=
  `df096e70d6126d585132e14dc9796038855b37bf4d9ef76528b9feb6a1330205`；全量
  `830 passed, 37 skipped`。

## Decisions

- refinement cap 统一固定 128/32，不沿用 NRIR-19 按 workload 的 128/16/128 配置，避免在看到
  NRIR-30 结果后对 ResNet 单独调参。
- primary gate 是完整 workload closure；timing 只用于 whole-deadline accounting，不计算 stage
  speedup。
- OVAL 完整 closure 三次复现，且所有 workload 无 baseline verdict 回退，因此以 property-coverage
  `VALIDATED-REDUCED` 关闭。ResNet 0/9 明确保留为 shared top-width refinement 的 hard boundary。

## Follow-Ups

- 下一分支只替换 hard-clause refinement selection：每个 projected scalar objective 编译
  objective-influence Plan/Task/Schedule；保持 baseline/admission/31-node/deadline/aggregate 不变，
  判断 MNIST clause 8 或 ResNet root/closure 是否严格改善。

## Links

- plan: `gemini_doc/BOUNDFLOW_TYPED_HARD_CLAUSE_ESCALATION_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
