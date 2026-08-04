---
status: complete
updated: 2026-08-04T01:52:02Z
type: plan
topic: boundflow
slug: complete-verifier-query-v1
stage: s01
---

# Complete Verifier Query v1 Plan

## Goal

- 关闭 NRIR-13 的“单 clause + caller-supplied candidate”边界：实现多 clause 性质聚合、
  deterministic candidate search、unsafe short-circuit 和 query-level deadline/unknown，形成可直接
  调用的 complete verifier query control contract。

## Scope

- v1 property 为 conjunction：对每个 clause `C_i f(x) >= threshold_i`，全部 verified
  才返回 verified；任一可重放 counterexample 立即 unsafe；其余 unknown。
- candidate search 使用 deterministic box-projected gradient descent；search failure 只记录
  `not_found`，绝不升级为 proof。任何 found candidate 仍必须通过 NRIR-13 concrete replay。
- query deadline 在 clause/stage 边界 cooperative 检查；到期后未执行 clauses 显式 pending，
  整体 unknown。v1 不伪称可中断正在运行的 kernel。
- 保留 NRIR-12/13 artifact identity；CPU correctness/control only，不声明 speedup。

## Tasks

1. 使 concrete Task IR executor 可选保留 autograd，同一 primal 语义同时服务 attack
   search 和最终 no-grad witness replay。
2. 实现 typed PGD policy/search trace，记录 objective/input hash、iteration、best value、early stop
   和 found/not-found；box projection 和 finite gradient fail closed。
3. 实现 clause/query trace 与 executor，串联 search→optimized queue→NRIR-13 verdict；聚合
   verified/unsafe/unknown，实现 unsafe 短路与 timeout pending accounting。
4. toy 覆盖 multi-clause verified、search-found unsafe、unknown、deadline；固定 ResNet 9 clauses
   输出 verified/unknown clause 分解与总体 sound unknown。
5. 生成/replay artifact，增加 clause order、aggregation、candidate、deadline、short-circuit
   和 claim-inflation tamper probes。

## Acceptance Criteria

- verified query 必须 completed count = clause count 且每个 NRIR-13 verdict 都是 verified。
- unsafe query 必须绑定恰一个已重执行 witness，后续 clauses 标记 short-circuit skipped。
- unknown query 必须显式列出 unresolved/pending clauses；attack `not_found` 不能作为
  verified evidence。
- fixed ResNet 使用 9 个真实 objectives；已证 clauses 与未闭合 clauses 分开记录。
- focused/full pytest、artifact replay、Black、Mypy、Pylint、diff 与 DocOps lint 全过。

## Closure

- 2026-08-04 按 `VALIDATED-REDUCED` 关闭。multi-clause conjunction、deterministic
  candidate search、concrete witness replay、unsafe short-circuit 与 cooperative deadline
  均有代码、测试和冻结 artifact。
- 固定 ResNet 九个真实 clauses 全部完成管线执行，但 native scalarized lower bounds
  仍过松，正确结果为 `unknown`（unresolved 9/9），不是 real-property closure。
- artifact：`artifacts/complete-verifier-query/vnncomp21-resnet2b-prop0-cpu-v1/`，
  replay hash=`d17f7d7e960491ad7ef3f33bad41a4cfbf21a9fd5213df3637584b6a753968f1`。
- 验证：相关 `39 passed`；全量 `670 passed, 37 skipped`；Black、Mypy、Pylint
  与 replay 全过。

## Rollback

- 新 query/search runtime 独立于 NRIR-13；若聚合门禁失败，保留单 clause sound
  verdict `VALIDATED-REDUCED`，不修改已冻结 queue/verdict artifacts。

## Links

- predecessor: `gemini_doc/BOUNDFLOW_NATIVE_PROPERTY_TERMINATION_VERDICT_V1_PLAN_2026_08_04.md`
- changelog: `gemini_doc/BOUNDFLOW_COMPLETE_VERIFIER_QUERY_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
