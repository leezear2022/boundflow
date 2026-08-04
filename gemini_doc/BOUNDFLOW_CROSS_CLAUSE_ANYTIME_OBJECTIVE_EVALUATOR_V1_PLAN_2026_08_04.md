---
status: complete
updated: 2026-08-04T17:11:07Z
type: plan
topic: boundflow
slug: BOUNDFLOW_CROSS_CLAUSE_ANYTIME_OBJECTIVE_EVALUATOR_V1
stage: s01
---

# Cross-Clause Anytime Objective Evaluator v1 Plan

## Goal

- 在 ResNet property 0 的同一 60 秒全局 cooperative deadline 内，先保留 NRIR-31 已验证的 9-clause
  original-ordinal baseline，再把剩余预算用于 cap128 sibling-packed objective-ancestral escalation；
  禁止再次出现“clause 0 有深树、clauses 1..8 未执行”的 query-level budget monopolization。

## Scope

- 固定 workload、search/optimizer policy、NRIR-31 objective-hard-clause Program 与 NRIR-34
  sibling-packed queue，不改算法参数、node/depth 或 deadline。
- v1 只 admit original clause 0 进入 anytime queue；其 root source 必须是 NRIR-31 clause 0 已接受的
  objective refinement execution，不能重建相似但未绑定的 source。
- global order 固定：compile all→execute NRIR-31 baseline/objective stage→Decision→optional packed
  clause 0→original-ordinal aggregate→emit。NRIR-31 final verdict 是 sound floor，packed 结果只能升级，
  不能删除或降级其他 clause。
- 本阶段是 cross-clause control/integration gate；没有 property closure 前不形成 performance claim。

## Tasks

1. [x] 运行 additive feasibility：同一进程先执行 frozen NRIR-31，再把 wrapper 起始时间传给 NRIR-34
   clause-0 queue；记录 baseline elapsed、remaining budget、packed accepted nodes 与 root parity。
2. [x] feasibility 通过后新增 Anytime Plan/Decision/Task/Schedule：hash-bind NRIR-31 program/execution、
   admitted ordinal、root refinement lineage、NRIR-34 plan 与 global deadline。
3. [x] 实现 monotone aggregate：9 条 baseline original ordinals 必须始终有 sound status；deadline before/
   during packed queue 时保留 baseline，complete atomic sibling groups 才可进入 additive evidence。
4. [x] 固定 artifact replay 与 tamper tests：wrong ordinal、wrong source execution、deadline reset、baseline
   omission、non-monotone aggregate、partial group 均 fail closed。
5. [x] 单次 pilot gate：baseline completed ordinals 必须为 `[0..8]`；packed root lower 与 NRIR-31 clause 0
   exact/tolerance parity；至少提交一个 complete sibling group；query final status/verified set 不回退。
6. [x] pilot 通过后运行三 fresh repeats；只报告 all-clause preservation、additive node/depth/tightness 和
   cooperative wall time。若仍无 property closure，下一路线转向 multi-clause time slicing/priority，而不是
   再把所有剩余预算给 clause 0。

## Validation

- baseline 的 decision、shared refinement、9 个 objective child trace 与 final aggregate 必须通过原
  NRIR-31 validator；不复制或弱化其 soundness contract。
- clause-0 packed plan 的 source Plan/semantic/final-bound hash 必须等于 exact accepted NRIR-31 child；
  source `(1,1,10)`→evaluator `(1,10)` projection显式保留。
- global start 只能创建一次；NRIR-34 queue 必须消费相同 `whole_query_started_ns`，不得获得新的 60 秒。
- final original ordinal accounting 必须覆盖 9 条；`verified_after ⊇ verified_before`，unsafe witness 不得
  被 packed unknown 覆盖。

## Rollback

- additive 新文件可删除回到 `main@796a64e`；NRIR-31/32/33/34 frozen replay 必须继续 PASS。

## Feasibility Result

- NRIR-31 floor elapsed=`22.180302879 s`，completed objective ordinals=`[0..8]`，final unresolved=
  `[0..8]`，fallback=`none`。
- packed stage 从同一 global start 的 `24.510176706 s` 处进入，exact 绑定 accepted clause-0
  refinement；packed/root lower diff=`0.0`。
- 剩余预算内提交 7 nodes/3 complete sibling groups；mechanism gate PASS，evidence hash=
  `244eac460a34f2736aab70a61b8271fab92577980e78e9465517f539b8894825`。
- 该结果只授权任务 2—4 的正式 IR/runtime，不形成 property 或 performance claim。

## Formal Result

- 一等 cross-clause Plan/Decision/6-stage Task/Schedule 已落地；NRIR-31 floor executor 新增可选
  `whole_query_started_ns`，NRIR-35 与 packed queue 消费同一 global start。Decision 绑定 exact
  accepted clause-0 refinement Plan/semantic/final-bound hash；Aggregate 保留全部 9 个 original
  ordinals，packed unknown 不能改写 floor。
- 三个 fresh process 的 floor elapsed 为 `22.227251/21.622773/21.834220 s`，均 completed
  `[0..8]`、unresolved `[0..8]`；余量内 packed accepted nodes=`[7,7,9]`，每轮均为完整 atomic
  sibling pairs。
- cooperative whole elapsed=`61.991720/62.598928/68.042604 s`；超过 60 秒来自已开始 sibling
  group 的原子完成，不是硬实时或 wall-clock speedup。三轮 final 仍为 sound `unknown`，故
  `performance_claimed=false`。
- formal hash=`74533c9c211a3007bf5af43c08865febd95c3f9ccf1a268e56738793ec9d14d5`；artifact replay、六类
  同步重哈希篡改、关联 29 tests、全量 `874 passed, 37 skipped` 与静态门禁均通过。
- 本阶段以 cross-clause control/original-ordinal preservation `VALIDATED-REDUCED` 关闭；下一分支
  `feat/multi-clause-anytime-priority-v1`，在同一 60 秒 global budget 内把 additive work 分配给多个
  unresolved clauses，而不是继续独占 clause 0。

## Links

- changelog: `gemini_doc/BOUNDFLOW_CROSS_CLAUSE_ANYTIME_OBJECTIVE_EVALUATOR_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_SIBLING_PACKED_OBJECTIVE_ANCESTRAL_EVALUATOR_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
