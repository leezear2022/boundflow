---
status: closed
updated: 2026-08-05T03:15:00Z
type: plan
topic: boundflow
slug: BOUNDFLOW_MULTI_CLAUSE_ANYTIME_PRIORITY_V1
stage: s01
---

# Multi-Clause Anytime Priority v1 Plan

## Goal

- 在固定 ResNet2B property 0 的同一 60 秒 global cooperative deadline 内，先保留 NRIR-31 九子句
  sound floor，再按可审计 priority 将剩余预算分配给至少两个 unresolved original clauses；禁止
  clause 0 继续独占余量，也禁止每条 clause 获得新的 60 秒。

## Scope

- frozen predecessors：`main@3755667`、NRIR-31 objective-hard-clause floor、NRIR-34 sibling-packed
  queue 与 NRIR-35 monotone original-ordinal aggregate；不得修改其代码或使 artifact replay 失效。
- priority v1 只使用 floor 已产生的 sound root lower：按 lower 降序（更接近 threshold 优先）、
  ordinal 升序打破平局；固定选 top-2 accepted unresolved clauses。
- slice v1 在每次 dispatch 前，把当前真实 remaining global budget 等分给仍待执行的 selected
  clauses。slice cutoff 通过 NRIR-36 私有 one-shot clock adapter 向 frozen packed queue 报告 global
  expiry；执行结束后恢复真实 clock。已完成 sibling group 才能进入证据。
- 每条 selected clause 必须消费 exact NRIR-31 refinement execution。多条 packed verdict 只能单调
  升级 floor；unknown/timeout 不得删除任何 floor verified/unresolved/unsafe 事实。
- 本轮是 multi-clause allocation/control gate。property 未闭合时，nodes/timing 只作机制证据，
  `performance_claimed=false`。

## Tasks

1. [x] 审计 NRIR-35 三轮 timing 与 NRIR-31 九条 root lower，冻结 priority/slice policy。
2. [x] 运行 top-2 等分余量 feasibility；要求两个 clause 都至少提交 root + 一个 atomic sibling pair。
3. [x] 新增 MultiClause Policy/Decision/Task/Schedule/Aggregate IR，hash-bind ranked candidates、selected
   ordinals、slice allocation、exact source lineage、packed results 与 single global deadline。
4. [x] 实现 native runtime；不得修改 NRIR-31/34 predecessor 文件，并验证其 frozen replay。
5. [x] 添加 wrong rank/selection/source、deadline reset、slice inflation、ordinal omission、non-monotone
   aggregate、partial group 与顶层 trace binding 的负向测试。
6. [x] 单次 first-class pilot 通过后运行三 fresh repeats；只在三轮均覆盖 selected top-2、保持 9-clause
   floor 且 replay/tamper/full suite 全过时关闭。

## Validation

- 审计结果：NRIR-31 root lower priority 为 `[2,3,4,5,0,8,6,7,1]`，对应前两条
  `-139.107880/-152.287033`；历史 clause 0 为 `-204.173157`，不应继续硬编码为第一优先级。
- feasibility：floor=`21.372808 s`；clause 2 获得 `18.313683 s` slice 并提交 3 nodes/1 group，
  worst active lower=`-95.557861`；clause 3 获得 `15.095226 s` 并提交 3 nodes/1 group，worst
  active lower=`-100.999161`。两条都在同一 global start 下执行，global actual=`66.256234 s`。
- first-class validator 必须从 floor execution 独立重算 rank、source、allocation 与 aggregate，不能
  信任 artifact 自报布尔值。
- NRIR-31、NRIR-34 formal/full-query、NRIR-35 formal artifact 均必须继续 replay。
- 正式三 fresh repeats 均重算 priority=`[2,3,4,5,0,8,6,7,1]`、selected=`[2,3]`。floor
  elapsed=`[21.637124,21.604930,21.871310] s`，whole cooperative elapsed=
  `[67.213556,66.833706,60.228863] s`；packed nodes 分别为 `[[3,3],[3,3],[3,1]]`。repeat 2 的
  clause 3 只提交 root、未提交 atomic sibling pair，故三轮 acceptance gate 失败；final 均 9/9 unresolved。
- formal replay hash=`2a2081af4c38de3df7a23c62cfcecfeb74d4b15132390a069e04a28bb65bfbf0`；
  16 个 focused/tamper tests、NRIR-31/34/35 predecessor replay、Black/mypy/Pylint `10.00/10` 通过；
  最终全量 `890 passed, 37 skipped`（7 条既有 deprecation/NVML warnings）。
- 关闭判定为 multi-clause allocation acceptance `VALIDATED-NO-GO`：IR/control 可保留，但 equal-remaining
  slice 未稳定保证两条 clause 各提交一个 atomic pair。下一门禁必须转向 shared parametric
  compiler/root/evaluator 与更强 bound/candidate 信息，不得以调 top-k 或 slice 常数推翻 NO-GO。

## Rollback

- 删除 additive NRIR-36 新文件即可回到 `main@3755667`；不改 predecessor IR/runtime/artifact。

## Links

- changelog: `gemini_doc/BOUNDFLOW_MULTI_CLAUSE_ANYTIME_PRIORITY_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_CROSS_CLAUSE_ANYTIME_OBJECTIVE_EVALUATOR_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
