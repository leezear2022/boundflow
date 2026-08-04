---
status: closed
updated: 2026-08-05T03:15:00Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_MULTI_CLAUSE_ANYTIME_PRIORITY_V1
stage: s01
---

# Multi-Clause Anytime Priority v1 Changelog

## Summary

- NRIR-36 启动。NRIR-35 已恢复 9/9 floor accounting，但把全部余量交给历史固定 clause 0；本轮
  改为由 floor root evidence 决定优先级，并在相同 global deadline 内覆盖多个 unresolved clauses。

## Changes

- 用 DocOps 冻结 root-lower priority、top-2 selection、dynamic equal-remaining slice、exact source、
  one-shot cutoff 与 monotone aggregate 门禁。
- 执行一次不改代码的真实 feasibility，确认 frozen NRIR-31/34 可通过私有 clock adapter 组合为
  多 clause cooperative execution。
- 新增 `multi_clause_anytime` Policy/Plan/Candidate/Decision/8-task Task/Schedule/Slice/Outcome/Aggregate
  IR；priority、top-2、dynamic equal-remaining allocation、exact source、packed result 与 original
  ordinal aggregate 均进入 canonical hash。
- 新增 native runtime：先消费 exact NRIR-31 floor，再用 NRIR-36 私有 one-shot clock 将两个 slice
  截断映射到 frozen NRIR-34 的 cooperative deadline；未完成 sibling group 不进入 proof。
- 新增 formal generate/replay runner、三 fresh-process shards/manifest 与同步重哈希 tamper tests；未改
  NRIR-31/34/35 predecessor 文件。

## Validation

- priority 独立重算为 `[2,3,4,5,0,8,6,7,1]`；top-2 clauses 2/3 均提交 3 nodes/1 atomic group。
- feasibility whole actual=`66.256234 s`，两条 worst active lower 分别为
  `-95.557861/-100.999161`；final property 尚未闭合，不形成 performance claim。
- 正式三轮 floor elapsed=`[21.637124,21.604930,21.871310] s`，whole cooperative elapsed=
  `[67.213556,66.833706,60.228863] s`；selected 每轮均为 `[2,3]`，packed nodes=
  `[[3,3],[3,3],[3,1]]`。前两轮两条均有 atomic pair；repeat 2 clause 3 只提交 root，worst active
  lower 保留 floor 值 `-152.287033`，三轮 gate 因此失败。
- formal hash=`2a2081af4c38de3df7a23c62cfcecfeb74d4b15132390a069e04a28bb65bfbf0`；
  replay、16 focused/tamper tests、NRIR-31/34/35 predecessor replay、Black/mypy/Pylint `10.00/10`
  通过；最终全量 `890 passed, 37 skipped`（7 条既有 deprecation/NVML warnings）。

## Decisions

- v1 先验证跨 clause 资源分配，而不是同时改变 refinement cap、node/depth、optimizer 或 global
  deadline；这样可以隔离 scheduler 因果。
- 排序选择最接近 threshold 的 clauses 2/3，不沿用 clause 0 历史顺序。
- 三轮 final 都是 sound `unknown`、9/9 unresolved；预注册的“两条 selected clauses 每轮都至少提交
  一个 atomic pair”未成立，本阶段以 multi-clause allocation `VALIDATED-NO-GO` 关闭，
  `performance_claimed=false`。IR/control 可保留；不声明硬实时、speedup、property closure、GPU、
  competitor、multi-workload 或 ASPLOS-ready。

## Follow-Ups

- 下一门禁转向 shared parametric compiler/root/evaluator 与 stronger candidate/bound，先量化两个
  selected clause 的 compile/root/child phase，再冻结复用合同和 tightness gate；不继续只调 slice 常数。

## Links

- plan: `gemini_doc/BOUNDFLOW_MULTI_CLAUSE_ANYTIME_PRIORITY_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
