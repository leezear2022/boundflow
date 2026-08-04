---
status: completed
updated: 2026-08-04T20:05:00Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_FULL_FRONTIER_TIGHTNESS_ATTRIBUTION_V1
stage: s01
---

# Full Frontier Tightness Attribution v1 Changelog

## Summary

- NRIR-38 启动：NRIR-37 已以 shared compiler ownership + fixed-deadline coverage
  `VALIDATED-REDUCED` 合并；final 仍 9/9 unresolved，下一变量从 control/coverage 转为 bound tightness。

## Changes

- 合并并同步 PR #48，建立 `feat/full-frontier-tightness-attribution-v1`。
- 预注册 exact depth-4 frontier attribution 与唯一候选 `optimizer steps 5→15`；冻结 source、重放与
  GO/NO-GO 门禁。
- 新增 `FrontierTightnessAttribution` Plan、七阶段 Task/Schedule、31-node depth/path/refinement/
  alpha-beta state 归因，以及按原八个 sibling pair 的 baseline/candidate 双 cache 反事实 runtime。
- 真实 clauses 2/3 各覆盖 31 evaluations、16 active depth-4 nodes；baseline replay lower/upper max
  diff=0，split/refinement exact。steps15 改善 32/32 nodes、0 regressions，但 worst-active lower 只提高
  `+0.055496/+0.028557`，未达到 `+1.0` 门禁，结果为 `VALIDATED-NO-GO`。

## Validation

- artifact generate/replay 通过；8 类 active/policy/candidate/decision/Task/cache/sibling/evidence tamper
  fail closed；13 focused tests、全量 `930 passed, 37 skipped`、mypy clean、Pylint `10.00/10`。
- pilot hash=`2719347a8e1c5c49c418b3a396ff405a004b0f4ace96af94d335e4026f7a24a2`。

## Decisions

- cap8—128 已在 NRIR-33 全部无 coverage 收益，typed two-pass 已在 NRIR-26 NO-GO，因此本阶段不再
  重试 cap/multipass；选择 optimizer steps 作为唯一 stronger-bound 变量。
- 先在 exact source frontier 做反事实，避免 full-query control/coverage 差异污染 tightness 因果归因。
- depth-4 alpha interior fraction 只有 clauses 2/3 的 `2.164%/2.518%`，且 steps15 的 median lower
  delta 虽为 `+0.107208/+0.132715`，worst gain 极小；继续加 optimizer steps 缺乏量级依据。

## Follow-Ups

- optimizer-step 轴已冻结；下一阶段只接入仓库已有 objective branch Plan/Task/Schedule，做 widest vs
  objective-bound-impact 的 exact fixed-tree 单变量对照。

## Links

- plan: `gemini_doc/BOUNDFLOW_FULL_FRONTIER_TIGHTNESS_ATTRIBUTION_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
