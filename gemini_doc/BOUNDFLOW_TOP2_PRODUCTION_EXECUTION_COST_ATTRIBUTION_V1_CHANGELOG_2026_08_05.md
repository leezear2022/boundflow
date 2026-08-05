---
status: validated-reduced
updated: 2026-08-05T13:03:00Z
type: changelog
topic: boundflow
slug: top2-production-execution-cost-attribution-v1
stage: s01
---

# BoundFlow Top-2 Production Execution Cost Attribution v1 Changelog

## Summary

- NRIR-48 additive attribution runner、three-repeat artifact、replay/tamper 与路线判定已完成；
  `child_refinement_execute` 跨 clause/repeat 稳定 dominant，内部唯一合格子类为 `selected_crown`。

## Changes

- 基线冻结为 `main@1e44949` 的 NRIR-45 default production route；NRIR-47 candidate 保持禁用；
- 冻结七个互斥顶层类别和 child-refinement-execute 内部五个诊断子类；
- 冻结 three-fresh-process paired control/profile、exact semantics、`<=1%` closure 与 `<=1.05`
  instrumentation perturbation 门禁；
- 冻结 dominant category 的跨 clause/repeat share、稳定性与 pooled-MAD 门禁。
- 新增 additive formal runner 与 4 条 attribution/closure/tamper/scope contract tests；
- artifact：`artifacts/top2-production-execution-cost-attribution/`
  `vnncomp21-resnet2b-property0-three-repeat-cpu-phase0-v1/`。

## Validation

- profile/control median ratio=`1.023199/1.020221`，semantic exact 与 instrumentation gate 通过；
- child execute median/share=`3.816002 s/32.1966%`、`3.704755 s/31.1640%`，两条 3/3 排第一；
- selected-CROWN median/share of child execute=`2.663321 s/71.7725%`、
  `2.694436 s/72.7291%`，为唯一过 `>=30%` 的子类；
- formal hash=`571c2e47c0c8906d2486e5e19e8152eb1ef0d3024b08cf561e25ed4f71d177a4`；
  6 profile rows replay 与 synchronized category tamper 拒绝通过。
- focused `4 passed`；全量 `996 passed, 37 skipped`；Black、mypy、Pylint `10.00/10` 通过。

## Decisions

- 本轮只做 attribution，未实现优化；
- 顶层类别必须互斥，内部 inclusive 时间不得重复计入 total；
- 若 residual dominant，先增加计时点；若 child execute dominant 但内部无 `>=30%` 子类，继续细分；
- 预注册 dominance/stability 门禁已过，NRIR-49 唯一路线为 selected-CROWN execution。

## Follow-Ups

- 关闭并发布 NRIR-48；
- 另立 NRIR-49 预注册，先比较 selected-CROWN 的 shape/chunk/backend/call decomposition，再冻结唯一
  execution 优化；不得把 attribution 结果直接写成 speedup。

## Links

- plan: `gemini_doc/BOUNDFLOW_TOP2_PRODUCTION_EXECUTION_COST_ATTRIBUTION_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
