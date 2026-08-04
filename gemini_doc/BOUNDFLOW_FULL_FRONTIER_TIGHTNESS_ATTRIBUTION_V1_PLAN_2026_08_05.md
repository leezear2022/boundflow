---
status: completed
updated: 2026-08-04T20:05:00Z
type: plan
topic: boundflow
slug: BOUNDFLOW_FULL_FRONTIER_TIGHTNESS_ATTRIBUTION_V1
stage: s01
---

# Full Frontier Tightness Attribution v1 Plan

## Goal

- 在不改变 NRIR-37 的 floor、priority、top-2、slice、cap128、31/depth4、branch 或 cache 语义的前提下，
  对 clauses 2/3 的完整 depth-4 active frontier 建立 first-class Plan/Task/Schedule 归因证据。
- 只检验一个预注册 stronger-bound 变量：native alpha/beta optimizer `steps=5→15`。先在 exact frozen
  frontier 上做反事实重放；不在看到结果后改候选、阈值或 frontier。

## Scope

- 基线：`main@813006b` / NRIR-37 formal hash
  `9234dcbe77803e0e7d7e62ca88c62e1b859c95af4ad8e3a19b85c0ab87294b83`。
- workload 固定为 VNN-COMP 2021 `cifar10_resnet:000` property 0，selected original clauses 固定
  `[2,3]`；只使用 CPU、8 torch threads，`performance_claimed=false`。
- attribution 覆盖 source queue 的全部 31 evaluations、全部 active/terminal frontier、parent path、
  per-node refinement pass、selected alpha/beta state；active frontier 不能由报告方手选。
- counterfactual 保持 objective、threshold、split state、parent warm tensor、ancestral refinement、
  sibling grouping 与 tensor dtype/device 不变；baseline `steps=5` 与 candidate `steps=15` 各用独立
  query-owned parametric template cache。
- 不修改 frozen NRIR-31/34/36/37 文件，不形成 property closure、wall-clock speedup、GPU、competitor、
  multi-workload 或 ASPLOS-ready claim。

## Tasks

- [x] 新增 `FrontierTightnessAttribution` Plan/Task/Schedule IR：source hash、metric contract、active node
  IDs、baseline/candidate policy、sibling batches、decision 与 emit 进入 canonical hash。
- [x] 新增 runtime validator：从 source execution 独立重算 active frontier、depth/path/refinement/state
  统计；拒绝节点遗漏、伪造 parent、非 active 节点注入、policy 漂移和 task/schedule 脱钩。
- [x] 逐 exact sibling pair 重放 `steps=5` 与 `steps=15`，baseline 必须复现 source lower/upper 与
  refinement final-bound hash，candidate 只能改变 optimizer policy/selected state/bounds。
- [x] 生成单次真实 pilot artifact 和 replay/tamper tests；依据下述冻结门禁给出 GO/NO-GO。
- [x] 若 GO，下一阶段只允许把 steps15 接入同一 full queue 做三 fresh repeats；若 NO-GO，冻结该轴并
  转向 branch-bound candidate，不在本阶段补试其他 step 数。

## Validation

- source gate：两条 clause 均 `31 evaluations`，active frontier 恰为 `16` 个 depth-4 nodes，node/
  refinement/state 一一对应，source runtime validators 通过。
- replay gate：baseline frontier lower/upper 按 frozen allclose
  `atol=1e-5,rtol=1e-5` 复现，split、parent、sibling grouping、refinement final-bound hashes exact。
- candidate GO gate（两条 clause 必须同时满足）：
  1. 16/16 candidate lower delta `>=-1e-5`；
  2. candidate worst-active lower 相对 baseline 至少改善 `+1.0`；
  3. 至少 12/16 active nodes 严格改善 `>1e-5`；
  4. 所有 candidate bounds finite 且 `lower<=upper`。
- 任一 source/replay/gate 条件失败即 `VALIDATED-NO-GO`；全部通过才是 fixed-frontier tightness
  `VALIDATED-REDUCED`，仍不得升级为完整 query 或性能结论。
- targeted pytest、artifact replay/tamper、mypy、Pylint、全量 `pytest tests`、`dol validate` 与
  `dol lint --soft`。

实际结果：两条 clause source 均为 31 evaluations / 16 active depth-4 nodes；baseline replay lower/
upper max diff 均为 0，refinement hashes exact。steps15 对 32/32 nodes 严格改善且无退化，但 clauses
2/3 worst-active lower 仅改善 `+0.055496/+0.028557`，远低于 `+1.0` 门禁，因此按预注册规则
`VALIDATED-NO-GO`。pilot hash=
`2719347a8e1c5c49c418b3a396ff405a004b0f4ace96af94d335e4026f7a24a2`。

## Rollback

- 删除本分支 additive NRIR-38 IR/runtime/script/tests/artifact/docs 即回到 `main@813006b`；frozen
  predecessor source 与 artifact 不变。

## Links

- changelog: `gemini_doc/BOUNDFLOW_FULL_FRONTIER_TIGHTNESS_ATTRIBUTION_V1_CHANGELOG_2026_08_05.md`
- predecessor: `gemini_doc/BOUNDFLOW_SHARED_PARAMETRIC_OBJECTIVE_EVALUATOR_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
