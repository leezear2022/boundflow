---
status: completed
updated: 2026-08-04T21:00:00Z
type: plan
topic: boundflow
slug: BOUNDFLOW_OBJECTIVE_BRANCH_SHARED_EVALUATOR_V1
stage: s01
---

# Objective Branch Shared Evaluator v1 Plan

## Goal

- 把仓库已有的 objective-bound-impact branch Plan/Task/Schedule 接入 NRIR-37 已验证的
  shared-parametric + ancestral-refinement evaluator，消除当前生产路径固定使用 widest branch 的集成缺口。
- 在 clauses 2/3 上做单变量、同预算、完整 31-node/depth-4 对照；只改变 branch candidate selection，
  不改变 floor、priority、optimizer、refinement、cache、queue budget 或 sibling atomic commit。

## Scope

- 基线：`main@20a8ac3`；NRIR-37 formal hash=
  `9234dcbe77803e0e7d7e62ca88c62e1b859c95af4ad8e3a19b85c0ab87294b83`，NRIR-38 pilot hash=
  `2719347a8e1c5c49c418b3a396ff405a004b0f4ace96af94d335e4026f7a24a2`。
- workload 固定为 VNN-COMP 2021 `cifar10_resnet:000` property 0，selected original clauses `[2,3]`；
  CPU、8 torch threads，`performance_claimed=false`。
- control 使用 NRIR-37 的 widest-unsplit-ambiguous-ReLU；candidate 冻结为历史 NRIR-17 policy：
  `top_width_per_relu_v1`、`candidates_per_relu=8`、`candidate_batch_size=64`、
  `max_candidates=256`、`maximize_worst_child_then_mean`。看到结果后不得改 policy 或 gate。
- 两侧保持 optimizer steps=5、cap128 ancestral refinement、parent warm state、shared template ownership、
  best-first queue、31 nodes/depth4、完整 sibling pair commit、dtype/device 与 source objective/threshold 不变。
- 不修改 frozen NRIR-31/34/36/37/38 实现或 artifact；只新增组合 IR/runtime/script/tests/artifact/docs。
- 不形成 property closure、wall-clock speedup、GPU、competitor、multi-workload 或 ASPLOS-ready claim。

## Tasks

- [x] 新增 `ObjectiveBranchSharedEvaluator` Plan/Task/Schedule：显式绑定 frozen shared plan、branch policy、
  每节点 objective-branch program/execution、shared batch commit、queue transition 与最终 emit。
- [x] 新增组合 runtime：复用 frozen shared batch evaluator与 objective branch 5-stage program；root/每个仍有候选
  节点必须恰有一次 branch execution，branch decision 必须与 selected candidate exact 对齐。
- [x] 正向/负向测试覆盖 policy/source/branch coverage/selected candidate/Task/Schedule/batch commit/lineage；
  control 必须逐字段复现 frozen NRIR-37 widest execution。
- [x] 对 clauses 2/3 生成真实 fixed-budget pilot artifact，并以独立重算和同步重哈希 tamper replay 验证。
- [x] 按冻结门禁给出 GO/NO-GO；只有 GO 才允许下一阶段做 three-repeat whole-query formal。

## Validation

- source/control gate：两条 clause 的 widest control 都必须为 31 evaluations、16 个 depth-4 active frontier；
  lower/upper、split、branch、refinement semantic hash、selected alpha/beta state 与 NRIR-37 source exact/
  frozen-allclose 一致。
- candidate structure gate：两条 clause 都必须提交 31 evaluations、16 个 depth-4 active frontier；每个发生
  expansion 的节点恰有一个合法 objective-branch execution，queue decision、child split 与 execution selected
  candidate exact 对齐；无 partial sibling、lineage 或 soundness 违规。
- candidate GO gate（两条 clause 必须同时满足）：
  1. objective worst-active lower 相对 widest 至少改善 `+1.0`；
  2. objective median active lower 不低于 widest median（容差 `1e-5`）；
  3. objective root bound 与 widest root bound按 `atol=1e-5,rtol=1e-5` 一致；
  4. 所有 bounds finite 且 `lower<=upper`。
- 任一结构/复现条件失败为 validation failure；结构成立但收益 gate 失败则
  `VALIDATED-NO-GO`；全部通过才是 fixed-budget branch selection `VALIDATED-REDUCED`。
- targeted pytest、artifact replay/tamper、mypy、Pylint、全量 `pytest tests`、`dol validate` 与
  `dol lint --soft`。

实际结果：两侧 root lower exact；widest clauses 2/3 worst-active lower=
`-37.574287/-35.900215`，objective candidate=`-35.530926/-30.258448`，改善
`+2.043362/+5.641768`；median active lower 改善 `+2.537640/+5.885233`。两条均为 31 evaluations、
16 个 depth-4 active nodes、31/31 branch executions，全部通过 `+1.0` 门禁，因此本阶段为 fixed-budget
branch selection `VALIDATED-REDUCED`。pilot hash=
`dde1cc4076ea766e7b4859e75ec9ff214d61f3cf245385285274b47f541a72cc`；下一阶段允许执行 three-repeat
whole-query formal，但本结果本身不是 wall-clock/property/ASPLOS-ready claim。16 focused、含 frozen
predecessors 的 40 tests、全量 `940 passed, 37 skipped`、mypy/Pylint 与 DocOps 静态门禁通过。

## Rollback

- 删除本分支 additive NRIR-39 文件即可回到 `main@20a8ac3`；frozen predecessor 文件和 artifact 不变。

## Links

- changelog: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SHARED_EVALUATOR_V1_CHANGELOG_2026_08_05.md`
- predecessor: `gemini_doc/BOUNDFLOW_FULL_FRONTIER_TIGHTNESS_ATTRIBUTION_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
