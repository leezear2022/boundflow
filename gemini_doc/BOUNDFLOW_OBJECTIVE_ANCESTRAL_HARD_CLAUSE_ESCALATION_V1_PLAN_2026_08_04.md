---
status: completed
updated: 2026-08-04T14:55:45Z
type: plan
topic: boundflow
slug: objective-ancestral-hard-clause-escalation-v1
stage: s01
---

# Objective-Ancestral Hard-Clause Escalation v1 Plan

## Goal

- 以 NRIR-31 validated per-clause objective refinement execution 作为 typed root source，再把每个
  dynamic child refinement 严格绑定到 parent execution，判断巨大 ResNet root tightening 能否继续
  改善第一层 child/frontier；不得把 native root execution 伪装为 external seed。
- feasibility probe 通过后才实现完整 PlanTemplate/PlanInstance、queue runtime 与 31/depth4 artifact；
  失败则以 `VALIDATED-NO-GO` 收口，不改 frozen NRIR-31/30 文件。

## Scope

- 首个 pilot 固定 VNN-COMP 2021 ResNet2B property 0 clause 0、CPU threads=8、5-step optimizer，
  shared top-width 128/32 单 pass + objective 128/32 单 pass，复用 exact NRIR-31 root execution。
- 冻结 root 选择出的同一 branch 和两个 depth-1 child。root-global 对照与 ancestral child 均 serial
  evaluation，隔离 batching；前者对两个 child 重用 root bounds，后者各编译一份 split-exact、
  objective-directed refinement，并以 root execution 作 `source_refinement_execution`。
- pilot 只验证 first-child tightness/source lineage，不声明完整 queue、property、performance、GPU、
  competitor 或 ASPLOS-ready。

## Tasks

1. [x] 实现只读 feasibility runner，保存 shared/objective root Plan/semantic trace、exact branch、两个
   child split/source lineage、root-global 与 ancestral lower/upper。
2. [x] 冻结 pilot gate：root exact；两个 common child lower 均不退化（tol `1e-5`）；worst-child
   lower 严格改善 `>1e-4`。任一 soundness/lineage gate 失败即 NO-GO。
3. [x] 若 pilot 通过，新增 additive objective-ancestral queue IR/runtime，动态 node 必须拥有
   refinement TaskInstance/ScheduleInstance 和 parent Plan/semantic/final-bound hash。
4. [x] 运行固定 ResNet clause 0、`31/depth4`、三 fresh repeats 正式门禁；因 whole deadline 只
   接受 7 个节点，本轮 claim 限定为 typed lineage 与 committed-frontier tightness，不扩展到多 clause/
   多拓扑或 property closure。

## Validation

- root source 必须是 `NativeIntermediateRefinementExecution`，child Plan 的 source constraint/Plan/
  semantic trace 三哈希必须逐项等于 parent final execution；child split hash 与 branch ordinal exact。
- 对照与 ancestral 使用相同 module/input/objective/threshold/optimizer/config/root selected state 和
  branch；只改变 child intermediate-bound source。
- pilot timing 仅作诊断，`performance_claimed=false`。
- 正式三轮的 committed queue trace、Task IR 与 node-refinement 哈希必须分别唯一；deadline 后已计算
  但未 commit 的诊断 stage 不进入 proof identity。

## Result

- two-child pilot 的 ancestral child lower 为 `-142.703659/-142.854645`，相对同 child 的
  root-global lower 改善 `+59.367462/+59.253479`，预注册 gate 通过。
- 正式三轮均提交 7 nodes、24 个 Task/Schedule actions、depth 2；root lower 与 31-node root-global
  对照 exact parity=`-204.17315673828125`。
- ancestral worst active lower=`-104.76541137695312`，root-global worst active lower=
  `-200.46539306640625`，三轮 delta 均为 `+95.69998168945312`。
- cooperative deadline 在一次 late child evaluation 后丢弃未提交工作并保留 accepted frontier；无
  property closure、performance、GPU、competitor、multi-clause 或 ASPLOS-ready claim。
- 结论：objective-ancestral committed-frontier tightness `VALIDATED-REDUCED`。下一门禁为预注册的
  child refinement budget/cap Pareto，使 tighter frontier 在 60 秒内转化为更多 committed nodes，
  不直接放宽 deadline。

## Rollback

- feasibility runner/artifact 为 additive；删除本轮新增文件即可回到 `main@cb94ac6`。不得修改或
  重签 NRIR-31/30 frozen artifact。

## Links

- changelog: `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
