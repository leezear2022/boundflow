---
status: completed-no-go
updated: 2026-08-04T15:14:00Z
type: plan
topic: boundflow
slug: objective-ancestral-child-budget-pareto-v1
stage: s01
---

# Objective-Ancestral Child Budget Pareto v1 Plan

## Goal

- 在不改变 NRIR-32 objective root source、branch/optimizer、31 nodes/depth 4、CPU threads=8 或
  60 秒 whole deadline 的前提下，降低 dynamic child refinement 的单位成本，把已验证的 frontier
  tightness 转化为更多 committed nodes，并为后续 hard-clause/full-query integration 选择一个可审计
  的 child cap。
- 选择必须由预注册 Pareto 规则决定；pilot 只负责校准，正式 claim 必须来自独立 fresh repeats。

## Scope

- 固定 workload 为 VNN-COMP 2021 ResNet2B property 0 clause 0；root 仍为 shared top-width 128/32
  单 pass + objective-influence 128/32 单 pass，child 只改变 `max_neurons_per_relu`。
- pilot candidate caps 固定为 `[8, 16, 32, 64, 128]`，passes=1、chunk=32、objective influence、
  sound parent constraint consumption 全部不变；candidate 顺序固定为轮转后的 `[32, 8, 128, 16, 64]`，
  避免单调 warm/order 偏差。
- 选择规则：以 cap128 的 `root-global→ancestral` worst-active-lower gain 为 reference；在 root exact、
  parent lineage valid 且 gain retention `>=0.90` 的 candidates 中选择最小 cap。若无 candidate 达标，
  NRIR-33 以 `VALIDATED-NO-GO` 关闭，不放宽阈值。
- pilot timing 只用于 resource diagnosis，不能形成 performance claim。

## Tasks

1. [x] 新增 additive child-budget Policy/Decision/Plan IR；不得修改 NRIR-32 frozen source/artifact。
2. [x] 实现 thin runtime wrapper，复用 NRIR-32 validated queue engine，同时让 selected cap、candidate
   set、selection mode 与 calibration evidence 进入 Plan hash 和 Task/Schedule identity。
3. [x] 运行单次 five-cap fresh-process pilot，按预注册 retention rule 冻结 winner。
4. [x] 正式重复门禁不启动：winner 恰为 cap128，且 five-cap 全部 7 nodes；NRIR-32 已有 cap128
   三 fresh repeats 均 7 nodes，因此“winner 每轮严格大于 7”在进入新实验前已被反证。
5. [x] full-query 回接不启动；cap-only 不能增加 clause-0 node coverage，按门禁以
   `VALIDATED-NO-GO` 关闭并转向 sibling packed/parametric evaluator。

## Validation

- 所有 candidate root lower 对 cap128/root-global tolerance `1e-5`；child source final-bound/Plan/
  semantic trace 必须逐 parent exact。
- pilot winner 必须完全由冻结规则重算得到；同步修改 winner/summary/digest 必须被 replay 拒绝。
- 正式三轮 winner accepted nodes 必须均严格大于 NRIR-32 cap128 的 7；gain retention 每轮
  `>=0.90`；无 late work 能进入 committed proof identity。
- 若完整 query 回接，baseline verified/unsafe 不得回退，original clause ordinal 必须保持双射。

## Result

- five-cap accepted nodes 全部为 `7`，max depth 全部为 `2`；whole diagnostic time 为
  `65.494/65.689/65.655/67.146/67.937 s`（cap 8/16/32/64/128）。
- worst active lower 随 cap 为 `-173.078613/-162.253326/-148.134460/-126.962929/-104.765411`；
  root-global reference 为 `-200.465393`，cap128 gain=`+95.699982`。
- 90% retention 规则只能选择 cap128；所有较小 cap 都降低 bound quality且没有增加 committed
  nodes。因此 child cap 不是当前吞吐主因，cap-only 路线 `VALIDATED-NO-GO`。
- pilot hash=`db9b406eebebad0c1c4d6f39e8088667935f10e3d54f38cb848dce792dd757eb`；
  replay 与 5 个 focused tests 通过。下一门禁为 sibling packed refinement/evaluation + parametric
  evaluator ownership，保持 cap128 与 60 秒 deadline。

## Rollback

- 新文件均 additive；删除 NRIR-33 IR/runtime/tests/scripts/docs/artifact 即可回到 `main@f58daec`。
  NRIR-32 evidence hash `8fba8dec…11bfc` 必须继续 replay。

## Links

- changelog: `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PARETO_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
