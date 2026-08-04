---
status: completed
updated: 2026-08-05T07:18:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_OBJECTIVE_BRANCH_SCORER_OWNERSHIP_V1
stage: s01
---

# Objective Branch Scorer Ownership v1

## Goal

- 把 NRIR-41 定位的 candidate enumeration/validation 重复改为一等 scorer ownership：candidate table
  只在每个 node 的 branch Plan 编译时生成一次，执行器和下游验证消费同一个不可变 typed capsule，
  不再重建候选。
- 在 exact branch/score/child-bound/queue parity 前提下验证该单变量能否让 objective branch 恢复
  NRIR-40 global 60 秒下的 31/31 production coverage。

## Scope

- 基线：`main@355e80b`，NRIR-41 formal hash=
  `fe67b77197905a8a4d7f92ad5eac686892243dfb0e7d7b7c7434861aaa794834`；分支
  `feat/objective-branch-scorer-ownership-v1`。
- workload 固定 VNN-COMP `cifar10_resnet:000` property 0 clauses 2/3、CPU 8 threads、steps5、cap128、
  31/depth4、best-first、parent warm、sibling atomic commit、query cache、objective policy 全部不变。
- 唯一变量：old 每 31 branch programs 触发 341 次 enumeration/多次 full `program.validate()`，new 由
  `ValidatedBranchProgramCapsuleIR` 单一拥有 candidate table 与 admission token；compile 恰好 1 次
  enumeration/node，execute 恰好 0 次。
- 不修改 NRIR-39/40 frozen 文件；新增 optimized scorer/capsule 与 additive queue composition。
- timing 只作内部准入，`performance_claimed=false`；不形成 competitor/GPU/multi-workload claim。

## Tasks

- [x] 新增 capsule Plan/Task/Schedule binding：绑定 objective、split、selected optimizer state、policy、
  candidate table、task/schedule 与 immutable tensor-content token；任一漂移 fail closed。
- [x] 新增 prevalidated scorer runtime：执行五阶段 schedule 时 ENUMERATE task 读取 Plan-owned candidates，
  materialize/evaluate/reduce/select 语义不变；禁止调用 historical `_enumerate_candidates`。
- [x] 接入 additive shared production queue；对 clauses 2/3 的 31 nodes 与 NRIR-39 historical execution
  逐节点比较 selected branch、全 score rows、child-lower hash、queue lower/upper、split、α/β、refinement。
- [x] Phase A 做 3 fresh counterbalanced old/new fixed-31 paired runs；只有 parity/call-count/wall gate 全过
  才运行 Phase B three fresh whole-query/global-60s formal。
- [x] artifact replay 重算 capsule/parity/call/timing/global gates；同步重哈希 capsule/candidate/score/call/
  deadline tamper 仍 fail closed。

## Validation

- Phase A correctness：clauses 2/3 old/new 各 31 evaluations/15 groups，root、每节点 lower/upper、selected
  candidate、score rows、child-lower、split、optimizer/refinement hashes exact；cache ownership不变。
- ownership gate：每 clause 31 nodes 总 enumeration calls 恰好 `31`，其中 compile=`31`、execute=`0`；
  不得用跳过验证、弱化 hash 或 audit-only 重放伪造。
- internal cost gate：3 fresh counterbalanced runs 中，两条 clause 的 new/old queue median ratio 均
  `<=0.75`，且每个 ratio 的改善幅度大于各自 MAD；未过则以 scorer optimization NO-GO 关闭。
- Phase B production GO（只在 Phase A 全过后运行）：three fresh whole-query repeats 均 selected `[2,3]`，
  clauses 2/3 各 31 nodes/15 groups，worst-active lower 相对 NRIR-37 widest 各 `>=+1.0`，whole cooperative
  `<=70s`，无 partial/reset/recompile/evidence omission。否则保留 correctness 但 production NO-GO。
- targeted/full pytest、Black、mypy、Pylint、`dol validate`、`dol lint --soft`。

实际结果：Phase A clauses 2/3 new/old queue median ratio=`0.706888/0.698486`，median 节省
`5.468696/5.680614 s`，均大于两侧 MAD；每轮 old/new 都为 `341→31` enumeration，new
compile=`31`、execute=`0`，六组 31-node exact parity 全过。Phase-A formal hash=
`0d310c2ffc96844648a83f9921bc7f353ec8425986bccb36f75e6d1cd2b25b58`。

条件 Phase B 三 fresh whole-query 均 selected `[2,3]`、accepted `[[31,31],[31,31],[31,31]]`，
whole=`[57.175184,57.697757,58.114412] s`，worst active lower 固定为
`-35.530926/-30.258448`，相对 NRIR-37 widest 改善 `+2.043362/+5.641768`；formal hash=
`7274e834b3bf08a9e138fa3284b70222620cf3c571395331e1a87ed5fee7d759`。targeted `10 passed`，
全量 `958 passed, 37 skipped`；Black/mypy/Pylint 通过，Pylint `10.00/10`。

本阶段以 fixed ResNet2B property 0、CPU8、global-60s production admission
`VALIDATED-REDUCED` 关闭；final 仍 unknown，且不形成 competitor/GPU/multi-workload/ASPLOS-ready claim。

## Rollback

- 删除本分支 additive NRIR-42 文件即可回到 `main@355e80b`；NRIR-39/40/41 evidence 不变。

## Links

- changelog: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SCORER_OWNERSHIP_V1_CHANGELOG_2026_08_05.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_PRODUCTION_COST_ATTRIBUTION_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
