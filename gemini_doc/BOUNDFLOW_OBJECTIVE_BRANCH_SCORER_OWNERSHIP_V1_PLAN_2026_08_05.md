---
status: active
updated: 2026-08-04T22:16:52Z
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

- [ ] 新增 capsule Plan/Task/Schedule binding：绑定 objective、split、selected optimizer state、policy、
  candidate table、task/schedule 与 immutable tensor-content token；任一漂移 fail closed。
- [ ] 新增 prevalidated scorer runtime：执行五阶段 schedule 时 ENUMERATE task 读取 Plan-owned candidates，
  materialize/evaluate/reduce/select 语义不变；禁止调用 historical `_enumerate_candidates`。
- [ ] 接入 additive shared production queue；对 clauses 2/3 的 31 nodes 与 NRIR-39 historical execution
  逐节点比较 selected branch、全 score rows、child-lower hash、queue lower/upper、split、α/β、refinement。
- [ ] Phase A 做 3 fresh counterbalanced old/new fixed-31 paired runs；只有 parity/call-count/wall gate 全过
  才运行 Phase B three fresh whole-query/global-60s formal。
- [ ] artifact replay 重算 capsule/parity/call/timing/global gates；同步重哈希 capsule/candidate/score/call/
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

## Rollback

- 删除本分支 additive NRIR-42 文件即可回到 `main@355e80b`；NRIR-39/40/41 evidence 不变。

## Links

- changelog: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SCORER_OWNERSHIP_V1_CHANGELOG_2026_08_05.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_PRODUCTION_COST_ATTRIBUTION_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
