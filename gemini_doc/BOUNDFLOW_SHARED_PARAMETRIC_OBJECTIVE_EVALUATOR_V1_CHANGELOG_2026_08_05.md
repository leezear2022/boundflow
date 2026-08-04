---
status: completed
updated: 2026-08-05T00:55:00Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_SHARED_PARAMETRIC_OBJECTIVE_EVALUATOR_V1
stage: s01
---

# Shared Parametric Objective Evaluator v1 Changelog

## Summary

- NRIR-37 启动。NRIR-36 equal-remaining slice 三轮 coverage=`[[3,3],[3,3],[3,1]]`，第二条 clause
  在一轮只提交 root；本轮不调 slice 常数，改为复用已存在但尚未进入 ancestral sibling evaluator 的
  parametric optimizer compiler，并移除 production path 不需要的 selected-native audit re-execution。

## Changes

- 合并并同步 PR #47，NRIR-36 以 `VALIDATED-NO-GO` 冻结在 `main@c5ce3e6`。
- 建立 `feat/shared-parametric-objective-evaluator-v1`，以 DocOps 登记下一 active gate。
- 对 clauses 2/3 执行一次只读 phase profile，未修改 runtime 或 artifact。
- 新增 `NativeSharedParametricAncestralPlanIR/BatchIR/TaskIR/ScheduleIR`：frozen NRIR-34 plan
  作为嵌套 source contract，root 或完整 sibling pair 才能形成 atomic batch commit；每个 commit
  精确绑定 production batch、cache event、instance、template/task/schedule、refinement 与 evaluation。
- 新增 shared-parametric ancestral queue：query 内复用 NRIR-28 template cache，root/child 只实例化
  exact objective/split/intermediate/warm/refinement state；生产路径不构造 audit hash chain，也不执行
  selected-native replay。deadline 后完成的 pair 不提交，并单列 discarded compiler trace。
- 新增 NRIR-36 control × NRIR-37 evaluator 的一等 multi-clause runtime；保留 frozen floor、rank、
  top-2、dynamic equal-remaining slices 与 aggregate，只新增一个跨 batch/跨 clause cache owner。
- 新增 pilot/formal generate+replay runner、14 个 runtime contract tests 与 12 个 artifact/tamper tests；
  冻结真实 ResNet pilot 和三 fresh-process formal artifact。

## Validation

- batch total=`6.657—7.405 s`；每 batch 可直接归因的重复 compile 为 optimizer
  `0.979—1.103 s` + selected-native `0.417—0.532 s`，selected-native execute 另占
  `1.295—1.314 s`。第二条 root+pair 约 `14.0 s`，与可用 slice 临界相撞。
- profile run 仍选择 `[2,3]`，coverage=`[3,1]`，复现 NRIR-36 failure mode；数据只用于门禁设计，
  不形成性能 claim。
- first-class 实现前的 clause 2 parity feasibility：audit→parametric root=
  `6.672017→0.033680 s`、child pair=`7.324277→0.677186 s`，cache 首次 miss、随后 exact hit。
  lower/split/alpha/beta 与两个 refinement hashes exact；upper max diff=`1.5258789e-5`，满足既有
  `allclose(atol=1e-5,rtol=1e-5)`。据此在 formal pilot 前修正计划中误写的纯绝对容差。
- first-class parity 复核：frozen audit root+pair=`14.096428 s`，shared evaluator=`1.211498 s`；
  lower/branch/split/α/β/refinement hash exact，upper max diff=`1.52587890625e-5` 且 allclose 通过。
- 单轮 coverage：floor=`20.615271 s`、whole=`51.083770 s`，selected `[2,3]` 均 `31 nodes/15 groups`，
  cache 1 miss + 31 hits；pilot hash=
  `5c79bcc6e744ed1d29520a76331c9823b2ccfa144332e96c401271241616bf86`。
- 三 fresh processes：floor=`[21.704740,21.802033,21.784891] s`，whole=
  `[52.032317,52.268473,51.926746] s`；三轮 packed nodes 均 `[[31,31]]`、每轮只编译一次，
  formal hash=`7ff6aef76f6fe2b8778faba2e599e440c2dbf14ac4808bfb0c7e07f72fb74238`。
- replay、wrong rank/selection/source、slice inflation、partial group、ordinal omission、second miss、
  template count、event ordinal、native-reexecution 与 compiler coverage tamper 均 fail closed；26 focused
  tests、全量 `916 passed, 37 skipped`、mypy clean、Pylint `10.00/10`。

## Decisions

- 复用 NRIR-28 template/instance/cache，而不是另造无 lineage 的 shape cache；跨 clause 只共享静态
  contract，objective content 与 ancestral refinement 仍逐 instance hash-bound。
- 先做 frozen audit-vs-parametric first-pair parity，再接 top-2 global-budget runtime；不以更快但更松的
  production bound 换取 node 数。
- ad-hoc feasibility 只证明值得实现；正式 claim 仍须由 first-class Batch/Cache IR、独立 validator 与
  artifact pilot 重新建立，不能引用 monkeypatch timing 作为最终性能数据。
- 预注册 coverage gate 三轮全部成立，NRIR-37 以 `VALIDATED-REDUCED` 关闭；该结果不推翻 NRIR-36
  allocation NO-GO，而是证明其失败原因可归因于旧 evaluator 的重复编译/审计路径。最终性质仍
  9/9 unresolved，CPU 单 workload、无 GPU/competitor/multi-workload/ASPLOS-ready 或 speedup claim。

## Follow-Ups

- 下一门禁不得继续调 top-k/slice/cache；先对已完整覆盖的 clauses 2/3 depth-4 frontier（worst lower
  `-37.574287/-35.900215`）做 tightness attribution，区分更深搜索、branch score、refinement cap 与
  optimizer bound 的收益，再预注册一个单变量 stronger-bound/candidate gate。

## Links

- plan: `gemini_doc/BOUNDFLOW_SHARED_PARAMETRIC_OBJECTIVE_EVALUATOR_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
