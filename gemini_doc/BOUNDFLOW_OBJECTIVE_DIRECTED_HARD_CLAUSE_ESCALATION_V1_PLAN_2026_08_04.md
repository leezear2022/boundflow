---
status: completed
updated: 2026-08-04T13:36:26Z
type: plan
topic: boundflow
slug: objective-directed-hard-clause-escalation-v1
stage: s01
---

# Objective-Directed Hard-Clause Escalation v1 Plan

## Goal

- 保持 NRIR-30 baseline、exact unresolved admission、shared top-width refinement、31-node escalation、
  original-ordinal aggregate 与 60 秒 whole deadline 不变；对每个 admitted scalar clause 再编译一份
  objective-influence selected-CROWN refinement，并以 shared refinement 的 validated final bounds 作为
  sound source constraints，判断目标相关 tightening 能否关闭 MNIST clause 8 或改善 ResNet hard clauses。
- per-clause compile/refine/query 必须是 guarded first-class Task/Schedule，不允许在 runner 中暗循环；
  deadline 到期后保留 NRIR-30/shared baseline verdict，不得使用未完成 child proof。

## Scope

- 固定三个 NRIR-30 workload、CPU threads=8、5-step optimizer、4-step candidate search、baseline
  `7/depth2`、hard query `31/depth4`、batch `2/4`、60 秒 whole deadline。
- shared source policy 固定 top-width `128 targets/ReLU, chunk32`；每个 admitted clause 的 second
  refinement 固定 objective-influence `128 targets/ReLU, chunk32`、一 pass，并显式绑定 exact scalar
  objective hash、shared source Plan/semantic trace/final bounds。
- 任务模板为 baseline/admit/shared refine，加每个 original clause 的 guarded compile-objective/
  execute-objective/execute-query，再 aggregate/emit；未 admitted clause 必须记录 guard skip。
- 不修改 NRIR-30 frozen IR/runtime/runner；不声明 speedup、GPU、external verifier、完整 suite 或
  ASPLOS-ready。

## Tasks

1. [x] 定义 objective-escalation PlanTemplate、per-clause guarded Task/Schedule 与 dynamic clause trace；
   绑定 NRIR-30 base program hash、shared source lineage 和 exact objective ordinal/hash。
2. [x] 实现 additive runtime，执行 baseline→admit→shared source→per-hard-clause objective refinement→
   single-clause parametric query→aggregate，所有 stage 共用一个 deadline。
3. [x] 增加 source-lineage、objective/ordinal、guard、deadline discard、aggregate non-regression 与同步
   digest tamper 测试。
4. [x] 先跑三 workload 单次 pilot；只有 final verified 严格增加，或 ResNet 所有 common root lower
   不退化且至少一条严格改善 `>1e-4`，才启动三 fresh repeats；否则直接 NO-GO。
5. [x] pilot 通过后完成三次 fresh repeats、artifact/replay、focused/static 验证与权威文档更新；
   full regression、DocOps 收口和 PR 发布在交付门禁执行。

## Validation

- NRIR-30 baseline/admission/shared refinement 必须按 program/hash/semantic trace 对齐；objective child
  的 source constraints 必须来自已验证 shared execution，不能裸传 mapping。
- final verified 必须包含 NRIR-30 final verified；所有 admitted clauses 都必须有 original↔scalar
  ordinal 双射或显式 deadline skip。objective child common bounds 必须相对 shared source 单调。
- 三次重复仅在 pilot gate 通过后执行；performance 仍为 false，raw timing 只验证 whole deadline。
  若只改善 root lower 而无新增 closure，最多 tightness `VALIDATED-REDUCED`；完全无严格改善则
  `VALIDATED-NO-GO`。

## Rollback

- 删除本轮新增 objective-escalation IR/runtime/runner/tests/artifact 即回到 `main@6306a34`；任何
  source/objective/ordinal/deadline/aggregate gate 失败都拒绝 child result并保留 NRIR-30 路径。

## Closure

- 三拓扑单次 pilot 通过：没有新增 closure，但 ResNet 9/9 common root lower 全部改善，最小
  `+81.522583`，超过预注册 `1e-4` 门槛；随后才启动三次正式重复。
- 9 个 fresh worker 全部 `fallback=none`。MNIST 保持 8/9、OVAL 保持 9/9；ResNet 仍 0/9，
  但九条 root lower delta 三轮逐值一致，范围 `+81.522583—+179.970459`。
- focused `8 passed`、全量 `838 passed, 37 skipped`、Black/Mypy/Pylint `10.00/10` 与
  `git diff --check` 通过。artifact evidence hash=
  `fb9e503bdf93cb9ce56f52915f1965f1f542e092945d4d7d77d8b8c4bd91764a`，source-to-program
  replay 与 digest gate 通过。结论为 root-bound tightness `VALIDATED-REDUCED`，不是 property
  closure、performance 或 ASPLOS-ready。
- 下一门禁为 NRIR-32 objective-ancestral hard-clause escalation：把 validated objective root
  execution 作为动态 child refinement 的 typed parent source，检验 root tightening 能否传到
  frontier/closure；不再继续堆 root-only pass。

## Links

- changelog: `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_TYPED_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
