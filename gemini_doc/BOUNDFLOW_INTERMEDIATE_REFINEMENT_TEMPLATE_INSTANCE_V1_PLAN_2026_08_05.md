---
status: preregistered
updated: 2026-08-05T02:44:32Z
type: plan
topic: boundflow
slug: intermediate-refinement-template-instance-v1
stage: s01
---

# BoundFlow Intermediate Refinement Template/Instance v1 Plan

## Goal

- 把 NRIR45 每个 child 都重建的 exact refinement Program 拆成 first-class
  `IntermediateRefinementPlanTemplateIR` 与 node-specific `IntermediateRefinementPlanInstanceIR`；
- 每条 31-node queue 只完整编译/降低/准入一次静态 Template，30 个 child 只绑定动态 Instance；
- 保持目标选择结果、selected-CROWN、bounds、optimizer、branch candidate、queue order、31/depth4、
  NRIR44 floor 与 global-60s deadline 完全不变；
- 先证明 compiler IR ownership 真正减少静态 compile/lower/validate/hash，再以 frozen NRIR45 为
  control 判断 CPU 墙钟；不能把 Template 名称当作已发生复用。

## Scope

### 依赖与基线

- preregistration branch=`feat/intermediate-refinement-template-instance-v1`；
- stacked documentation base=`a2d8f96`，依赖 NRIR45 draft PR #56；
- 正式实现/计时只有在 `nrir45-20260805` 外部审计 approve 且 PR #56 合入 main 后启动；
- frozen NRIR45 feature=`8b8766e`，Phase A formal hash=
  `be1ccb4229d8b88970c9f9f5bae9d6ff8156d4e9b53c84a218a2a1dd6005d439`，Phase B payload hash=
  `4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8`；
- control whole trace=`31.262521/31.319772/31.470078 s`，measured wall=
  `36.396631/36.513683/36.611709 s`，selected `[2,3]`、nodes `[31,31]`、final 9/9 unknown。

### 开工前 residual attribution

NRIR45 Phase-B 三个 raw shards 独立拆分为：

- `execute_floor` median=`10.818262 s`；
- 两条 `execute_packed_slice` 六次样本的 median=`9.932808 s`；
- `compile_packed_plan` 六次 median=`0.146457 s`；
- `rank_candidates` median=`0.024966 s`，aggregate/emit 近零；
- action 累加与 whole trace 的差约 `0.32 s`；measured wall 与 trace 的差约 `5.2 s`，主要属于
  artifact/source validation 与序列化，不得伪装成 production compiler gain。

一次 `repeat=0`、CPU8、diagnostic-only 的低开销函数计时得到：whole trace=`30.826307 s`；60 个
child 的 prepared compile=`5.300590 s`，prepared execute=`5.659414 s`，两条 queue 的 per-child
refinement 合计=`10.975123 s`；32 个 optimizer execute 合计=`1.156098 s`。该次插桩结果只用于
选路线，不是正式性能 claim，也不替代 three-repeat formal。

另一次故意使用非冻结 `repeat=99` 的 cProfile 在最终 worker validation 被拒绝，证明 formal runner
不会接受非预注册 repeat；其 pstats 只作诊断，不进入 artifact 或 claim。

### 唯一变量

只改变 intermediate-refinement compiler ownership：

- `PlanTemplate` 拥有 primal graph、policy/multi-pass/chunking、target-selection recipe、动态 slot
  schema、Task topology 与 Schedule topology；
- `PlanInstance` 拥有 exact input/split/source lineage、initial bounds、objective/influence、动态 target
  ledger 及所有 node-specific hashes；
- `ScheduleTemplate` 一次准入，Instance binding 逐 child 执行；selected-CROWN 仍逐 child、同顺序、
  同 chunk，NRIR46 不引入跨节点数值 batching；
- target table 明确是 Instance-owned，因为初始 bounds、objective influence 与 ambiguous set 会随节点
  变化；禁止把某个 child 的 target ledger 错误共享给其他 child；
- full replay 从 Template + Instance 重建原 NRIR45 exact Program/Task/Schedule 并逐项比较。

冻结不改：NRIR44 floor/rank/top-2，NRIR45 capsule semantics，refinement policy/pass/cap、target math、
selected-CROWN math、optimizer steps/state、objective scorer、node/depth budget、queue/cache、dtype、threads、
workload、threshold、global deadline、aggregation 与 `performance_claimed=false`。

### IR 与 Schedule 所有权

- `IntermediateRefinementPlanTemplateIR`：静态 module/policy/selection/chunk contract 与 dynamic slot types；
- `IntermediateRefinementPlanInstanceIR`：Template hash、node/split/source/objective/target ledger exact binding；
- `IntermediateRefinementTaskTemplateIR`：
  `ADMIT_TEMPLATE -> MATERIALIZE_INSTANCE -> SELECT_INSTANCE_TARGETS ->
  EXECUTE_SELECTED_CROWN -> COMMIT_INSTANCE_RESULT -> EMIT_INSTANCE_RECEIPT`；
- `IntermediateRefinementScheduleTemplateIR`：静态依赖与 launch kinds；
- `IntermediateRefinementInstanceScheduleIR`：只绑定 instance value IDs、ordinal 与 exact owner，不重新
  lower 静态拓扑；
- Template/Instance receipt 必须包含 versioned schema、stable hash、exact lineage 与 full-validation
  counters；禁止裸 dict、`Any`、object-ID-only cache 或无版本 bool token。

## Tasks

### 0. 审计与归因门禁

- [ ] NRIR45 exchange approve，executor close，PR #56 合入 main；
- [ ] 在无 cProfile 扰动下把 `5.300590 s` compile 进一步拆为 forward materialization、objective
  influence、target selection、Plan construction、lowering、full validation/hash；
- [ ] 明确 static-shareable 与 dynamic-required 成本；若 static-shareable whole-query median `<1.5 s`
  或 ceiling gain 不大于 pooled MAD，本路线直接 `VALIDATED-NO-GO`。

### A. Template/Instance IR

- [ ] 新增 versioned PlanTemplate、PlanInstance、TaskTemplate、ScheduleTemplate、InstanceSchedule 与
  receipt dataclasses；
- [ ] 固定 static/dynamic field ownership 和 deterministic canonical hashes；
- [ ] 用一个 Template 绑定 30 个 exact child Instances，禁止共享动态 target ledger；
- [ ] wrong template/instance/module/policy/chunk/split/source/objective/target、stale Tensor 与 mutation
  全部在 selected-CROWN 前 fail closed；
- [ ] additive 接入 prepared per-child/shared queue，frozen NRIR42/44/45 文件不做破坏性改写。

### B. Phase A compiler/queue formal

- [ ] clauses 2/3 各做 three fresh counterbalanced NRIR45-control / NRIR46-template 31-node queues；
- [ ] 每条 candidate queue Template compile/lower/full-admit=`1`，Instance bind/full-replay=`30/30`；
- [ ] target selection 与 selected-CROWN semantic launches 仍为 30，target ledger 逐 node exact；
- [ ] branch/score/state/ancestry/refinement/bounds/worst lower/31 nodes 全 exact；
- [ ] 两条 queue candidate/control median ratio 均 `<=0.90` 且改善大于 pooled MAD；
- [ ] typed reconstruction 与 synchronized outer-rehash Template/Instance cross-bind tamper fail closed。

### C. Phase B whole query

- [ ] 只有 Phase A correctness、ownership、timing 全过才启动；
- [ ] three fresh CPU8 global queries，floor/rank/selected `[2,3]`、`[31,31]` nodes 与 worst lower exact；
- [ ] trace median ratio vs frozen NRIR45 `<=0.90` 且改善大于 pooled MAD；
- [ ] measured wall 同向改善且改善大于 pooled MAD，不设事后放宽门槛；
- [ ] final 9/9 unknown、60/60 Instance full replay、artifact/tamper/full suite/Black/mypy/Pylint/DocOps
  全过。

## Validation

Phase 0 若证伪 shareable ceiling，或 Phase A 任一 exact/ownership/timing gate 失败，则
`VALIDATED-NO-GO` 并禁止 Phase B。只有 Phase A/B 全部门禁成立，才允许 fixed ResNet2B property 0
CPU8 compiler ownership `VALIDATED-REDUCED`。

正式 artifact 必须记录 Template compile、Instance bind、target selection、selected-CROWN、full replay
计数和 action timing；replay 必须先验验证全部 digest，再重建完整 IR。所有 artifact 继续
`performance_claimed=false`。

本阶段即使通过，也不形成公平竞品、10x、GPU、多 workload、property closure、完整 verifier E2E 或
ASPLOS-ready claim。理论上即使把已测 `5.30 s` prepared compile 全部消除，当前约 `31.3 s` trace
仍只能到约 `26 s`；因此 NRIR46 是 compiler IR 基础设施与局部降本，不是论文性能终点。

## Rollback

- 所有实现 additive；失败时继续使用 audited NRIR45 prepared path；
- 不以降低 target cap/pass、node/depth、跳过 target selection、共享动态 targets、放宽 validation、
  减少 clauses 或更改 deadline 换取收益；
- CPU 上若只减少 launch/compile count 但 timing 不过，机制可保留为 correctness/compiler ownership
  evidence，production 默认保持 NRIR45；
- PR #56 审计未批准前，本分支只允许预注册文档，不实现或生成正式结果。

## Links

- changelog: `gemini_doc/BOUNDFLOW_INTERMEDIATE_REFINEMENT_TEMPLATE_INSTANCE_V1_CHANGELOG_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
- dependency audit: `gemini_doc/BOUNDFLOW_NRIR45_EXTERNAL_AUDIT_HANDOFF_2026_08_05.md`
