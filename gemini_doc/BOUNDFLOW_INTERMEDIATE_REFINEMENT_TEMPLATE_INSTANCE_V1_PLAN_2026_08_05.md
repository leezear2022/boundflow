---
status: validated-no-go
updated: 2026-08-05T11:10:00Z
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
- integration base=`main@6cd229a`（PR #56 merge）；
- 用户已明确豁免外部模型 review；NRIR45 通过 executor deterministic replay/tamper/tests/static/DocOps
  门禁后合入，未声称获得独立 auditor approval；
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

### 0. 合并与归因门禁

- [x] 用户豁免外部 review；NRIR45 executor 自检通过，PR #56 合入 `main@6cd229a`；
- [x] 在无 cProfile 扰动下把 `5.300590 s` compile 进一步拆为 forward materialization、objective
  influence、target selection、Plan construction、lowering、full validation/hash；
- [x] 明确 static-shareable 与 dynamic-required 成本；若 static-shareable whole-query median `<1.5 s`
  或 ceiling gain 不大于 pooled MAD，本路线直接 `VALIDATED-NO-GO`。

Phase 0 三个 fresh process 的 compile total=`5.356892/5.366369/5.452290 s`，strict static
topology=`1.071197/1.062492/1.071704 s`，Template/Instance ownership-convertible ceiling=
`2.097255/2.102134/2.109857 s`。strict static median=`1.071197 s < 1.5 s`，预注册门禁失败；
因此 NRIR46 在 Phase 0 以 `VALIDATED-NO-GO` 关闭。

### A. Template/Instance IR

- [x] **GATED OFF**：Phase 0 strict static gate 失败，未实现 PlanTemplate/PlanInstance；
- [x] **GATED OFF**：未改 frozen NRIR42/44/45 production path，也未制造虚假的共享 target ledger。

### B. Phase A compiler/queue formal

- [x] **GATED OFF**：按预注册规则未启动 Phase A，不存在 NRIR46 queue timing claim。

### C. Phase B whole query

- [x] **GATED OFF**：Phase A 未启动，故 Phase B 也未启动；NRIR45 的 final 9/9 unknown 与
  `performance_claimed=false` 保持不变。

## Validation

Phase 0 若证伪 shareable ceiling，或 Phase A 任一 exact/ownership/timing gate 失败，则
`VALIDATED-NO-GO` 并禁止 Phase B。只有 Phase A/B 全部门禁成立，才允许 fixed ResNet2B property 0
CPU8 compiler ownership `VALIDATED-REDUCED`。

正式 artifact 必须记录 Template compile、Instance bind、target selection、selected-CROWN、full replay
计数和 action timing；replay 必须先验验证全部 digest，再重建完整 IR。所有 artifact 继续
`performance_claimed=false`。

实际 Phase 0 artifact 位于
`artifacts/intermediate-refinement-template-instance/vnncomp21-resnet2b-property0-three-repeat-cpu-phase0-v1/`，
formal hash=`712ce359501a010a197797909ab71fb127ebda43329dd3a7a8e21b6dbb4cf846`。
replay 逐项验证 source/artifact digest，并在同步更新外层 hash 后仍拒绝改变 distinct target identity
count 的篡改。三轮均保持 selected `[2,3]`、nodes `[31,31]`、60/60 capsules/full replay；每轮
60 个 target identity/table hash 均互异，而 primal graph、Task/Schedule topology 各只有 1 种。

本阶段即使通过，也不形成公平竞品、10x、GPU、多 workload、property closure、完整 verifier E2E 或
ASPLOS-ready claim。理论上即使把已测 `5.30 s` prepared compile 全部消除，当前约 `31.3 s` trace
仍只能到约 `26 s`；因此 NRIR46 是 compiler IR 基础设施与局部降本，不是论文性能终点。

## Rollback

- 所有实现 additive；失败时继续使用 self-validated NRIR45 prepared path；
- 不以降低 target cap/pass、node/depth、跳过 target selection、共享动态 targets、放宽 validation、
  减少 clauses 或更改 deadline 换取收益；
- CPU 上若只减少 launch/compile count 但 timing 不过，机制可保留为 correctness/compiler ownership
  evidence，production 默认保持 NRIR45；
- Phase 0 strict static ceiling 已失败，NRIR46 禁止实现；下一路线只能针对已测的 64 次冗余 target
  reselection 建立更窄的单次 admission receipt，且必须另行预注册，不能把它冒充 NRIR46 通过。

## Links

- changelog: `gemini_doc/BOUNDFLOW_INTERMEDIATE_REFINEMENT_TEMPLATE_INSTANCE_V1_CHANGELOG_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
- review-waiver record: `gemini_doc/change_2026-08-05_nrir45_external_review_waiver.md`
