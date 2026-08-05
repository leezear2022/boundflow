---
status: preregistered
updated: 2026-08-05T12:39:58Z
type: plan
topic: boundflow
slug: top2-production-execution-cost-attribution-v1
stage: s01
---

# BoundFlow Top-2 Production Execution Cost Attribution v1 Plan

## Goal

- 在 frozen NRIR-45 production baseline 上，把 clauses 2/3 两条各约 10 秒的 31-node top-2 queue
  分解为互斥、可重放的 compile、refinement math、optimizer、branch 与 queue/control 成本；
- 用 three-fresh-process paired control/profile 判断 dominant execution category，而不是直接猜测
  selected-CROWN、optimizer、Python queue 或 backend；
- 只为 NRIR-49 选择一个可行动的单变量，不在本轮修改算法、IR/runtime 语义或声称 speedup。

## Scope

### 基线与边界

- branch=`feat/top2-production-execution-cost-attribution-v1`；integration base=
  `main@1e44949`（PR #58 merge）；
- frozen production source=NRIR-45 Phase B hash=
  `4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8`；
- NRIR-47 hash=`a7561e5187a6e396905d261e739280e39f2c3480e83ba2af0fbe6e3b1ec042ce`，
  candidate 不默认启用；本轮必须运行 legacy NRIR-45 prepared production route；
- 已知但未升级的单次 diagnostic：60 child prepared compile/execute=`5.300590/5.659414 s`，
  optimizer execute=`1.156098 s`。这些只用于定义分类，不能作为最终 dominant claim。

冻结 workload=VNN-COMP 2021 ResNet2B property 0 CPU8、clauses `[2,3]`、objective branch、steps5、
cap128 ancestral refinement、31 nodes/depth4、sibling pairs、dtype、thread count、cache、budget、deadline、
selected targets、bound math 与 `performance_claimed=false`。本轮不启用 NRIR-47 receipt、不改 target
cap/pass、optimizer steps、branch policy、node/depth 或 queue order。

### 互斥计时分类

每条 queue 的 `queue_elapsed_ns` 必须闭合为下列顶层互斥类别：

1. `child_refinement_compile`：30 个 prepared child Program/capsule construction；
2. `child_refinement_execute`：30 个 prepared execution，内部再诊断为
   `fast_validate`、`runtime_target_select`、`selected_crown`、`propagate_forward`、
   `refinement_hash_trace_residual`；
3. `optimizer_prepare`：template acquire、warm-state build、instance instantiate 与 production IR lower；
4. `optimizer_execute`：16 个 batch optimizer execution；
5. `branch_bind_score`：compile-owned candidate binding/score/selection；
6. `materialize_commit`：node/state slicing、batch commit、Task/Schedule/trace/hash materialization；
7. `queue_control_residual`：root setup、heap/branch transitions、deadline checks 与未被上述类别覆盖的
   queue control；该 residual 必须由总墙钟减去其他互斥顶层类别得到，不得用重叠 inclusive 时间相加。

内部子分类只解释父类别，不与顶层再次相加。所有 timer 用 `perf_counter_ns`，fresh worker PID，
control/profile counterbalanced；原始 inclusive/exclusive/call count 和 parent-child closure 全部入 artifact。

## Tasks

- [ ] 新增只读 attribution runner，不修改 frozen production runtime 文件；
- [ ] clauses 2/3 各运行 three fresh counterbalanced control/profile 31-node queues；
- [ ] 记录每个 wrapper 的 calls、inclusive/exclusive ns、顶层互斥 category、residual 与 total closure；
- [ ] 保持 branch/score/state/ancestry/refinement/bounds/worst lower/31 nodes exact；
- [ ] typed artifact reconstruction、source/input digest 与同步外层重哈希 category/timer tamper fail closed；
- [ ] 根据预注册路由规则只选择一个 NRIR-49 方向，或明确 attribution inconclusive。

## Validation

- correctness gate：6/6 control/profile exact；accepted nodes=`31`、sibling groups=`15`，clauses 2/3
  worst active lower 与 frozen NRIR-45 exact；
- instrumentation gate：每条 profile 的 category ns 非负、call count 符合生产路径，顶层类别和
  residual 对 queue total 的 closure error `<=1%`；
- perturbation gate：每条 clause profile/control queue median ratio `<=1.05`，且 profile/control
  semantic trace exact；超过则 attribution `INVALID`，不得选路线；
- dominance gate：同一顶层类别须在 clauses 2/3 各自的 3/3 repeats 排第一，且两条 clause 的
  category median share 均 `>=20%`。若不成立，本轮以 `VALIDATED-NO-GO/INCONCLUSIVE` 关闭；
- stability gate：dominant category 在每条 clause 的 max-min share `<=10` percentage points，且其
  median exclusive ns 大于三轮 pooled MAD；
- 若 `child_refinement_execute` dominant，内部子类必须同时满足两条 clause median share of parent
  `>=30%` 才能成为 NRIR-49；否则只允许继续更细的 execution attribution；
- 若 `queue_control_residual` dominant，只能先增加具名计时点，不得直接声称 Python queue 是根因；
- 本轮只有 attribution claim，不产生 speedup、property closure、GPU、competitor、multi-workload 或
  ASPLOS-ready claim。

## Rollback

- runner/instrumentation 全部 additive；不修改 frozen runtime 或 artifact；
- profile overhead 超门槛、类别不闭合、语义漂移或 dominant 不稳定时直接停止并修测量协议；
- 不通过 post-hoc 合并/拆分类别、删除慢 repeat、改 workload 或降低 dominance gate 选择路线。

## Links

- changelog: `gemini_doc/BOUNDFLOW_TOP2_PRODUCTION_EXECUTION_COST_ATTRIBUTION_V1_CHANGELOG_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
- predecessor: `gemini_doc/BOUNDFLOW_SINGLE_PASS_TARGET_ADMISSION_RECEIPT_V1_PLAN_2026_08_05.md`
