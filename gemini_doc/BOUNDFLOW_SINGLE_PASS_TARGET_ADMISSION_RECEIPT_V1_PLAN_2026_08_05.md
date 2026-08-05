---
status: validated-no-go
updated: 2026-08-05T12:32:05Z
type: plan
topic: boundflow
slug: single-pass-target-admission-receipt-v1
stage: s01
---

# BoundFlow Single-Pass Target Admission Receipt v1 Plan

## Goal

- 将 NRIR46 Phase 0 已测得的每轮 64 次冗余 target reselection 收敛为一次 exact selection + typed
  admission receipt；
- production compile 对每个 child 只调用一次 target selector，显式 full replay 仍从 exact
  bounds/objective/policy 重算并逐项比较；
- 保持 60 个 node-specific target ledger 全部动态独立，不引入 Template/Instance、跨节点 target
  共享、数值 batching 或算法语义变化；
- 用 compiler-only、per-clause queue 和 whole-query 三层 fresh-process 门禁判断小而真实的收益，
  不把计数下降自动升级为性能 claim。

## Scope

### 基线与证据来源

- branch=`feat/single-pass-target-admission-receipt-v1`；integration base=`main@ca0bcf3`
  （PR #57 merge）；
- frozen NRIR45 production feature=`8b8766e`，Phase A formal hash=
  `be1ccb4229d8b88970c9f9f5bae9d6ff8156d4e9b53c84a218a2a1dd6005d439`，Phase B hash=
  `4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8`；
- NRIR46 Phase 0 hash=`712ce359501a010a197797909ab71fb127ebda43329dd3a7a8e21b6dbb4cf846`：
  compile total median=`5.366369 s`，target selection observed/semantic=`124/60`，冗余选择估计
  median=`1.038153 s`；60/60 target ledger 互异；
- frozen NRIR45 queue control median：clauses 2/3=`9.444103/9.666283 s`；whole trace=
  `31.262521/31.319772/31.470078 s`，measured wall=
  `36.396631/36.513683/36.611709 s`；final 9/9 unknown。

### 唯一变量

只改变 target-selection admission ownership：

- selector 从 exact `initial_relu_pre + effective selection policy + objective influence` 生成 targets
  和 `NativeTargetAdmissionReceiptIR`；
- receipt 绑定 primal graph、input bounds、split state、initial intermediate bounds、policy、objective、
  objective influence、ordered target table、target count、selector schema 与一次 selection count；
- production validator 消费 receipt 并验证所有绑定/hash/顺序，不再次调用 selector；
- `validate_full` 必须绕过 fast path，从源状态再次调用 selector并逐字段比较，artifact replay 也必须
  记录 full-replay selector count；
- prepared capsule/Task/Schedule 显式绑定 receipt hash，Schedule 分开记录 production selection 与
  replay selection，不使用裸 dict、`Any`、无版本 bool token 或仅 object-ID 的语义凭证。

冻结不改：initial forward/materialization、objective influence 计算、target 排序/打分/math、multi-pass、
selected-CROWN、optimizer、branch candidate/scorer、queue、root projection、top-2、node/depth/cap、deadline、
dtype、threads、workload、aggregation、final verdict 与 `performance_claimed=false`。

### 允许的实现边界

- 可在 `boundflow/ir/` 新增 typed receipt IR，并对 native compiler/prepared compiler 做最小重构，
  使“build once”与“full replay”成为两个具名入口；
- frozen NRIR45 artifact 与既有 public `compile_native_intermediate_refinement_program` 的 full validation
  语义保持兼容；旧调用者默认仍获得 replay-grade validation；
- production queue 只切换到显式 single-pass compiler，不更改执行、budget 或 queue control。

## Tasks

### A. Receipt IR 与 fail-closed admission

- [x] 新增 versioned `NativeTargetAdmissionReceiptIR` 与 canonical stable hash；
- [x] 绑定 exact selection inputs、effective policy、ordered target table 和 selection_count=`1`；
- [x] 以 additive compiler construction 与 full replay 具名路径保持旧 public compiler 文件/语义不变；
- [x] prepared Plan/Task/Schedule/capsule 显式消费 receipt hash；
- [x] wrong graph/input/split/bounds/policy/objective/influence/target count/order/value/receipt、cross-program
  receipt、Tensor mutation/stale owner 全部在 selected-CROWN 前拒绝；
- [x] full replay 真实重调 selector；未把 receipt 自校验冒充 semantic replay。

### B. Phase A compiler/queue formal

- [x] clauses 2/3 各 three fresh counterbalanced NRIR45-control / NRIR47-single-pass 31-node queues；
- [x] 每条 candidate queue child compile selector=`30`、compile reselection=`0`；root source + 30 child
  receipt=`31`；既有 runtime semantic selector 仍为 `30`，不得混入 compile ownership 计数；
- [x] 每条 candidate queue 显式 full replay=`31`、replay selector=`31`，且排除在 production timing 外；
- [x] target tables、selected-CROWN、branch/score/state/ancestry/refinement/bounds/worst lower、31 nodes exact；
- [ ] two-queue compiler-only ratio 实测 `0.936003 > 0.85`，虽改善大于 pooled MAD，门禁失败；
- [ ] clauses 2/3 queue ratio 实测 `1.011205/1.019338 > 0.97`，且改善未超过 pooled MAD；
- [x] typed reconstruction 与同步外层重哈希 semantic tamper fail closed。

### C. Phase B whole query

- [x] Phase A timing 未全过，按预注册禁止启动 Phase B；
- [ ] three fresh CPU8 global queries，floor/rank/selected `[2,3]`、nodes `[31,31]`、worst lower exact；
- [ ] 每轮 child compile selector/reselection=`60/0`、root+child receipt=`62`、既有 runtime semantic
  selector=`60`、显式 full replay/replay selector=`62/62`；
- [ ] trace median ratio vs frozen NRIR45 `<=0.98` 且改善大于 pooled MAD；
- [ ] measured-wall median ratio `<=0.98` 且改善大于 pooled MAD；
- [ ] final 9/9 unknown、artifact/replay/tamper/full suite/Black/mypy/Pylint/DocOps 全过。

## Validation

Phase A 任一 exact、ownership、compiler timing 或 queue timing gate 失败，直接
`VALIDATED-NO-GO` 并禁止 Phase B；门槛不得事后放宽。Phase B 全过时只允许 fixed ResNet2B
property 0 CPU8 target-admission ownership `VALIDATED-REDUCED`。

formal artifact 必须记录 control/candidate 交替顺序、fresh PID、selector 调用来源、receipt/full replay
计数、compiler/queue/whole timing、MAD、exact semantic fields、源码与输入 digest。replay 必须先验校验
所有文件 digest，再重建 receipt 和 full selector semantics；同步改 payload 与 outer digest 后的语义
篡改仍须拒绝。所有工件保持 `performance_claimed=false`。

### Phase A 最终判定

- correctness/parity 与 ownership 通过：candidate 每条 queue compile selector/reselection=`30/0`，
  runtime selector=`30`，receipt/full replay=`31/31`；三轮两条 clause 共 replay 186 份 receipt；
- compiler control/candidate median=`2.739226/2.563922 s`，ratio=`0.936003`，未过 `0.85`；
- clause 2 queue control/candidate median=`10.099396/10.212559 s`，ratio=`1.011205`；
- clause 3 queue control/candidate median=`10.056289/10.250753 s`，ratio=`1.019338`；两条均未过
  `0.97` 且改善未超过 pooled MAD；
- formal hash=`a7561e5187a6e396905d261e739280e39f2c3480e83ba2af0fbe6e3b1ec042ce`；
  replay 与 synchronized outer-rehash tamper probe 通过；
- 结论=`VALIDATED-NO-GO`，Phase B gated off；receipt 机制仅作为未默认启用的 correctness/ownership
  结果保留，不形成 performance、property closure 或 ASPLOS-ready claim。

即使完全兑现 NRIR46 所测约 `1.038 s` ceiling，相对约 `31.32 s` trace 也只有约 3.3% 改善；本轮是
compiler IR ownership 清理，不是公平竞品、10x、GPU、多 workload、property closure 或 ASPLOS-ready
终点。通过后下一主矛盾仍需回到约 `8.6 s` floor 与两条约 `9.9 s` refinement execution，而不是继续
无限细拆验证开销。

## Rollback

- 实现保持 additive，失败时 production 继续使用 NRIR45 prepared compiler；
- 不通过减少 target cap/pass/node/depth、跳过 full replay、缓存跨节点 target、修改 policy、放宽
  deadline 或减少 clauses 换取 timing；
- 若计数过而墙钟不过，可保留 receipt correctness/ownership 机制但不得默认启用或声称 speedup；
- 若 receipt 无法在不削弱 full replay 的条件下消除重选，NRIR47 直接关闭并转 execution math/queue
  attribution，不继续包装 validator fast path。

## Links

- changelog: `gemini_doc/BOUNDFLOW_SINGLE_PASS_TARGET_ADMISSION_RECEIPT_V1_CHANGELOG_2026_08_05.md`
- closure: `gemini_doc/change_2026-08-05_nrir47_phase_a_nogo.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
- predecessor: `gemini_doc/BOUNDFLOW_INTERMEDIATE_REFINEMENT_TEMPLATE_INSTANCE_V1_PLAN_2026_08_05.md`
