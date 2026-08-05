---
status: validated-no-go
updated: 2026-08-05T00:05:57Z
type: plan
topic: boundflow
slug: cross-axis-verification-batch-schedule-v1
stage: s01
---

# BoundFlow Cross-Axis Verification Batch Schedule v1

## Goal

- 将 NRIR-42 中按 `clause → node → candidate` 顺序发射的独立 lower-bound 工作，lower 为显式
  `clause × node × candidate × branch-value` ragged batch Schedule；
- 保持每条 clause 自己的 queue 顺序、objective、selected optimizer state、split、refinement 与
  objective-branch policy 不变，只改变已经 ready 的独立张量工作的装箱与发射方式；
- 在固定 ResNet2B property 0、CPU8 上把 NRIR-42 的 62 次逐节点 scorer lower 发射压缩为最多
  16 次 ready-set 发射，并验证端到端 three-repeat whole-query median ratio `<=0.80`；
- 该阶段仍是内部生产准入，不直接形成 ASPLOS 性能 claim。只有后续公平竞品重跑与多工作负载扩展
  才能升级论文结论。

## Scope

### 基线

- integration base：`main@34ca6c6`；NRIR-42 功能 merge：`8969064`；
- frozen Phase-A hash：`0d310c2ffc96844648a83f9921bc7f353ec8425986bccb36f75e6d1cd2b25b58`；
- frozen Phase-B hash：`7274e834b3bf08a9e138fa3284b70222620cf3c571395331e1a87ed5fee7d759`；
- frozen workload：VNN-COMP 2021 CIFAR10 ResNet2B property 0，CPU、8 Torch threads；
- NRIR-42 whole elapsed：`57.175184/57.697757/58.114412 s`，floor elapsed：
  `21.788137/21.894675/22.100945 s`，selected clauses 均为 `[2,3]`。

### 唯一变量

唯一允许变量是 ready work 的 batch Schedule：

1. 同一 evaluator batch 内，把 1 个 root 或 2 个 sibling nodes 的 48-candidate child lowers 合并；
2. 两条 selected clauses 同一逻辑 round 均 ready 时，把它们的 root 或 sibling work 合并；
3. 通过 typed ragged segment 把联合结果拆回原 clause/node/candidate，并按各自 queue 原顺序提交。

不得修改：NRIR-31 floor、clause ranking/selection、objective branch candidate policy/reduce policy、
optimizer steps/α/β、intermediate refinement、queue priority、`31 nodes/depth 4`、cache policy、threshold、
60 秒 global deadline、数值 dtype 或模型/属性。

### IR 所有权

- `CrossAxisVerificationBatchPlanIR` 拥有 axes、容量、program/capsule identities 与 ragged segments；
- `CrossAxisVerificationBatchInstanceIR` 只绑定本轮 ready clauses/nodes，不重新枚举 candidates；
- `CrossAxisVerificationBatchTaskIRModule` 固定
  `ADMIT_READY_SET → PACK_CHILD_DOMAINS → EXECUTE_LOWER_BATCH → SEGMENT_REDUCE → COMMIT_BRANCHES`；
- `CrossAxisVerificationBatchScheduleIR` 只调度 Task，不隐式重建 Plan 数据；
- 每个 segment 必须携带 clause ordinal、node ID、objective/capsule/state hash、candidate offset/count、
  child-domain offset/count；segment 必须连续、无重叠、全覆盖；
- 不允许 `Any`、静默 fallback、丢弃 partial pack 或跨 clause 共享 queue 状态。

## Tasks

### A. Typed IR 与 sibling scorer pack

- [x] 新增 cross-axis Plan/Instance/Task/Schedule/Trace IR 及 stable hash、fail-closed validate；
- [x] 编译每节点 NRIR-42 capsule 后，按 ragged segment 合并 child splits、relu-pre、α、β、objective；
- [x] 一次 `_evaluate_state` 后按 segment 还原每节点 child lower、score 与 selected branch；
- [x] root 单节点和 sibling 双节点都走同一 Schedule，不保留未记录的旁路。

### B. 两 clause ready-set coordinator

- [ ] 将 selected clauses `[2,3]` 的 production queue 改写为两个独立 queue state + 单一 ready-set scheduler；
- [ ] root round 合并 2 个 clauses；每个后续 round 最多合并 2 clauses × 2 sibling nodes；
- [ ] optimizer node evaluation 与 objective scorer evaluation 都必须显式记录 clause/node segments；
- [ ] 任一 clause prune/terminal/budget/deadline 后允许 ragged 缩窄，但不得改变另一 clause 的 queue 次序；
- [ ] 原 ordinal aggregate、verdict、deadline 和 evidence owner 保持 NRIR-42 语义。

Phase B 因 Phase A timing gate 失败而按预注册整体 gated off；以上项目未实现、未运行。

### C. 正式证据

- [x] Phase A：三次 counterbalanced NRIR-42/new paired scorer/queue 运行；
- [x] Phase B：只有 Phase A 全过才执行三次 fresh global-60s whole query；本轮条件不成立，未启动；
- [x] generate/replay 必须重建 typed Plan/Instance/Task/Schedule/Trace，并逐段重算语义；
- [x] 加入 Phase-A synchronized outer-rehash segment offset/objective owner/launch count 与 typed
  capsule/node owner tamper；
- [ ] Phase-B partial commit/deadline tamper：Phase A 未过，按门禁未实现；
- [x] targeted、predecessor-inclusive、全量 pytest、Black、mypy、Pylint 与 DocOps 门禁。

## Validation

### Phase A acceptance

必须同时满足：

1. 两条 clause 的 31 个节点均覆盖；candidate table、selected branch、全部 48-entry score、child lower、
   queue lower/upper、split、α/β、refinement 与 NRIR-42 对应节点一致；
2. 每个 sibling pack 的两个节点由同一 typed launch 覆盖，所有 segment 连续、无重叠、无遗漏；
3. 两 clause 合计 scorer lower launch 从 `62` 降到 `<=32`；进入跨 clause coordinator 后降到 `<=16`；
4. 三次 paired queue 的每条 clause new/NRIR-42 median ratio 均 `<=0.85`，改善严格大于 pooled MAD；
5. 任一 correctness、ownership、launch 或 timing gate 失败，Phase A 以 `VALIDATED-NO-GO` 关闭，
   不启动 Phase B。

### Phase B acceptance

必须同时满足：

1. floor 9 clauses、rank 与 selected `[2,3]` 和 NRIR-42 frozen evidence 一致；
2. 两条 clause 每轮均提交 `31 nodes/15 groups/31 capsules`，没有 partial/reset/recompile/evidence omission；
3. node/branch/score/queue/split/state/refinement 逐节点等价；若底层 batch kernel 不能 bitwise exact，
   tensor max diff 必须 `<=3.2e-6`、selected branch 与 queue order 必须 exact，且 lower 不得非保守上移；
4. optimizer ready-set launch `32→<=16`，scorer lower launch `62→<=16`；至少 15 个 pack 的 clause width=2、
   node width=4、candidate child-domain width=384；
5. three fresh whole-query 每轮 `<=45 s`，且 new/NRIR-42 whole median ratio `<=0.80`，改善大于 pooled MAD；
6. formal replay 与全部 tamper、测试、静态检查通过。

通过后仅能声明：fixed ResNet2B property 0 CPU8 cross-axis production admission
`VALIDATED-REDUCED`、`performance_claimed=false`。不得声明 property closure、GPU/multi-workload、
公平竞品数量级领先或 ASPLOS-ready。

## Phase A 正式结果

- 6 个 clause-repeat old/new 组的 queue、branch、48-entry scores、child lower、selected state、split、
  α/β 与 refinement 全部 exact；typed segment/capsule owner 和连续覆盖门禁通过；
- 每条 31-node queue 的 scorer physical launches 从 `31→16`，node widths 固定为
  `[1,2,…,2]`，launch gate 通过；
- clause 2：NRIR-42/cross-axis median=`12.821506/13.477127 s`，ratio=`1.051134`，
  median delta=`-0.655621 s`；
- clause 3：NRIR-42/cross-axis median=`13.004753/13.584418 s`，ratio=`1.044573`，
  median delta=`-0.579665 s`；
- 两条 timing gate 均失败；减少发射没有转化为 CPU 加速，192-domain scorer batch 比两个
  96-domain launches 更慢；
- formal hash=`692b9e273661fce9f12129e134550547afa4023361e2a79d751c437c92f30390`，
  decision hash=`1054192454234aa151db494c98e399f1ad44cf6a93e4790f7ce0590da45142bb`；
- targeted `10 passed`，全量 `968 passed, 37 skipped`，Black/mypy/Pylint `10.00/10` 通过。

## 决定

NRIR-43 以 `VALIDATED-NO-GO` 关闭，`performance_claimed=false`。Phase B 不启动，NRIR-42
production 路径不被替换。下一变量转为 NRIR-44 Root-Projection Floor Schedule：消除 floor 中
ranking consumer 不需要的九条深层 queue work，而不是继续扩大 CPU domain batch。

## Rollback

- 实现只能新增 additive IR/runtime/runner/test；NRIR-42 frozen 源码与 artifact 不改；
- Phase A 未过：冻结失败证据并回到 NRIR-42；
- Phase A 过、Phase B 未过：只保留 scorer batch 机制结论，不接入 production claim；
- 内存、shape、dtype、objective owner 或 deadline 不满足时 fail closed，不静默退回串行路径；
- 不以放宽 node/depth/deadline、减少 candidate 或降低精度换取 timing 通过。

## Links

- changelog: `gemini_doc/BOUNDFLOW_CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_V1_CHANGELOG_2026_08_05.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SCORER_OWNERSHIP_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
