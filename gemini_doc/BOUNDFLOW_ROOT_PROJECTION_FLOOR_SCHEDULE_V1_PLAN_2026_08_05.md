---
status: validated-reduced
updated: 2026-08-05T01:04:00Z
type: plan
topic: boundflow
slug: root-projection-floor-schedule-v1
stage: s01
---

# BoundFlow Root-Projection Floor Schedule v1

## Goal

- 将 NRIR-42 whole-query floor 中 9 条顺序 `n31d4` objective queues，按 downstream consumer
  contract 投影为只产生 root evaluation 的 `n1d0` Schedule；
- 保持 baseline、shared/objective refinement、每条 root lower/upper/branch、ranking、selected `[2,3]`、
  top-2 NRIR-42 production queues 与 global-60s deadline 不变；
- fixed ResNet2B property 0 CPU8 上，把 floor median 压到 `<=11 s` 且 old/new ratio `<=0.50`，
  whole-query 每轮 `<=48 s` 且 median ratio `<=0.82`；
- 明确这是 sound-but-less-complete 的 ranking-floor specialization：非 top-2 clause 不再在 floor
  阶段尝试深层证明，不能外推一般 complete-verifier 等价。

## Scope

### 基线与诊断

- integration base：`main@d9d76da`；NRIR-43 merge：`2d245d6`；
- frozen NRIR-42 Phase-B hash：
  `7274e834b3bf08a9e138fa3284b70222620cf3c571395331e1a87ed5fee7d759`；
- NRIR-42 floor=`21.788137/21.894675/22.100945 s`，whole=
  `57.175184/57.697757/58.114412 s`；
- 单次只读 breakdown：baseline `4.8227 s`，shared compile/execute `0.2165 s`，9 条 objective
  refinement compile/execute 约 `2.69 s`，9 条 deep objective query 约 `13.88 s`；
- 路线冻结前 probe：同一 refinement 下 9 条 `n1d0` root queries 合计 `0.789371 s`，9/9
  root lower/upper/branch 与原 `n31d4` root exact。该 probe 只用于选路线，不是正式 claim。

### 唯一变量

只改变 floor objective-query Schedule 的 result liveness：

- 原路径：9 clauses × `n31d4`，最多 279 evaluations；
- 候选路径：9 clauses × `n1d0`，恰好 9 root evaluations；
- downstream ranking 本来就只读取每条 accepted child 的 root lower margin；
- selected top-2 后续仍执行 NRIR-42 的完整 `31 nodes/15 groups/31 capsules` production queue。

冻结不改：baseline n7d2、PGD/search、shared/objective refinement policy、optimizer policy、objective
branch policy、rank tie-break、top-2、production node/depth/queue/cache、threshold、dtype、CPU threads、
模型/属性和 global deadline。

### Soundness 边界

- root lower 达阈值仍可 verified；反例仍必须由既有 candidate-search 证据产生；否则标为 unknown；
- 不得把 old floor 深层可能得到的 verified 结论伪装成 projected floor 已证明；
- current frozen workload 要求 old/new floor 都是 9/9 unknown，最终 whole status 仍 unknown；
- specialization 必须由 typed consumer contract 显式启用，不得静默替换一般 complete-verifier floor。

### IR 所有权

- `RootProjectionFloorPlanIR` 绑定 source floor、consumer fields、full/projected budgets 与 9 clause owners；
- `RootProjectionFloorInstanceIR` 绑定本轮 objective/refinement hashes，不改变 Plan policy；
- `RootProjectionFloorTaskIRModule` 固定
  `ADMIT_SOURCE → ANALYZE_CONSUMERS → EXECUTE_BASELINE → REFINE_OBJECTIVES → EXECUTE_ROOT_PROJECTIONS → RANK_ROOTS → EMIT_FLOOR`；
- `RootProjectionFloorScheduleIR` 证明 deep queue result 不活跃，并记录 `279→9` evaluation projection；
- 所有 projection row 必须绑定 original clause ordinal、objective、threshold、refinement、root result
  与原 full-query root reference；禁止 `Any`、无证据 fallback 或事后改 budget。

## Tasks

### A. Root projection IR/runtime

- [x] 新增 Plan/Instance/Task/Schedule/Trace 与 consumer-liveness validate；
- [x] additive clone NRIR-31 objective floor，只将每条 child query queue config 投影为 n1d0；
- [x] 保留 baseline、refinement、deadline、aggregate、原 ordinal 与 trace owner；
- [x] 对 budget/consumer/trace 转移做 fail-closed 单测。

### B. Phase A floor formal

- [x] three fresh counterbalanced old/projected floor runs；
- [x] 9/9 objective/refinement/root lower/upper/branch exact，rank/selected exact；
- [x] old evaluations `9×31`、projected `9×1`，无隐藏 deep queue；
- [x] typed replay 重建 Plan/Instance/Task/Schedule/Trace 与每条 projection row；
- [x] synchronized outer-rehash consumer/budget/evaluation-count tamper fail closed。

### C. Phase B whole query

- [x] 只有 Phase A 全过才把 projected floor 接到 NRIR-42 production multi-clause runtime；
- [x] three fresh global-60s whole queries；
- [x] selected `[2,3]`、两条 `[31,31]` nodes、worst-active lower 与 NRIR-42 exact；
- [x] final aggregate 9/9 unresolved、cache/capsule/deadline/evidence 完整；
- [x] replay/tamper、targeted 与静态门禁；全量结果见关闭记录。

## Validation

### Phase A acceptance

必须同时满足：

1. baseline semantic hash、9 条 objective/refinement lineage 与 9 个 root lower/upper/branch exact；
2. floor final statuses 均为 unknown，rank=`[2,3,4,5,0,8,6,7,1]`、selected=`[2,3]` exact；
3. objective-query evaluations 从 `279→9`，projected 每条恰好 1，深层 decision/child node 为 0；
4. three paired projected floor 每轮 `<=11 s`，projected/NRIR-42 median ratio `<=0.50`，改善大于
   pooled MAD；
5. 任一 gate 失败即 `VALIDATED-NO-GO`，不得启动 Phase B。

### Phase B acceptance

必须同时满足：

1. three fresh queries 每轮 floor `<=11 s`、whole `<=48 s`；
2. projected/NRIR-42 whole median ratio `<=0.82`，改善大于 pooled MAD；
3. selected clauses、31-node coverage、branch/score/queue/state/refinement/worst lower 与 NRIR-42 exact；
4. 没有 partial/reset/recompile/evidence omission，formal replay 与 tamper 全过；
5. full pytest、Black、mypy、Pylint 与 DocOps 全过。

通过后仅允许 fixed ResNet2B property 0 CPU8 ranking-floor + production admission
`VALIDATED-REDUCED`、`performance_claimed=false`。公平 competitor、GPU、多工作负载、property closure
与 ASPLOS-ready 仍需后续独立门禁。

## Results

- Phase A 三轮 old/projected floor elapsed 分别为
  `[24.235039,22.859521,24.252771] / [9.739498,10.740998,9.876515] s`；projected
  median ratio=`0.407530`，最大单轮=`10.740998 s`，改善超过 pooled MAD；
- 三轮 baseline、9 条 objective refinement、root lower/upper/branch、9/9 unknown、rank=
  `[2,3,4,5,0,8,6,7,1]` 与 selected=`[2,3]` exact，objective evaluations=`279→9`；
- Phase A formal/decision hash=`ecb553d8…ff0fe` / `72840c37…fabdd`；
- Phase B floor=`[8.538814,8.622447,8.648849] s`，whole execution trace=
  `[43.571040,44.144990,44.095736] s`，相对 frozen NRIR-42 whole median ratio=
  `0.764254`；每轮 clauses 2/3 均为 `[31,31]` nodes、15 groups，worst-active lower exact 为
  `-35.530926/-30.258448`；
- Phase B formal payload hash=`2f22d44f…7272d9`；两个 formal replay 与同步外层重哈希 budget/
  deadline tamper 均 fail closed；targeted `11 passed`、全量 `979 passed, 37 skipped`，
  `performance_claimed=false`。

## Rollback

- 全部实现 additive；NRIR-31/42/43 frozen runtime 与 artifact 不改；
- Phase A 未过则冻结证据并继续使用 NRIR-42 floor；
- Phase A 过、Phase B 未过则只保留 root-projection mechanism，不替换 production；
- 不以跳过 refinement、修改 rank/top-k、减少 candidate、放宽 deadline 或降低 dtype 换取通过；
- 一般 complete-verifier 调用若未声明 ranking-only consumer contract，必须继续使用完整 floor。

## Links

- changelog: `gemini_doc/BOUNDFLOW_ROOT_PROJECTION_FLOOR_SCHEDULE_V1_CHANGELOG_2026_08_05.md`
- predecessor: `gemini_doc/BOUNDFLOW_CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
