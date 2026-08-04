---
status: completed
updated: 2026-08-05T00:55:00Z
type: plan
topic: boundflow
slug: BOUNDFLOW_SHARED_PARAMETRIC_OBJECTIVE_EVALUATOR_V1
stage: s01
---

# Shared Parametric Objective Evaluator v1 Plan

## Goal

- 在 `main@c5ce3e6` 的 NRIR-36 NO-GO 之后，保持 NRIR-31 exact floor、root-lower priority、top-2
  clauses 2/3、dynamic equal-remaining slices、NRIR-34 cap128 ancestral refinement 与 sibling atomic
  commit 不变；把逐 batch audit compiler/re-execution 改为 query-shared parametric optimizer template，
  使第二条 clause 在同一 global 60 秒预算内稳定提交 root + 至少一个 atomic sibling pair。

## Scope

- frozen predecessors：NRIR-28 parametric optimizer template/instance/cache、NRIR-31 floor、NRIR-34
  sibling group/refinement semantics、NRIR-36 selection/slice/aggregate；不得修改其文件或 artifact。
- 新 evaluator 必须有自己的 Plan/Task/Schedule/Batch/Cache IR；template contract 只共享 graph、input
  non-batch shape、objective shape/dtype/device、ReLU layout、optimizer policy 与 provenance。objective
  content、split、intermediate bounds、warm state 与 refinement lineage 必须留在 exact instance。
- query 内固定一个 cache owner；预期第一次 root 为 `miss_compiled`，后续跨 batch 与跨 clause 都只能
  `hit_exact_contract`。cache-key/contract/instance/event/tensor 漂移 fail closed。
- production evaluator 不构造 selected-native audit hash chain，也不做 selected-native re-execution；但
  pilot 必须与 frozen NRIR-34 audit evaluator 对 root + first sibling pair 做 lower/upper、split、branch、
  refinement final bounds 与 selected α/β state parity。
- 本轮不改 top-k、slice policy、cap、node/depth、optimizer steps、candidate search、deadline 或 workload，
  `performance_claimed=false`；首要 claim 是 same-algorithm compiler ownership 与 fixed-deadline committed
  coverage，不是 competitor speedup。

## Tasks

1. [x] profile NRIR-36 clauses 2/3 root/child phase，定位 compiler/re-execution 可回收成本。
2. [x] 新增 shared-parametric ancestral evaluator IR/runtime 与独立 validator，不修改 frozen predecessor。
3. [x] 添加 cache-key/contract/instance/event、source lineage、ordinal、partial group、state/bound drift、
   selected-native re-execution 与 global deadline reset 负向测试。
4. [x] 单次 audit-vs-parametric first-pair parity pilot；不通过则直接 NO-GO，不跑正式三重复。
5. [x] pilot 通过后接入 NRIR-36 top-2 slices并跑一次 coverage pilot；要求两条都至少 3 nodes/1 group。
6. [x] coverage pilot 通过后才跑三 fresh repeats、artifact/replay/tamper/full suite；否则关闭 NO-GO。

## Validation

- 只读 phase profile：program compile=`0.004562 s`、floor=`21.749737 s`。packed plan compile 每 clause
  约 `0.145 s`；每 node batch total=`6.657—7.405 s`，optimizer compile=`0.979—1.103 s`、optimizer
  execute=`1.392—1.423 s`、selected-native compile=`0.417—0.532 s`、selected-native execute=
  `1.295—1.314 s`、child refinement=`0.612—0.632 s`、其余 materialization/hash/validation约
  `2.52—2.63 s`。
- clause 3 的 root + first pair 约需 `6.66+7.34 s`，而本次第二 slice 只有约 `14.1 s`；NRIR-36
  repeat 2 的 `1 node/0 group` 与该临界成本一致。共享 optimizer template 理论上每个 hit 去掉约
  `1.0 s` compile；取消 selected-native audit path 还能去掉约 `1.7—1.8 s/batch`，值得进入正式门禁。
- feasibility 在 first-class/formal pilot 前发现 audit selected-native re-execution 的 upper 对 optimizer
  selected bounds 有 `1.5258789e-5` 绝对漂移；仓库 frozen guard 本来就是
  `torch.allclose(atol=1e-5,rtol=1e-5)`，另有 trace ceiling `2e-3`。因此正式 parity 使用该既有
  relative+absolute guard，而不是误写成纯绝对 `1e-5`；split/branch/refinement final bounds exact，
  alpha max diff `<=1.1e-4`、beta max diff `<=1e-7`。coverage acceptance 为三轮每轮 selected=`[2,3]`
  且 packed nodes 每项 `>=3`、`nodes=1+2*groups`。
- clause 2 root/first-pair feasibility：audit→parametric=`6.672017→0.033680 s`、
  `7.324277→0.677186 s`；cache=`miss_compiled→hit_exact_contract`。lower/split/alpha/beta exact，
  child refinement hashes exact；upper max diff=`1.5258789e-5`，满足上述 frozen allclose guard。
- first-class parity pilot：frozen audit root+pair=`14.073795 s`，NRIR-37=`1.198798 s`；lower、split、
  branch、α、β 与 refinement final-bound hashes exact，upper max diff=`1.52587890625e-5`，既有
  allclose guard 通过；cache=`miss_compiled→hit_exact_contract`。
- top-2 pilot：floor=`20.291832 s`，whole=`50.548707 s`，rank 固定
  `[2,3,4,5,0,8,6,7,1]`，selected=`[2,3]`，两条均提交 `31 nodes/15 groups`；32 个事件为
  1 miss + 31 hits，只有一个 template，pilot hash=
  `c96fff3fa2bc2563b4d46886d69b33f51ac985b19ad80d916309db57fe6cfefa`。
- formal 三 fresh processes：floor=`[21.733539,21.941763,21.925033] s`，whole=
  `[51.996191,52.251681,52.695640] s`，三轮 selected 都是 `[2,3]`、packed nodes 都是
  `[31,31]`、cache miss 都是 1；formal hash=
  `9234dcbe77803e0e7d7e62ca88c62e1b859c95af4ad8e3a19b85c0ab87294b83`。
- artifact replay、11 类 control/compiler 同步重哈希 tamper、Task/Batch commit binding tamper、27
  focused tests、全量 `917 passed, 37 skipped`、mypy clean、Pylint `10.00/10` 通过。closure=
  `VALIDATED-REDUCED`；只关闭 same-algorithm shared compiler ownership 与 fixed-deadline coverage，
  final 仍 9/9 unresolved，`performance_claimed=false`。

## Rollback

- 删除 additive NRIR-37 新文件即可回到 `main@c5ce3e6`；不改 NRIR-28/31/34/36 source 或 artifact。

## Links

- changelog: `gemini_doc/BOUNDFLOW_SHARED_PARAMETRIC_OBJECTIVE_EVALUATOR_V1_CHANGELOG_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
