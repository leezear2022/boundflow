---
status: completed
updated: 2026-08-04T00:16:51Z
type: plan
topic: boundflow
slug: native-optimized-relu-split-bab-v1
stage: s01
---

# Native Optimized ReLU-Split BaB v1 Plan

## Goal

- 关闭 NRIR-11 的 single-node 边界：让 NRIR-9 best-first ReLU-split queue 的每个 node batch 都由
  NRIR-11 optimizer Plan/Task/Schedule 求得 selected alpha/beta state，再经 NRIR-10 native
  Bound/Plan/Task/Schedule 执行，形成从 queue control 到 optimized bound execution 的连续 IR 链。

## Scope

- 新增独立 optimized queue runtime，不改变 NRIR-9 plain-CROWN API/artifact identity。
- root 使用 policy initialization；child 只允许从已执行 parent selected state 构造同 batch layout 的
  monotonic-refinement warm initialization。parent state 不得作为 child exact state。
- packed/serial 必须共享同一 optimizer policy，并保持 node lineage、branch、decision、selected
  bounds/state 一致；batch layout 只改变 stack count。
- CPU correctness/control ownership only；fixed step，无 dynamic early stop、complete property verdict、
  CUDA、latency、memory、OOM、Pareto 或 speedup claim。

## Tasks

1. 新增 optimized queue evaluation/stack/trace contract，绑定 node、parent state、warm decision、
   optimizer Plan/Task/Schedule/trace 与 selected native compiler hashes。
2. 实现 parent states → batched warm state 的语义重绑定，并验证每个 child split 都是 monotonic
   refinement、alpha/beta payload 未漂移。
3. 对每个 node batch 编译/执行 optimizer Schedule，随后编译/执行 selected-state native stack；
   两次 bounds 必须一致。
4. slice batched selected state 为 per-node scoped state，供后续 children warm start；逐节点 state/
   parent hash、gradient、projection、evaluation/best iteration 进入 trace。
5. toy complete queue 与 fixed ResNet bounded queue 对 packed/serial；加入 lineage/warm/hash/order/
   state/native-reexecution tamper tests 和 artifact replay。

## Validation

- toy complete queue 必须结束于 `complete`，packed/serial node/decision/state/bounds 一致；所有非 root
  node warm kind 为 `monotonic_split_refinement`。
- fixed ResNet 必须重现确定性 bounded frontier；每个 active-split batch 至少存在 beta gradient，
  每个 selected state 由已评估 iteration 选出并经 native compiler执行。
- optimizer action count、native task count、stack count、node coverage 和 parent→child hash lineage
  全部闭合；同步重哈希后的语义篡改仍 fail closed。
- focused/full pytest、artifact generate/replay、Black、Mypy、Pylint、diff 与 DocOps lint 全过。

## Rollback

- 新 runtime 与 artifact 独立于 NRIR-9；若 optimized queue 门禁失败，保留 NRIR-9/10/11 各自
  VALIDATED-REDUCED 状态，不修改 plain queue 默认路径或升级论文 claim。

## Result

- 新增独立 optimized queue runtime；每个 node batch 先执行 8-action NRIR-11 optimizer Schedule，
  再执行 21-task NRIR-10 selected-state native stack。
- toy complete queue 为 15 nodes；packed/serial 为 5/15 stacks，node/decision/split/bounds/state hash
  全部一致，所有 child 均为 monotonic-refinement initialization-only。
- fixed ResNet bounded queue 为 7 nodes/3 expands/4 frontier，packed/serial 为 3/7 stacks；bounds
  max diff=`1.220703125e-04/1.8310546875e-04`，alpha/beta tensor max diff=
  `4.172325134277344e-07/7.450580596923828e-09`。exact batch-layout state/scope hash 不伪称相同。
- packed child stacks 的 beta gradient L1 为 `20.047863006591797/42.26581954956055`；每个 selected
  state 对 native re-execution lower/upper max diff 均为 `0.0`。
- artifact generate/replay hash 为
  `e813826c8fe74161505ab2379b37fa67247fd40c3bd0cb8f82b77880ce403787`；聚焦
  `18 passed`，全量 `630 passed, 37 skipped`，静态门禁全过。

状态为 optimized queue integration/control ownership `VALIDATED-REDUCED`。下一门禁是 sound property
termination/verdict v1：只有 frontier 完整关闭才允许 verified，unsafe 必须带可执行 concrete witness，
预算或深度未闭合必须保持 unknown。

## Links

- predecessor: `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZER_STEP_SCHEDULE_V1_PLAN_2026_08_04.md`
- queue base: `gemini_doc/BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1_PLAN_2026_08_04.md`
- changelog: `gemini_doc/BOUNDFLOW_NATIVE_OPTIMIZED_RELU_SPLIT_BAB_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
