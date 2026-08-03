---
status: completed
updated: 2026-08-03T23:07:01Z
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1
stage: s01
---

# Native ReLU-Split BaB Queue v1 Plan

## Goal

- 将 ReLU inactive/active split 从 legacy `ReluSplitState` 提升为 native Bound/Plan/Task/Schedule
  可验证输入，并用 deterministic best-first queue 在 fixed ResNet 上实际形成、批量执行、prune/
  expand 多代节点；和 same-policy serial node evaluation 对齐。

## Scope

- v1 使用 plain CROWN + exact split-constrained IBP，不宣称 α/β optimization；split state 是
  first-class native IR input，不再只藏在 intermediate-bound hash 中。
- source query 使用同一 root input box 与一个 property objective；queue 节点通过 ReLU
  `int8{-1,0,+1}` state 区分，child state 只继承 split constraints，不继承 parent exact bounds。
- queue 冻结 priority、parent-before-child、branch choice、node budget、prune/expand/terminal reason；
  node batch decision 必须驱动真实 child IR stacks，另有 same-policy serial reference。
- fixed ResNet 只跑有界深度/节点预算，结果可为 budget-exhausted；不冒充完整 verifier verdict。
- CPU correctness/control-flow ownership only；无 β、CUDA、latency、memory、TTVerify 或 speedup claim。

## Tasks

1. 扩展 native plain-CROWN Bound IR：typed split inputs、ReLU binding、hash/version、mixed dtype Plan。
2. 扩展 Task/Schedule interpreter：runtime split payload exact binding 与 preactivation feasibility gate。
3. 实现 typed queue/node/branch/prune/termination trace 和 packed/serial node evaluator。
4. 添加 toy complete queue 测试与 fixed ResNet bounded-queue artifact/replay/tamper tests。
5. 运行相邻、全量、静态、DocOps 门禁并同步 ASPLOS claims/limitations。

## Validation

- 默认无 split 的 compiler hash 与 NRIR-8 integration base 一致；NRIR-4 frozen replay 通过。
  NRIR-1 manifest 在本轮前已因后续 Plan 演化陈旧；独立 `3e52408` 快照复现同一旧/新 hash
  差异，不能把该历史不一致误记为 NRIR-9 回归。
- split tensor content/key/order/parent/branch/range 任一篡改 fail closed；child exact state 与 parent
  exact state 不可混用。
- toy packed/serial 完整执行 15 个节点；5 个 packed stacks 与 15 个 serial stacks 的 node bounds、
  branch 和 queue signature 完全一致。
- fixed ResNet 有界执行 7 个节点、3 次 expand/4 个 frontier；packed-4 使用 3 个 native stacks，
  serial-1 使用 7 个；lower/upper max diff 为 `1.8310546875e-04` / `1.220703125e-04`，在
  `atol=rtol=2e-4` 下 allclose，queue/branch/split identity 一致。
- artifact generate/replay 均 exit 0；聚焦 `68 passed`；全量 `577 passed, 37 skipped`（7 条
  既有环境/依赖 warning）。

## Rollback

- additive split-aware compiler/runtime entry；默认 plain-CROWN API 与 NRIR-8 domain runtime 保持兼容。

## Completion boundary

- 只把 native split-state ownership 与 bounded queue control flow 升为 VALIDATED-REDUCED。
- 无 α/β optimization、完整搜索终止、property verdict 或公平 timing 时，不升级 full BaB/C3/performance。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
