---
status: completed
updated: 2026-08-03T23:07:01Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1
stage: s01
---

# Native ReLU-Split BaB Queue v1 Changelog

## Summary

- NRIR-9 在 NRIR-8 合并后启动，从 input-box domain batching 推进到 first-class ReLU split state
  与真实 best-first queue/control flow。

## Changes

- 用 DocOps 创建 plan/changelog，冻结 plain-CROWN split、bounded queue、no-performance 边界。
- `BoundDomainConfig(CROWN)`、`WorkloadProfile`、ReLU attrs/value contract 支持 typed
  `int8{-1,0,+1}` split graph input；split value/content hash 进入 ReLU op、Bound module、Plan、Task
  与 Schedule identity。默认无 split 路径保持兼容。
- plain-CROWN lowering/interpreter、Task executor 和 representation compiler 支持 exact runtime
  split binding；key/shape/dtype/device/range/content hash、active/inactive preactivation 约束全部
  fail closed。local split-constrained IBP 与 external verifier provenance 明确分离。
- structured representation rewrite 保留 split side input，只转换四元 affine coefficient state；
  mixed float32/int8 Plan capability、hardware 与 workload linkage 可验证。
- 新增 deterministic widest-ambiguous-ReLU branch、rounded best-first priority、typed node/parent/
  branch/prune/expand/terminal trace、node/depth budget与 replay validator。child 只继承 discrete
  split state，exact interval/CROWN state 每批重新计算，`parent_state_consumed_as_exact=false`。
- node evaluator 将 queue 形成的 child batches 真正编译/执行为 representation-bound
  Bound/Plan/Task/Schedule stacks；same-policy serial 只把 eval batch size 改为 1。
- 新增 toy complete-queue tests 与 fixed ResNet generate/replay artifact、manifest、同步重哈希后的
  parent/branch/IR-stack/bound/claim tamper tests。
- 同步 execution memo、claims map、current status、master plan、README 与 change log。

## Validation

- toy：15 nodes；packed/serial native stacks=`5/15`；bounds、exact-state、branch 和 queue signature
  bitwise/equality闭环。
- fixed ResNet：6/6 ReLU split inputs；7 nodes、3 decisions、4 frontier；packed/serial stacks=`3/7`；
  lower/upper max diff=`1.8310546875e-04/1.220703125e-04`，queue signature 与 split hash 一致。
  batched/serial local IBP exact tensor hash因 CPU batch 数值布局不同而不伪称 bitwise equal。
- fixed artifact generate/replay exit 0，evidence hash
  `0296774ac41be8dc2c80a45357c839761945d9a89c2e395e6056016ad0aefcce`。
- 聚焦新旧 IR/RVIR/queue/artifact tests：`68 passed`。
- Black、Mypy 9 source files clean、Pylint `10.00/10`、`git diff --check` 通过。
- 全量 pytest：`577 passed, 37 skipped`（7 条既有环境/依赖 warning）；首次发现的 external
  αβ split ownership 回归已修复，相关 68-test regression 与全量复跑均全过。

## Decisions

- split state 必须成为 ReLU BoundOp 可解析输入并进入 Task/Schedule/hash，不能只依赖外部 payload。
- parent exact bounds 不可作为 child exact state；只继承离散 split constraints。
- fixed ResNet bounded run 若未完成证明，必须报告 budget-exhausted/unknown，不伪造 property verdict。

## Follow-Ups

- 下一阶段是 native α/β optimization state v1：把优化变量、warm-start validity 和 split constraint
  共同纳入 Bound/Plan/Task/Schedule，而不是把 plain-CROWN bounded queue 冒充完整 αβ-CROWN BaB。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
