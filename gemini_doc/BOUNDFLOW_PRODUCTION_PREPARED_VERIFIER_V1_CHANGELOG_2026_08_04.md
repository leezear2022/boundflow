---
status: completed
updated: 2026-08-04T11:50:00Z
type: changelog
topic: boundflow
slug: production-prepared-verifier-v1
stage: s01
---

# Production Prepared Verifier v1 Changelog

## Summary

- NRIR-27 已以 `VALIDATED-REDUCED` 关闭：新增 production prepared verifier 的一等
  Plan/Task/Schedule、complete-query 集成与三真实拓扑重复 CPU artifact。production 动态 batch
  不构造 audit tensor hash chain，也不再次执行 selected-native oracle；旧 audit 默认行为和 hash
  保持兼容。
- 当前主机为 `torch 2.12.1+cu132`，但 CUDA driver/NVML 不可用；本轮只有 CPU 内部模式对照，
  不产生 CUDA 或竞品 speedup claim。

## Changes

- 新增 `NativeProductionVerifierPlanIR`、typed task module、sequential fail-closed schedule 和四类
  action：validate program、execute optimizer、materialize node results、commit queue results。
- 新增 production ReLU-split queue 与 production complete-query。每个真实 dynamic batch 都拥有
  Plan/Task/Schedule、action phase timing、node/split/parent state lineage 和 optimizer IR identity；
  verdict/conjunction soundness 复用既有门禁。
- 扩展 multiworkload Plan/Task/Schedule 以区分 audit native、production native 与 external
  competitor；旧两方 baseline 的任务数、fresh-process 语义和序列化保持条件兼容。
- 新增 artifact runner，固定 MNISTFC、CIFAR10 ResNet2B、OVAL21，三组交替次序，每组分别执行
  clause-0 audit、clause-0 production 和 full production，共 27 个 fresh workers；replay 会重算
  source/IR/record/log/manifest digest。
- 新增四组测试文件，并补 multiworkload IR 的 production 路径覆盖；NRIR-26 的
  same-target two-pass NO-GO 不被本轮结果覆盖。

## Validation

- clause-0 相同算法、相同输入与预算、三组 fresh-process median audit→production：MNISTFC
  `4.510→3.301 s`（`1.3663×`）、ResNet2B `22.509→9.104 s`（`2.4723×`）、OVAL21
  `5.192→3.578 s`（`1.4511×`）；三者 semantic parity 均通过。
- full production median：MNISTFC `14.834 s`、ResNet2B `60.754 s`、OVAL21 `11.964 s`；均为
  unknown。ResNet 三次均完成 `9/9` clauses，历史 audit deadline 记录为 `2/9`，这里只作
  completion/accounting 证据。
- αβ-CROWN 历史单次参考 `4.312/64.198/4.527 s` 只绑定 provenance 并计算 diagnostic ratio；
  因算法完整性和采样协议不同，`performance_claimed=false`，不得转写为竞品 speedup。
- production full-query 的四类 action 合计只覆盖 execution median 的约 `35%–41%`；untraced
  execution median 为 MNISTFC `8.722 s`、ResNet2B `37.994 s`、OVAL21 `6.676 s`，指向逐
  dynamic batch compilation/preparation 是下一主要瓶颈。
- artifact fresh replay 通过，evidence SHA256=
  `7b650dce529d47c54eeadb168b2311e83a4346b47ffc341d5293b6468c6ac08b`；focused `19 passed`，
  全量 `800 passed, 37 skipped`，Black check、targeted Mypy、Pylint `10.00/10`、
  `git diff --check` 均通过。

## Decisions

- production verifier queue 进入主线并保留 audit mode 作为 replay/correctness oracle；本阶段只关闭
  first-class runtime 与内部 CPU overhead reduction。
- 下一主线为 parametric dynamic-batch compiler：冻结静态 graph/objective/policy
  `PlanTemplate`，把 split/intermediate/parent warm state 降为 `PlanInstance`，用 cache identity
  消除每 batch 重编译，同时保持 fail-closed 语义。
- 当前不产生 GPU、complete competitor、property closure 或 ASPLOS-ready claim；CUDA matrix 在
  可见 GPU 主机上复用冻结协议执行。

## Follow-Ups

- 先对 compile/preparation 内部继续分段，冻结 cache key、template/instance ownership、命中/失配
  trace 与零静默 fallback；再实现缓存并用同一三拓扑协议比较 production-v1/v2。
- 缓存路线只有在 exact semantic parity、三次重复和 full-query E2E 均成立时升级；否则以
  `VALIDATED-NO-GO` 关闭，不用 microbenchmark 替代系统结果。

## Links

- plan: `gemini_doc/BOUNDFLOW_PRODUCTION_PREPARED_VERIFIER_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
