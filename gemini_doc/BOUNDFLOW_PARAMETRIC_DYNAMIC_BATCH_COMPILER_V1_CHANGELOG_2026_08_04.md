---
status: completed
updated: 2026-08-04T12:25:00Z
type: changelog
topic: boundflow
slug: parametric-dynamic-batch-compiler-v1
stage: s01
---

# Parametric Dynamic Batch Compiler v1 Changelog

## Summary

- NRIR-28 已以 `VALIDATED-REDUCED` 关闭。新增真正的 parametric optimizer
  PlanTemplate/PlanInstance、可复用 Task/Schedule、query-scoped exact cache 与 additive v2
  queue/query；三真实拓扑 full-query production-v1→v2 重复 CPU E2E 均显著下降。

## Changes

- 新增 `NativeParametricOptimizerTemplateIR`：静态绑定 primal graph、input/objective tensor
  contract、ReLU layout、policy、provenance/refinement mode；cache key 排除 objective/split tensor
  content，但不排除 shape/dtype/device/semantics。
- 新增 `NativeParametricOptimizerInstanceIR`：逐 batch 绑定 input/objective/intermediate/split/scope/
  initial-state hash、batch size 与 warm-start kind；exact runtime tensor rebinding在执行前拒绝。
- template compiler 一次 lower reusable Task/Schedule；instance binder 只计算动态 forward/scope/
  warm state，不重建 replay-grade five-layer source compilation，也不做冗余 initial evaluation。
- 新增 query-scoped cache event IR、template/cache trace、parametric production queue 和 complete
  query；同一 cache 跨 clause 与 root/child batches 共享，miss/hit/compile/bind/execute phase 可审计。
- 新增 formal artifact runner，固定三 workload、三组交替 v1/v2、18 个 fresh workers，重算
  source、template/task/schedule、event/instance、semantic summary、log 与 manifest digest。
- 所有实现均在 additive v2 文件；NRIR-27 code revision 覆盖文件零修改，旧 artifact 继续 replay。

## Validation

- cProfile 单次 MNISTFC clause-0 production 诊断：optimizer program compile cumulative约
  `0.823 s`，prepared validation约 `0.567 s`，production plan construction约 `0.284 s`；主要成本
  来自递归 IR validation/stable hashing 与 replay-grade source compilation。该诊断只用于选择
  路线，不是性能 claim。
- NRIR-27 正式数据：full production untraced execution median 为 MNISTFC `8.722 s`、ResNet2B
  `37.994 s`、OVAL21 `6.676 s`，分别约占 execution 的 `65.0%/64.1%/63.5%`。
- 正式 production-v1 raw E2E：MNISTFC `14.722/14.807/15.014 s`、ResNet2B
  `60.870/61.239/61.280 s`、OVAL21 `12.325/13.021/13.085 s`；parametric-v2 raw E2E：
  `3.419/3.514/3.456 s`、`6.209/6.142/6.294 s`、`3.718/3.736/3.704 s`。
- median internal full-query speedup=`4.284938×/9.862981×/3.502396×`。三组每条的 solver
  status、clause accounting、logical queue、node count、selected state hash 与 root bounds 均
  exact/allclose；不是减少工作量造成的 timing 差异。
- 每次 query template/miss 都是 `1/1`；MNISTFC、ResNet、OVAL 的 instances/hits 分别固定为
  `19/18`、`27/26`、`11/10`。parametric execution median 分别约 `2.001/4.714/2.005 s`。
- artifact fresh replay 通过，evidence hash=
  `117fcecf8e089c16f4275abb97292039790bae75bc4b518ae699bc9ac432ce97`；NRIR-27 replay
  通过；focused `22 passed`、全量 `818 passed, 37 skipped`、Black、Mypy、Pylint `10.00/10`
  通过。

## Decisions

- 使用 additive v2 文件保持 NRIR-27 代码 revision 和 frozen artifact 可重放；不直接改写 v1
  compiler/queue/query。
- 正式性能门禁比较相同 production 算法的 v1/v2 full query，而不是 audit 或外部 competitor。
- 预注册的三 workload median 全部严格改善门禁成立，因此关闭为 internal CPU performance
  `VALIDATED-REDUCED`；保留 v1 作为 audit-oriented baseline，v2 作为下一轮系统执行路径。
- v2 仍把所有 property 判为 unknown。下一主线从继续削减 compiler 微开销切到 fixed-wall-clock
  typed BaB depth/node scaling，检验系统加速能否转化为更高搜索覆盖与 property closure。

## Follow-Ups

- 冻结 7/31/127 node、depth 2/4/6 与统一 60 秒 deadline 的 search-budget Plan/Task/Schedule；先做
  三拓扑 coverage/closure 探针，再决定是否生成下一正式 artifact。
- 在可见 GPU 主机复用同一 v1/v2 协议；当前不得外推 CPU speedup 到 CUDA 或外部 verifier。

## Links

- plan: `gemini_doc/BOUNDFLOW_PARAMETRIC_DYNAMIC_BATCH_COMPILER_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
