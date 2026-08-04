---
status: completed
updated: 2026-08-04T12:25:00Z
type: plan
topic: boundflow
slug: parametric-dynamic-batch-compiler-v1
stage: s01
---

# Parametric Dynamic Batch Compiler v1 Plan

## Goal

- 把 NRIR-27 每个 dynamic node batch 的 exact optimizer 编译拆成可复用的静态
  `PlanTemplate` 与逐 batch `PlanInstance`：graph、tensor contract、optimizer policy、Task/Schedule
  只编译一次；input/objective/split/intermediate/parent warm state 作为 exact instance 参数绑定。
- 在不减少搜索、optimizer steps、clause 数或语义门禁的前提下，降低 production full-query CPU
  E2E；以 production-v1 对 production-v2 的三真实拓扑重复对照决定 REDUCED/NO-GO。

## Scope

- 新增独立 parametric optimizer IR/runtime 与 query-scoped cache；NRIR-27 冻结源码和 artifact runner
  不修改，旧 artifact 必须继续 fresh replay。
- template key 只包含真正静态且可检查的 graph/input/objective contract、ReLU layout、policy、
  provenance/refinement mode；所有动态 tensor content 都进入 instance IR，不能依赖弱 key 或 Python
  object identity 静默命中。
- cache miss 可编译新 template；contract/key/content 不一致必须 fail closed，不允许回退 NRIR-27
  compiler 后仍记作 hit。v1 只做 query-local、单线程、内存缓存，不声明跨进程持久化或 CUDA graph。
- 数值与逻辑结果必须和 production-v1 对齐；GPU、αβ-CROWN speedup、complete-property closure 与
  ASPLOS-ready 不在本阶段 claim 内。

## Tasks

1. [x] 定义 parametric optimizer PlanTemplate/Task/Schedule 与 PlanInstance IR；静态 contract、动态
   bindings、template/instance hash、cache key 和 execution ownership 均一等可序列化。
2. [x] 实现 template compiler、query-scoped cache 与 exact instance binder；记录 miss/hit、compile/
   bind/validate phase 和拒绝原因，覆盖 objective/input/split/intermediate/policy/layout tamper。
3. [x] 实现 parametric optimizer executor；复用同一 fixed-step Task/Schedule 和数值 kernel，跳过
   NRIR-27 为每个 batch 重建 replay-grade five-layer source compilation及冗余 initial evaluation。
4. [x] 新增 additive parametric production queue/complete-query，把同一 cache 跨 clause、root/child
   batch 共享；保留旧 production-v1 作为 correctness/performance baseline。
5. [x] 增加 IR、cache、instance、queue/query parity、deadline、negative 与 NRIR-27 historical replay
   测试；运行 focused/full pytest、Black、Mypy、Pylint、diff 与 DocOps gate。
6. [x] 生成 MNISTFC/ResNet2B/OVAL21 三组交替次序 fresh-process full-query v1/v2 artifact；冻结 raw
   E2E、median/p90、cache hit/miss、phase、semantic parity、source/IR/log/manifest digest 与 replay。

## Outcome

- 三种 workload 的 full-query production-v1→parametric-v2 median E2E 分别为：MNISTFC
  `14.807→3.456 s`（`4.2849×`）、ResNet2B `61.239→6.209 s`（`9.8630×`）、OVAL21
  `13.021→3.718 s`（`3.5024×`）；每项均为三组交替次序 fresh process，严格通过预注册门禁。
- 每次完整 query 只编译一个 template：MNISTFC `19 instances=1 miss+18 hits`、ResNet2B
  `27=1+26`、OVAL21 `11=1+10`；三次重复计数完全一致。template compile median 约 3 ms，
  instance bind 总 median 为 `31/106/24 ms`。
- v1/v2 的 solver status、completed/pending/unresolved clauses、logical queue、node 数、selected state
  hash 与 root lower/upper 全部 exact/allclose；三类 property 仍为 unknown。
- artifact evidence SHA256=
  `117fcecf8e089c16f4275abb97292039790bae75bc4b518ae699bc9ac432ce97`；NRIR-27 historical
  artifact 继续 fresh replay，focused `22 passed`，全量 `818 passed, 37 skipped`。
- NRIR-28 以 internal full-query CPU performance `VALIDATED-REDUCED` 关闭；下一路线把节省的
  wall-clock 预算投入 typed depth/node search scaling。GPU、竞品 speedup、complete-property 与
  ASPLOS-ready 仍 pending。

## Validation

- 每个 full query 的 template compile count 必须为 `1`，后续所有 batch 为 exact hit；instance 数
  必须等于真实 queue batch 数，template/instance/task/schedule 与 runtime action 一一对应。
- production-v1/v2 的 completed/pending/unresolved clauses、verdict、logical domains、lower/upper、
  selected state 与 queue accounting exact/allclose；任何 cache contract 或 dynamic binding tamper
  在执行前拒绝。
- 性能 claim 要求每 workload 至少三组 fresh process、交替次序；只有三种 workload 的 v2 median
  full-query E2E 都严格低于 v1，才允许 `VALIDATED-REDUCED`，否则该 cache 策略
  `VALIDATED-NO-GO`。不得用 compile microbenchmark 代替系统结果。
- NRIR-27 artifact fresh replay、focused/full pytest、Black、targeted Mypy、Pylint、
  `git diff --check`、artifact replay、`dol lint --soft` 全过。

## Rollback

- parametric v2 是新入口且显式 opt-in；删除新增 IR/runtime/runner/tests 即完整回退到 NRIR-27。
  cache/instance 门禁失败时必须报错，不能静默调用 v1 或返回未绑定结果。

## Links

- changelog: `gemini_doc/BOUNDFLOW_PARAMETRIC_DYNAMIC_BATCH_COMPILER_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_PRODUCTION_PREPARED_VERIFIER_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
