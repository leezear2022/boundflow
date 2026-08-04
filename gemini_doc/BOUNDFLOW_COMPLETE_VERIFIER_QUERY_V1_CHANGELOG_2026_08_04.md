---
status: complete
updated: 2026-08-04T01:52:02Z
type: changelog
topic: boundflow
slug: complete-verifier-query-v1
stage: s01
---

# Complete Verifier Query v1 Changelog

## Summary

- NRIR-14 已按 correctness/control `VALIDATED-REDUCED` 关闭：complete query 可执行
  multi-clause conjunction、deterministic PGD candidate search、sound witness replay、
  unsafe short-circuit 与 cooperative deadline。
- 固定 ResNet 九子句全部执行，但 scalarized native lower bounds 过松，9/9 为 unknown；
  不宣称真实性质已闭合，也不声明性能。

## Changes

- concrete Task IR executor 新增可选 autograd-preserving path；默认 no-grad 行为不变。
- 新增 typed candidate-search policy/trace/execution：center-start sign-gradient descent、
  exact box projection、finite-gradient gate、best-candidate concrete replay，明确
  `proof_claimed=false`。
- 新增 complete clause/query trace 与 execution：ascending clause order、conjunction、
  verified/unsafe/unknown 聚合、unsafe suffix skip、deadline pending accounting，以及
  objective/policy/config/search/queue/verdict hash binding。
- 修复 optimized native re-execution trace 的 scale mismatch：实际 execution 继续使用
  `allclose(atol=2e-6, rtol=2e-6)`；序列化 trace 使用独立 `2e-3` 最大绝对差上限并拒绝
  non-finite 数值。固定 clause 6 的合法差为 `6.103515625e-05`。
- 新增 artifact runner、冻结 real/toy evidence 与 claim-inflation/tamper tests。

## Validation

- artifact generate/replay 一致，evidence hash=
  `d17f7d7e960491ad7ef3f33bad41a4cfbf21a9fd5213df3637584b6a753968f1`。
- fixed ResNet best concrete objective values 均为正：最小值
  `0.4761037826538086`；没有伪造 counterexample。九个 native lower bounds 均为负，
  因此 unresolved 9/9。
- toy matrix 覆盖 verified、search-found unsafe、attack-not-found unknown 与 deadline unknown。
- 相关 artifact/runtime 回归 `39 passed`；全量 `670 passed, 37 skipped, 7 warnings`。
- Black（Python 3.10 target）、Mypy clean、Pylint `10.00/10`。

## Decisions

- 性质聚合固定为 conjunction；unsafe 任一 clause 可短路，verified 必须全部 clauses
  闭合。
- attack 只是 candidate discovery，不是 lower-bound proof；最终 unsafe 仍交给 NRIR-13
  concrete replay。
- deadline 是 stage-boundary cooperative control；v1 不声称可以抢占正在运行的 kernel。
- NRIR-14 关闭 control contract，不关闭 fixed ResNet 的 proof tightness。

## Follow-Ups

- 下一阶段先冻结端到端 phase/tightness baseline：量化 candidate、bound optimization、queue、
  verdict 的时间与 proof gap，再决定 dynamic optimizer、branching/tightness 与 batching/cache
  的优化顺序；不得直接从 correctness artifact 推导 speedup。

## Links

- plan: `gemini_doc/BOUNDFLOW_COMPLETE_VERIFIER_QUERY_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
