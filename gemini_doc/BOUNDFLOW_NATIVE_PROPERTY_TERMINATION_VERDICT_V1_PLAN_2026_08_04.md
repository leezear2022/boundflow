---
status: completed
updated: 2026-08-04T01:11:23Z
type: plan
topic: boundflow
slug: native-property-termination-verdict-v1
stage: s01
---

# Native Property Termination and Verdict v1 Plan

## Goal

- 关闭 NRIR-12 的 `budget_exhausted/property_status=not_claimed` 边界：在不改变已冻结
  optimized queue trace 身份的前提下，新增可重放的 `verified / unsafe / unknown`
  三态性质结果和 proof/witness contract。

## Scope

- 性质语义固定为单标量 `C f(x) >= threshold`；`verified` 仅当 queue frontier
  为空、无 unresolved terminal，且所有 leaf 均由 sound lower bound 剪枝。
- `unsafe` 必须携带 concrete input witness；必须重执行 primal Task IR，验证 input box、
  ReLU split path 和严格的 `objective < threshold`。
- node/depth budget、不完整 frontier、无法由 witness 关闭的 terminal leaf 一律为
  `unknown`；queue `complete` 不自动等于 property `verified`。
- CPU correctness/control ownership only；不宣称 latency、memory、CUDA 或 speedup。

## Tasks

1. 增加 deterministic concrete Task IR executor，覆盖当前 linear/conv2d/relu/add/mul/
   concat/flatten/reshape/transpose/spec-linear 语义并返回中间 value trace。
2. 定义 verdict/proof/witness schema，绑定 queue trace hash、objective/input hash、resolved/
   unresolved node set 与 concrete tensor digests。
3. 实现 fail-closed verifier：verified 检查剪枝闭包；unsafe 重执行 witness 并检查
   split constraints；其余统一归类 unknown 且记录确定性 reason。
4. 三类 toy 完成性质验证，并将 fixed ResNet NRIR-12 bounded run 升级为
   显式 `unknown` 产物，保留原 optimized queue artifact 不变。
5. 增加 synchronized rehash 后的 input/objective/split/verdict 篡改探针与 fresh replay。

## Acceptance Criteria

- verified toy：所有非 expand 决策均为 `lower_bound_meets_threshold` 剪枝，frontier 为空。
- unsafe toy：witness 在 input box 内，primal/objective 重执行一致，且严格违反性质。
- unknown toy/fixed ResNet：任一 frontier 或 depth terminal 保持 unknown，不允许 claim
  inflation。
- 重放不信任序列化 verdict；从 queue 和 concrete execution 重算决策并逐字比较。
- focused/full pytest、artifact generate/replay、Black、Mypy、Pylint、diff 与 DocOps lint
  全过。

## Rollback

- 新 concrete/verdict runtime 与 NRIR-12 artifact 独立；若三态门禁失败，保留 NRIR-12
  `VALIDATED-REDUCED` 与 `property_status=not_claimed`，不修改原 queue schema/hash。

## Result

- 新增 concrete primal Task IR executor，固定 ResNet 17-op conv/ReLU/residual/flatten/linear
  graph 的 center objective 重执行结果为 `0.8564349412918091`。
- 新增 queue-hash-bound verdict/proof/witness runtime；sound prune 会独立复核 lower，不信任
  prune label；非 root witness 会复核对应 ReLU split path。
- toy matrix 实际产生 verified/unsafe/unknown。固定 ResNet 仍为 7 nodes/4 frontier，
  因此输出 `unknown/node_budget_frontier_open`，没有 claim inflation。
- artifact generate/replay hash=
  `9e3dceed23c8759c910938ba7c9f84caaeb949c8f19b72fab104ce4e1b733405`；聚焦
  `19 passed`；全量 `649 passed, 37 skipped, 7 warnings in 178.52s`；Black/Mypy clean、
  Pylint `10.00/10`、diff check 通过。

状态为 three-state verdict soundness/control ownership `VALIDATED-REDUCED`。下一门禁是
complete verifier query v1：candidate discovery、multi-clause property、timeout/dynamic early stop
和 real verified/unsafe closure；其后再进入端到端性能基线与执行优化。

## Links

- predecessor: `gemini_doc/BOUNDFLOW_NATIVE_OPTIMIZED_RELU_SPLIT_BAB_V1_PLAN_2026_08_04.md`
- changelog: `gemini_doc/BOUNDFLOW_NATIVE_PROPERTY_TERMINATION_VERDICT_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
