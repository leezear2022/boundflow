# 变更记录：IR-5B 公平 adaptive policy evaluator

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`98d570e`（IR-5A adaptive query context）
> 状态：policy evaluator contract validated；真实 measured held-out 仍 pending

## 1. 统一 observation/context 契约

新增 `AdaptiveEvaluationContext` 与 `AdaptivePlanObservation`。四种策略必须共享：

- 同一组 immutable PlanInstance hash；
- 同一 legal flag；
- 同一 memory budget；
- 同一 expected query count；
- 同一 exact cache-hit facts；
- 同一 measured latency samples、compile time 和 peak memory。

禁止为不同 policy 使用不同候选集、不同测量或偷偷 fallback。

## 2. 四种策略

- `fixed`：冻结一个 plan；不满足 budget 时如实 infeasible；
- `local_greedy`：只按 local score 选可行 plan；
- `global`：按 predicted TTV（compile miss + runtime × query count）选择；
- `oracle`：只用于评估，按 measured TTV 选择。

每个 outcome 统一输出：

- latency p50/p90/p99；
- time-to-verify；
- peak memory；
- Oracle regret；
- infeasible 不填伪造数值。

summary 按 policy 报告 context count、feasible count、regret p50/p90/max。

## 3. Contract artifact

新增 fresh-process generate/replay artifact，覆盖：

- cold single：global 选择无编译成本 fixed dense；
- repeated：global 选择 compile 后 steady-state 更快的 fused；
- warm cache hit：global 直接选择 fused；
- low memory：fixed 明确 infeasible，global/oracle 选择 structured low-memory。

artifact 写死：

`evidence_scope = synthetic_contract_only_not_heldout_performance`

因此该工件只证明 evaluator 口径、公平性和 replay，不是性能证据，不可用于 ASPLOS 图表。

## 4. 验证与下一步

- evaluator/Plan/compiler 定向：`25 passed`；
- 全量：`468 passed, 1 skipped, 6 warnings`，68.07 s；
- Mypy：0 issues；
- Pylint：10.00/10；
- Black、`git diff --check` clean。

下一步 IR-5C 必须用全新 typed compiler workload 实测候选 latency/compile/peak，冻结
calibration/held-out split，再把 observation 输入本 evaluator。完成前 IR-5 与论文性能
claim 均保持 pending。

