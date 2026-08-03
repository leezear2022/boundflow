# 变更记录：IR-5A adaptive PlanInstance query context

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`a3552d3`（IR-4 validated-reduced closure）
> 状态：IR-5A mechanism validated；IR-5 performance/evaluation 尚未关闭

## 1. 证据审计

仓库当前 `artifacts/phase7a-pr12/` 只有 4 个 kernel-foundation 文件，不含 PR-11/12 的
planner raw records。历史 1,416 executions / 23 feasible 只能保留为历史汇总，不能作为
新 typed IR-5 的逐记录输入或 held-out 结果。

本轮还确认现有 typed selector 的 compile/setup cost 只作为 runtime latency 之后的
tie-break，无法表达 cold/warm/repeated query distribution，也无法让 exact compiled-cache
hit 改变选择。

## 2. PlanSelectionContext

新增 query-time typed context：

- `query_distribution_id`；
- `expected_query_count`；
- canonical sorted `cached_artifact_keys`。

selector 的新目标为：

`predicted_runtime_latency + (uncached_compile_cost + setup_cost) / expected_query_count`

随后依次按 peak bytes、uncached compile cost 和 candidate IDs 确定性 tie-break。deadline
也对同一 amortized latency 生效，不再只看 steady-state runtime。

PlanInstance identity 与 provenance 现在包含：

- distribution ID；
- expected query count；
- exact cached artifact keys；
- amortized selection latency；
- uncached compile cost；
- selector v2 identity。

因此同一 BoundModule/PlanTemplate 在 cold、repeated、warm-cache context 下会产生不同、
可重放、可审计的 PlanInstance hash。

## 3. Compiler runtime 动态上下文

新增 `CompilerRuntimeContext`：

- available memory；
- memory budget；
- optional deadline；
- `PlanSelectionContext`。

每个 typed compiler query 可覆盖 runtime 默认 context。完整 context 同时进入：

- selector 参数；
- query bucket identity；
- compiled Plan/Task cache namespace；
- result 暴露的 typed PlanInstance。

这允许同一 runtime 进程按 query 时资源/cache/distribution 事实选择不同合法 plan，而不是
在构造 runtime 时冻结唯一 memory budget。

## 4. 验证与边界

新增门禁证明：

- cold single 在高 compile cost 下选择低编译成本 backend；
- repeated-100 选择低 steady-state latency backend；
- exact compiled artifact cache hit 也选择低 steady-state latency backend；
- 三个 context 的 PlanInstance hash 均不同；
- runtime per-query context 进入 plan provenance/cache namespace，final bounds 保持一致。

结果：

- Plan/compiler context 定向：`29 passed`；
- 全量：`466 passed, 1 skipped, 6 warnings`，66.06 s；
- Mypy：0 issues；
- Pylint：10.00/10；
- Black、`git diff --check` clean。

IR-5A 只证明 adaptive selection mechanism，不证明性能贡献。下一步必须新增 typed
fixed/local/global/oracle 公平策略、全新 held-out workload 和
regret/TTV/tail latency/peak memory artifact；在该证据前 IR-5 保持 pending。

