# 变更记录：IR-4E PR-13 query migration 与 IR-4 closure

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`b9338a8`（IR-4D typed compiler query + exact state runtime）
> 状态：IR-4 validated-reduced closure；下一阶段为 IR-5 adaptive PlanInstance

## 1. PR-13 query contract 的 compiler capability

`make_bound_query()` 现在区分三类 capability：

- `alpha_beta_dense_split`：历史 αβ-CROWN BaB；
- `alpha_dense`：历史 α-CROWN；
- `plain_crown_typed_ir`：唯一可进入当前 compiler IR 的 plain-CROWN 请求。

`plain_crown_typed_ir` 固定为：

- `BoundMethod.CROWN`；
- `OptimizationStage.FINAL_BOUND`；
- `requires_grad=False`；
- requested outputs 仅为 `bounds`；
- 不携带 alpha/beta/split state。

其余未支持 method 在 query 构造时 fail closed。

## 2. PR-13 BatchManager → typed compiler adapter

新增 `CompilerBoundQueryRequest`，将 immutable PR-13 query identity 与完整
`TypedCompilerQueryPayload` 配对，并逐项核对：

- method/stage/capability/requested outputs；
- input value、perturbation identity 和 center content hash；
- linear objective content hash；
- model structure/weight versions；
- alpha/beta/split state 必须为空。

新增 `CompilerSameSolverQueryRuntime`：

1. 用原 PR-13 `DynamicBatchManager` 做 compatibility bucket、deadline/memory accounting、
   OOM bisection 和结果顺序恢复；
2. batch executor 只调用 `TypedCompilerQueryRuntime.execute_bound_queries()`；
3. 后者严格执行 PlanInstance→TaskIR→ScheduleIR→typed backend；
4. audit 固定记录 `legacy_executor_dispatches=0`；
5. 同一 compatibility batch 当前仍逐 query 执行 typed schedule，不宣称 physical
   cross-query batching。

fresh-process artifact 已升级为
`boundflow.compiler-query-runtime-artifact/v2`，重放完整
PR-13 manager→typed compiler→state cache/reuse 链。

## 3. legacy α/β 路径退役边界

PR-14 已证明 external whole-query α/β/split 不能等价降级为当前 plain-CROWN compiler。
因此没有把旧 executor 伪装成 compiler backend。

`SameSolverQueryRuntime` 现在默认拒绝执行，错误明确说明：

- 这是 historical validated-reduced replay；
- 必须显式设置 `allow_legacy_alpha_beta=True`；
- PR-14 external mismatch 保持 No-Go；
- 不允许 compiler fallback。

只有 PR-13C/D 历史脚本与对应回归 fixture 设置该开关。当前 compiler runtime 不设置，
也不调用 legacy executor。这样既保留既有数值/性能证据的可重放性，又消除默认隐藏主路径。

## 4. IR-4 冻结门禁审计

| 门禁 | 结论 | 证据 |
|---|---|---|
| backend 只消费 Task/Schedule IR | validated-reduced | 当前 compiler query 只进入 typed registry；legacy α/β 默认关闭、仅 historical opt-in |
| PR-10—13 correctness 不退化 | pass | 全量 464 passed，PR-13A/B/C 定向全部通过 |
| compile cache/capability/state validity 由新 ID/hash 驱动 | pass | dispatch cache v2、query compatibility、Bound/Plan/Task hashes、exact state payload |
| PR-14 mismatch 显式 No-Go | pass | α/β adapter rejection + legacy default rejection，均无 plain-CROWN fallback |

IR-4 的 closure 范围是已冻结的 narrow plain-CROWN compiler subset。它不表示：

- α/β/split Task IR 已实现；
- PR-14 external verifier 已接入；
- non-toy whole-query bound equivalence 已恢复；
- C1/C2 已有足够 ASPLOS 性能证据；
- C3 恢复为独立 acceleration claim。

## 5. 验证结果与下一阶段

- IR-4E/PR-13 定向：`24 passed`；
- 全量：`464 passed, 1 skipped, 6 warnings`，66.33 s；
- Mypy：0 issues；
- Pylint：10.00/10；
- Black、`git diff --check` clean。

下一阶段严格进入 **IR-5 adaptive PlanInstance**：

- memory/cache/query-distribution 驱动 runtime selection；
- 对比 fixed、local greedy、ordinary batching 和公平 batched original；
- 多预算合法计划切换；
- held-out Oracle regret、TTV、tail latency、peak memory；
- 证明收益来自跨层规划/调度，而不是简单 pack batch。

IR-6 cached specialization 仍受 break-even 门禁约束，不提前启动。

