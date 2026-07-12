# PR-11 Selective Materialization Planner Closure Audit

> 日期：2026-07-12
> 判定：**validated-reduced；工程切片可以冻结，C2 论文主张仍需 PR-12/PR-13 扩展。**

## 1. 需求逐项审计

| PR-11 要求 | 判定 | 当前证据 |
|---|---|---|
| query/method/autograd/memory context | achieved | `MaterializationContext`、`StaticPlacementQuery`；CROWN/α/αβ capability guard |
| dense/structured/reduce-batch action | achieved reduced | plain CROWN dense/structured；不可行时 deterministic reduce-batch re-plan signal |
| shape/FLOPs/bytes/reuse/batch-axis cost summary | achieved | static barrier schema v2 显式字段；从 Task IR + forward shape 生成 |
| lazy-vs-materialize 全局决策 | achieved | 7 barrier 最多 128 个 mixed combinations；static model 预测 peak/latency |
| deterministic heuristic 与 cost-aware policy | achieved | dense/structured/threshold/local/global/bounded retry/Oracle |
| memory budget、capability、fallback | achieved reduced | safe budget、α/αβ structured 拒绝、6 次有界 retry、all-structured conservative fallback、真实 OOM |
| plan dump 与 decision reason | achieved | placement/plan/model JSON schema、manifest、attempted pattern trace |
| 不同 workload/budget 产生不同计划 | achieved | mini/branched、s32/s128、23 个 feasible budgets 的 mixed pattern 变化 |
| workload-family held-out | achieved reduced | 6-family LOO calibration；mini/branched final held-out 不进入拟合 |
| correctness/soundness | achieved reduced | 3× replicated profile 共 1,416/1,416 与 dense reference 对齐；全量测试通过 |
| 实际 OOM recovery | achieved reduced | 380 MiB allocator cap，dense OOM→structured success，3/3 独立进程 |
| production candidate foundation | achieved | static summary→model loader→candidate generator→plain-CROWN bounded runtime |

`spill-to-host` 没有伪装成已实现：v1 fallback 范围是 structured、bounded retry 与 reduce-batch
re-plan。真正的 spill/recompute scheduling 仍属于后续 memory planner 扩展。

## 2. 冻结参数与证据链

所有 calibration/held-out profile 运行 3 次独立 shuffled order，聚合器按相同 query/pattern 取跨运行
median，并保存 min/max/p90 与源文件 hash。总计 1,416 次 execution，聚合为 472 个 patterns。

calibration-only LOO：

```text
6 query families × 最多 8 个代表预算 = 36 budget points
selected ridge = 0.001
selected candidate-budget factor = 1.30
candidate cap = 6
```

选择工件：

```text
artifacts/phase7a-pr11/pr11-static-v3-agg-ridge-factor-loo-20260712/
```

## 3. Final held-out

| Query | feasible/oracle | unexpected | median regret | p90 | max | max attempts |
|---|---:|---:|---:|---:|---:|---:|
| mini-ResNet s32/d8 | 7/7 | 0 | 1.000× | 1.747× | 1.747× | 5 |
| mini-ResNet s128/d8 | 7/7 | 0 | 1.194× | 1.194× | 3.061× | 5 |
| branched-ResNet s32/d8 | 9/9 | 0 | 1.880× | 2.377× | 3.160× | 3 |

结论：23/23 feasible、0 unexpected，三个 query 的 median 都通过 ≤20% 内部门禁。p90/max 明显
更差，必须作为 limitation 保留，不能用 median 掩盖。

Final 工件：

```text
artifacts/phase7a-pr11/pr11-static-v3-agg-final-default-mini-s32-d8-20260712/
artifacts/phase7a-pr11/pr11-static-v3-agg-final-default-mini-s128-d8-20260712/
artifacts/phase7a-pr11/pr11-static-v3-agg-final-default-branched-s32-d8-20260712/
artifacts/phase7a-pr11/pr11-real-oom-retry-static-v3-final-380mib-20260712/
```

## 4. 质量门禁

- 全量 pytest：225 passed、1 skipped；
- Mypy：6 个 core files 与 8 个 PR-11 scripts 均无问题；
- Pylint：5 个新增 Planner 模块与 8 个 PR-11 scripts 逐文件 10.00/10；
- `git diff --check`：通过；
- runtime 历史文件不宣称 Pylint 10/10，以 Mypy、专项/全量测试和真实 OOM 覆盖。

## 5. Closure 决定

PR-11 的工程目标达到 **validated-reduced**，可以整理为独立提交并进入 PR-12 的 fused CROWN
task lowering。以下内容不属于“PR-11 已完成”的扩大解释：

- 更多真实模型/VNN-COMP workload；
- TVM fused lowering、compile amortization（PR-12）；
- QueryState、真实 BaB node stream、timeout/queue/cache（PR-13）；
- certified training（PR-14）；
- 论文级 p90/p99 与端到端 headline result。

当前工作区尚未 commit/push；需用户明确授权后执行版本控制操作。
