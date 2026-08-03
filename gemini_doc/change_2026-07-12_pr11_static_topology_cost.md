# PR-11 Static Topology/Liveness Cost 变更记录

> 日期：2026-07-12
> 状态：candidate-independent static feature、LOO retry calibration 与三组 held-out 已完成；
> production candidate wiring 已完成；BaB/QueryState wiring 与更广 workload 仍未完成。

## 1. 修复的问题

上一版 evaluator 的 `features_for_row()` 会读取每个 candidate 的 `trace_on` logical bytes。它不读取
latency/peak target，但部署时仍需先执行 candidate，因此只能算 profile-guided replay。

本切片将 placement feature 改为只依赖：

- Task IR 拓扑；
- query 已知的 spec/domain batch；
- forward bound 已知的 barrier shape/dtype；
- placement action 本身；
- calibration-only 的 latency/peak target。

final held-out candidate 的 trace、latency 和 peak 均不进入 prediction feature。

## 2. Static barrier schema

新增 `materialization_static_features.py`，为每个 ReLU pre-activation 生成：

- producer op type 与 topo index；
- per-domain value shape/numel、spec/domain batch axes、element size；
- coefficient elements/bytes 与 ReLU row-scaling estimated FLOPs；
- 显式 reuse count；
- direct consumer count 与 direct live span；
- downstream depth；
- downstream merge/branch/path count；
- merge-output、branch-source 标记。

这些字段从 `BoundTask` producer/consumer 图与 barrier shape 静态推导，不随 dense/structured
candidate 改变。static/profile/cost-model/evaluator schema 分别升为 v2/v3/v3/v4；evaluator 会拒绝
缺少 static barrier metadata 的旧 profile。

旧 `trace_on` 仍保留在 profile 中作为机制观测和审计字段，但不再用于 `features_for_row()`。

## 3. 有界 retry 的显式保守候选

最终 fallback 不再由错误的 peak model 任意选择。`PlacementRetryCandidate` 新增
`conservative` metadata；plain CROWN 的 all-structured legal candidate 被显式标为保守方案。
这使 static-v2 branched held-out 从 7/9 恢复为 9/9 feasible。

## 4. Candidate-budget factor calibration

为了应对新拓扑上的 peak prediction calibration shift，bounded retry 允许尝试 predicted peak
略高于真实预算的候选；实际安全仍由真实 CUDA OOM feedback、blacklist 和 6 次上限保证。

新增 `calibrate_phase7a_pr11_retry_factor.py`，在 6 个 workload/scale calibration family 上做 LOO，
每个 family 最多取 8 个代表预算，共 36 个 budget，并联合选择 ridge/factor。选择规则固定为：

```text
min unexpected failures
  → min p90 regret
  → min median regret
  → smaller factor
  → smaller ridge
```

候选序列 v3 使用“最快 + 最快 decile 内 topology-diverse + 80%/90% latency rank + 最快
near-conservative + all-structured”，最多 6 次。所有 calibration 与 held-out profile 均执行 3 次
独立 shuffled order，再按 query/pattern 聚合跨运行 median。最终选择 **ridge=0.001、factor=1.30**。
选择工件：

```text
artifacts/phase7a-pr11/pr11-static-v3-agg-ridge-factor-loo-20260712/
```

该因子随后冻结，不使用 final held-out 结果调参，并显式写入 evaluator manifest 与 runtime API。

## 5. Profile 与 final held-out

v3 replicated aggregate profile：

```text
artifacts/phase7a-pr11/pr11-static-v3-agg-calibration-s32-d8-20260712/
artifacts/phase7a-pr11/pr11-static-v3-agg-mini2-calibration-s128-d8-20260712/
artifacts/phase7a-pr11/pr11-static-v3-agg-mini-heldout-s32-d8-20260712/
artifacts/phase7a-pr11/pr11-static-v3-agg-mini-heldout-s128-d8-20260712/
artifacts/phase7a-pr11/pr11-static-v3-agg-branched-heldout-s32-d8-20260712/
```

3 轮共 1,416 个 executions 全部状态为 ok、与 dense reference 对齐；聚合后为 472 个唯一
query/pattern。聚合器验证三轮 coverage、placement 和 static metadata 完全一致，并记录每个 pattern
的 latency min/max/p90 与 peak range。

Global Bounded Retry final 结果：

| Held-out query | feasible / oracle | unexpected | median regret | p90 | max | 最大尝试数 |
|---|---:|---:|---:|---:|---:|---:|
| mini-ResNet s32/d8 | 7 / 7 | 0 | 1.000× | 1.747× | 1.747× | 5 |
| mini-ResNet s128/d8 | 7 / 7 | 0 | 1.194× | 1.194× | 3.061× | 5 |
| branched-ResNet s32/d8 | 9 / 9 | 0 | 1.880× | 2.377× | 3.160× | 3 |

final evaluator 工件：

```text
artifacts/phase7a-pr11/pr11-static-v3-agg-final-default-mini-s32-d8-20260712/
artifacts/phase7a-pr11/pr11-static-v3-agg-final-default-mini-s128-d8-20260712/
artifacts/phase7a-pr11/pr11-static-v3-agg-final-default-branched-s32-d8-20260712/
```

三组都通过 reduced feasibility、0 unexpected 和 median regret ≤20% 门禁。独立 branched
topology 已不再失败，证明 static topology/liveness feature 比 aggregate trace feature 更可信。

## 6. 仍然不能宣称的内容

- 最坏 held-out p90/max regret 为 2.377×/3.160×，不能只报告 median；
- replicated profile 的跨运行波动仍明显，说明结果必须保留 raw replicate range；
- static summary、model loader、candidate generator 和 plain-CROWN retry 已连通，但尚未纳入统一
  `QueryState`/BaB scheduler；
- 当前 cost model 仍是小样本 ridge，不代表跨任意 network/shape 泛化；
- 未接 BaB long-lived scheduler、timeout、queue、compiled-plan cache、online update；
- retry factor 只在当前 calibration families 上选定，扩 workload 后必须重新验证而非静默修改。

因此 C2 可标记 validated-reduced，但不是论文级 complete。按 PR 边界，BaB/QueryState 属于 PR-13；
PR-12 只能在 PR-11 closure audit 与提交边界整理完成后开始。

## 7. Production foundation 与真实 OOM

新增 `StaticPlacementQuery`、cost-model `from_dict()` 与
`generate_static_placement_candidates()`；专项测试已证明 static summary → loaded model → 4 个 legal
plans → bounded plain-CROWN runtime 可直接组合，不再通过 evaluator 私有逻辑生成候选。

v3 topology-density runtime 在 380 MiB cap 下再次做 3 个独立子进程：dense real OOM 后
all-structured 成功，3/3 attempts=2、OOM=1、finite 且 lower≤upper。工件：

```text
artifacts/phase7a-pr11/pr11-real-oom-retry-static-v3-final-380mib-20260712/
```

## 8. 最终验证

- 全量 pytest：225 passed、1 skipped；
- core/runtime 与脚本 Mypy：全部通过；
- 5 个新增 Planner 模块与 8 个 PR-11 脚本逐文件 Pylint：10.00/10；
- `git diff --check`：通过；
- closure 逐项审计：`gemini_doc/pr11_closure_audit_2026_07_12.md`。
