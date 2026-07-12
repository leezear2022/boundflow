# PR-11 Barrier Evaluator 与 Host Retry 记录

> 日期：2026-07-12
> Artifact：`artifacts/phase7a-pr11/pr11-barrier-eval-v2-global-retry/`
> 状态：measured placement evaluator、host retry 与真实 CUDA OOM 重复实验已落地。

> 后续更新：无界 retry 的跨规模尝试次数问题及有界分层策略见
> `gemini_doc/change_2026-07-12_pr11_bounded_stratified_retry.md`。

## 1. 严格数据边界

- calibration/validation profile：5 个 workload、56 个 exhaustive combinations；
- final held-out：3-block mini-ResNet、7 barriers、128 combinations；
- 两侧均为 deterministic shuffled order、`spec=32/domain=8`、warmup 2、repeats 5；
- cost model 只拟合 calibration rows；held-out measurement 只用于最终 actual/Oracle；
- all-dense baseline peak/latency 作为 repeated-query profile 输入，不使用其他 held-out target。

## 2. Evaluator

新增：

- `materialization_placement_cost_model.py`：interaction-aware peak/latency ridge model；
- `evaluate_phase7a_pr11_barrier_placements.py`：calibration → prediction → held-out
  Local/Global/measured-Oracle；
- raw JSONL、summary CSV、cost model 与 manifest/hash；
- `global_retry`：按预测 latency 排序候选，预算拒绝后 blacklist 并尝试下一个候选。

`global_retry` 的 evaluator 只给策略逐次返回“该候选是否超过预算”，不暴露 Oracle 最优 pattern。
但它仍是 **measured budget-rejection replay**，不是实际 CUDA OOM 运行。

## 3. Final held-out 结果

预算为 28、30、32、34、36、40、48、64 MiB；其中 7 档存在 Oracle-feasible combination。

| Policy | feasible / oracle | unexpected | median regret | p90 | max |
|---|---:|---:|---:|---:|---:|
| Always Dense | 3 / 7 | 4 | 1.000× | 1.000× | 1.000× |
| Always Structured | 7 / 7 | 0 | 5.486× | 6.975× | 6.975× |
| Memory Threshold | 7 / 7 | 0 | 2.668× | 5.486× | 6.975× |
| Local Predicted | 2 / 7 | 5 | 1.000× | 1.000× | 1.000× |
| Global Predicted | 5 / 7 | 2 | 1.159× | 1.722× | 1.722× |
| **Global Retry** | **7 / 7** | **0** | **1.159×** | **1.562×** | **1.722×** |

Global Retry 最多尝试 4 个候选。它相对 Always Structured/Memory Threshold 显著降低 regret，
并相对一次性 Global 消除 2 个 unexpected failures。内部 `median regret <= 20%` 与 0 unexpected
门禁在该 reduced held-out matrix 上通过。

## 4. Host runtime

`runtime/scheduler.py` 新增：

- ordered placement candidate execution；
- 只捕获真实 `torch.cuda.OutOfMemoryError`；
- OOM candidate blacklist；
- 可选 allocator cache 清理；
- attempts、OOM count、selected index、attempted patterns；
- 全候选失败时 `PlacementRetryExhausted`。

`run_crown_ibp_mlp_with_placement_retry` 将这一机制接到 plain CROWN。除 injected-OOM 单元测试外，
现已使用独立子进程的 PyTorch allocator cap 做真实实验：

```text
cap: 380 MiB
workload: mini-ResNet, spec=128, domain=32
candidate 0: DDDDDDD → real torch.cuda.OutOfMemoryError
candidate 1: SSSSSSS → success
```

3/3 独立重复均为 attempts=2、oom_failures=1、selected_index=1；结果 finite、lower≤upper；
peak allocated/reserved 为 372,348,928 / 381,681,664 bytes。工件：

```text
artifacts/phase7a-pr11/pr11-real-oom-retry-380mib-20260712/raw.jsonl
artifacts/phase7a-pr11/pr11-real-oom-retry-380mib-20260712/manifest.json
```

因此可以声称“在 process-local 真实 CUDA allocator cap 下，dense OOM 后 structured candidate
稳定恢复 3/3 次”。该结果仍只覆盖 plain CROWN 与一个 workload/configuration。

## 5. 当前边界

- 只有一个 final architecture family、一个 spec/domain 点；
- exhaustive Oracle 只适合 7-barrier reduced graph；
- online dense baseline profile 与 candidate retry 的 amortization 尚未计入 end-to-end；
- runtime retry 尚未接 BaB scheduler、timeout、queue 或 compiled-plan cache；
- 真实 OOM 已通过 3 次独立子进程；同一长生命周期 BaB 进程中的 allocator/state 泄漏仍待测；
- p90 regret 仍为 1.562×，需要更多 workload/budget 验证稳定性。

因此 PR-11 已越过“Global 等于简单阈值”的机制问题，并在 reduced held-out replay 上通过内部
feasibility/median-regret 门禁，但仍未达到整项 PR 的最终论文验收。

## 6. 验证

- 全量：212 passed、1 skipped；
- Mypy：placement cost model、scheduler、evaluator 通过；
- Pylint：新 Planner/evaluator 模块 10.00/10；
- `git diff --check` 在收尾执行。
