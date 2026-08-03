# PR-11 有界分层重试变更记录

> 日期：2026-07-12
> 状态：实现、两组 held-out replay 与真实 CUDA OOM 重复实验均已完成；PR-11 整体仍为 partial。

## 1. 问题

原 `global_retry` 按预测 latency 顺序逐个尝试所有预测可行组合。在
mini-ResNet `spec=32/domain=8` 上最多只需 4 次，但扩到 `spec=128/domain=8` 后，部分预算需要
23、43、56 次 measured budget-rejection replay。这个策略能找到可行计划，却不具备真实 host
runtime 所需的重试次数上界。

同时，简单的 5% predicted-peak backoff 虽把次数限制到 5，却使小规模 held-out median regret
恶化到 1.562×，因此没有作为默认策略保留。

## 2. 实现

新增 `latency_rank_stratified_v1`：

1. 在预测峰值不超过预算的候选中选择两个预测 latency 最低的方案；
2. 再覆盖 latency 排名的 80% 与 90% 分位；
3. 最后一槽固定留给全体候选中 predicted peak 最低的方案；
4. 默认总尝试次数上限为 5，并去重。

代码落点：

- `boundflow/planner/materialization_placement.py`：候选 cost schema 与有界分层排序；
- `boundflow/runtime/scheduler.py`：排序后执行、真实 CUDA OOM blacklist 与统计；
- `boundflow/runtime/crown_ibp.py`：plain CROWN wrapper 可通过 memory budget 启用有界排序；
- `scripts/evaluate_phase7a_pr11_barrier_placements.py`：新增 `global_bounded_retry` policy；
- real-OOM smoke/runner：改为走同一有界 runtime 入口。

## 3. 两组 held-out 结果

两组都只用 calibration profile 拟合 cost model；held-out exhaustive measurement 仅用于预算拒绝
replay 与最终 Oracle 比较。

| Query | Oracle-feasible | Planner-feasible | unexpected | median regret | p90 | max | 最大尝试数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mini-ResNet s32/d8 | 7 | 7 | 0 | 1.159× | 1.722× | 1.738× | 3 |
| mini-ResNet s128/d8 | 7 | 7 | 0 | 1.171× | 1.221× | 1.577× | 5 |

工件：

```text
artifacts/phase7a-pr11/pr11-barrier-eval-v6-stratified-s32-d8-20260712/
artifacts/phase7a-pr11/pr11-barrier-eval-v6-stratified-s128-d8-20260712/
```

这两组结果同时满足 0 unexpected failure 与 median regret ≤ 20% 的 reduced 内部门禁；大规模
query 的最大尝试数从无界 replay 的 56 降到 5。p90/max 仍必须报告，不能只展示 median。

## 4. 真实 CUDA OOM

在 `380 MiB` process-local allocator cap、mini-ResNet `spec=128/domain=32` 下重复 3 个独立子进程：

```text
DDDDDDD -> real torch.cuda.OutOfMemoryError
SSSSSSS -> success
```

3/3 均为 attempts=2、oom_failures=1，输出 finite 且 lower≤upper；peak allocated/reserved 为
372,348,928 / 381,681,664 bytes。新工件：

```text
artifacts/phase7a-pr11/pr11-real-oom-retry-stratified-380mib-20260712/
```

## 5. Claim 边界

- 可以声称：在两种 query scale 的 reduced held-out matrix 上，有界分层 retry 以最多 5 次尝试
  达到 7/7 可行、0 unexpected，并保持 median regret ≤ 20%；真实 CUDA OOM 回退稳定 3/3。
- 不能声称：成本模型已跨任意架构/规模泛化，或该策略已覆盖 BaB 长生命周期、queue、timeout、
  compiled-plan cache 与 allocator fragmentation。
- 下一步不是继续为这两个点调分位常数，而是增加独立 workload/query，并接入 BaB scheduler；
  若新查询失败，应记录失败并重新审视候选特征与 runtime fallback。

## 6. 当前验证

- PR-11 placement/evaluator 专项：15 passed；
- 两组 evaluator 均成功生成 JSONL、summary CSV、cost model 与 manifest；
- 真实 CUDA OOM：3/3 独立子进程稳定恢复；
- 全量 pytest：216 passed、1 skipped；
- Mypy：11 个 PR-11 Planner/runtime/script 文件无问题；
- Pylint：4 个 Planner 模块与 6 个 PR-11 脚本逐文件均为 10.00/10；
- `git diff --check`：通过。
