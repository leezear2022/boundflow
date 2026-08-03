# PR-11 独立拓扑 Held-out No-Go 记录

> 日期：2026-07-12
> 结论：feasibility 通过，但 latency-regret 与部署特征门禁失败；禁止将 PR-11/C2 标记 complete。

> 后续：该 No-Go 触发的 static topology/liveness 修复与重新评估见
> `gemini_doc/change_2026-07-12_pr11_static_topology_cost.md`；本文件保留失败证据，不回写结果。

## 1. 新 held-out topology

新增 `branched_resnet` profile workload：

```text
stem ReLU
  ├─ residual block left
  └─ residual block right
       ↓
      add + ReLU
       ↓
concat(stem) + fuse conv + ReLU
```

它包含并行 branch、residual add、concat、fuse conv 和 7 个实际 backward barrier，不同于原
mini-ResNet 的串行 3-block topology。完整 exhaustive profile 为 128 combinations，全部通过
dense-reference correctness：

```text
artifacts/phase7a-pr11/pr11-barrier-branched-heldout-shuffled-s32-d8-20260712/
```

配置为 CUDA、`spec=32/domain=8`、warmup 2、repeats 5、deterministic shuffled order。

## 2. Held-out 结果

calibration 仍只使用原 5 个 workload/56 rows；branched profile 不进入拟合。预算为
28、30、32、34、36、38、40、44、48、52 MiB，其中 9 档 Oracle-feasible。

| Policy | feasible / oracle | unexpected | median regret | p90 | max |
|---|---:|---:|---:|---:|---:|
| Always Dense | 1 / 9 | 8 | 1.254× | 1.254× | 1.254× |
| Always Structured | 9 / 9 | 0 | 6.374× | 6.781× | 7.199× |
| Memory Threshold | 5 / 9 | 4 | 6.781× | 7.199× | 7.199× |
| Global Predicted | 5 / 9 | 4 | 1.976× | 6.374× | 6.374× |
| Global Retry | 5 / 9 | 4 | 1.976× | 6.374× | 6.374× |
| **Global Bounded Retry** | **9 / 9** | **0** | **1.976×** | **4.494×** | **6.374×** |

工件：

```text
artifacts/phase7a-pr11/pr11-barrier-eval-v7-stratified-branched-s32-d8-20260712/
```

有界 retry 保住了 feasibility，却远未达到 median regret ≤ 20% 门禁。多个低预算点直接选择
all-structured；例如 38 MiB 下选择 44.890 ms 的 `SSSSSSS`，而 Oracle `DSSSSDD` 仅
7.043 ms。

## 3. 根因与额外审计发现

当前 cost summary 主要包含 barrier 数量/位置、聚合 persistent/ephemeral bytes 与 dense baseline，
没有显式表达 branch fanout、merge 类型和 live range。并行拓扑中的同数量 placement 因此无法被
可靠区分。

此外，当前 evaluator 的 `features_for_row()` 从每个 candidate profile 的 `trace_on` 读取 logical
materialization bytes。它不读取 held-out latency/peak target，因此不是直接 target leakage；但从
部署角度，cold planner 无法在不先运行 candidate 的情况下取得该特征。现有 evaluator 应被准确
描述为 **profile-guided replay**，不能作为静态 Planner 已可部署的证据。

## 4. Go/No-Go

本轮对“直接用 v1 aggregate feature + bounded retry 完成 PR-11”的判断为 **No-Go**。下一切片必须：

1. 从 Bound IR/Task IR 的 shape、producer/consumer、fanout 与 live interval 静态产生 barrier cost；
2. 将 merge/branch/liveness metadata 纳入 placement cost summary；
3. 明确 calibration、validation 和 untouched held-out topology；
4. 在不读取 held-out candidate trace 的条件下重跑 evaluator；
5. 继续保留当前失败工件，不改预算或删除 workload。

在上述条件完成前，不进入 PR-12 fused lowering，也不把 C2 从 partial 提升为 validated。

## 5. 验证

- 全量 pytest：217 passed、1 skipped；
- modified profiler Mypy：success；
- modified profiler Pylint：10.00/10；
- `git diff --check`：通过。
