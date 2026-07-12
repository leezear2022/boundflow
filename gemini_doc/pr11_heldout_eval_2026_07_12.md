# PR-11 Cost Model 与 Held-Out Evaluation 记录

> 日期：2026-07-12
> Artifact：`artifacts/phase7a-pr11/pr11-final-heldout-v4/eval/`
> 状态：第二实现切片完成；PR-11 总门禁仍未通过。

## 1. 数据隔离

本轮没有随机拆 360 行相邻 shape：

- calibration：MLP chain、CNN chain、residual block；
- validation/refit：add+concat DAG；
- final held-out：mini-ResNet；
- budget：64 MiB–8192 MiB 共 8 档；
- policy：Always Dense、Always Structured、Method-Only、Memory-Threshold、Local Greedy、
  Global；
- Oracle：每个 case/budget 使用实测 dense/structured status、peak 和 latency 选择最快合法可行
  action，不参与 held-out 决策。

最终模型按 method/action 分片，使用解释性 ridge linear peak model 与 log-latency model；α/αβ
structured 仍由 capability filter 禁止。

## 2. 证据链

```text
PR-10 normalized.csv
  → architecture-family split
  → cost_model.json
  → held-out raw.jsonl（1728 rows）
  → summary.csv
  → manifest.json + sha256
```

路径：

```text
artifacts/phase7a-pr11/pr11-final-heldout-v4/eval/raw.jsonl
artifacts/phase7a-pr11/pr11-final-heldout-v4/eval/cost_model.json
artifacts/phase7a-pr11/pr11-final-heldout-v4/eval/summary.csv
artifacts/phase7a-pr11/pr11-final-heldout-v4/eval/manifest.json
```

## 3. Final held-out 结果

mini-ResNet 上每种 policy 有 288 个 `method × shape × budget` 行，其中 Oracle 有 239 个可行点。

| Policy | feasible / oracle | unexpected | median regret | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| Always Dense | 231 / 239 | 8 | 1.00× | 1.00× | 1.00× | 1.00× |
| Always Structured | 91 / 239 | 148 | 6.52× | 7.97× | 9.17× | 9.17× |
| Method-Only | 239 / 239 | 0 | 1.00× | 6.86× | 9.17× | 9.17× |
| Local Greedy | 239 / 239 | 0 | 1.00× | 6.85× | 9.17× | 9.17× |
| Memory-Threshold | 239 / 239 | 0 | 1.00× | 1.00× | 5.44× | 9.17× |
| Global | 239 / 239 | 0 | 1.00× | 1.00× | 5.44× | 9.17× |

Global 在 3 个 strict-budget plain-CROWN 点中选择 structured，使 dense 超出该实验预算时仍有
可行 action。这里是 **budget-infeasible escape**，不是实际 CUDA OOM 恢复，不得写成“OOM
recovered”。

## 4. 当前判定

正证据：

- architecture-family final held-out 上 100% Oracle-feasible coverage；
- 0 unexpected failure；
- median/p90 regret 为 1.0；
- dense 与 structured 均被选择；
- capability guard 正确阻止 α/αβ structured；
- 相比 Always Dense 多覆盖 8 个 strict-budget 点；
- 相比 Method-Only/Local Greedy 显著改善 p90 tail。

未通过项：

- Global 与 Memory-Threshold 的 action count 和所有聚合指标完全相同；
- p99/max regret 仍为 5.44×/9.17×；
- 当前 decision unit 仍是整条 query/region，尚无 multi-barrier global placement；
- reduce-batch 尚未由 scheduler 自动执行；
- 没有真实 CUDA OOM → 自动恢复证据。

因此这批结果只能把 C2 更新为“held-out feasibility foundation validated”，不能证明 nontrivial
Global Planner contribution。下一步必须引入多个 materialization barrier/region 的联合计划，
否则 C2 会退化成 `if dense fits: dense else structured`。

## 5. 验证

- PR-11 专项：21 passed；
- 全量：200 passed、1 skipped；
- Mypy：4 个新/相关模块无问题；
- Pylint：10.00/10；
- plan schema contract、OOM retention、split disjointness、manifest hash 和 CSV summary 均有测试。
