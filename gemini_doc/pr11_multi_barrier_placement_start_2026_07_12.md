# PR-11 Multi-Barrier Placement 启动记录

> 日期：2026-07-12
> 状态：第三实现切片 foundation 完成；真实 held-out multi-barrier profile 尚未完成。

## 1. 动机

第二切片的 final held-out 结果表明 query-level Global 与 Memory-Threshold 完全相同。因此继续
调单一 query threshold 不会形成 C2。第三切片将 decision unit 改为独立 ReLU barrier，并联合
考虑 persistent、ephemeral、latency 与总显存预算。

## 2. 新增实现

`boundflow/planner/materialization_placement.py` 新增：

- `BarrierCost`：每个 barrier 的 dense/structured persistent bytes、structured ephemeral bytes
  与 latency；
- `PlacementContext`：barrier sequence、common persistent state、预算、method/autograd capability；
- `LOCAL_GREEDY`：每个 barrier 独立选最快 action，不协调总峰值；
- `GLOBAL_EXHAUSTIVE`：当前最多 20 个 barrier，枚举合法组合，在全局预算下选择最快计划；
- mixed `BarrierPlacement` 与 `MaterializationPlacementPlan` JSON dump；
- 无组合可行时返回确定性的 host reduce-batch request。

Runtime 新增按 ReLU pre-activation/source value 查找 action 的混合执行：同一次 CROWN backward
可让一个 ReLU 走 dense、另一个走 structured。α/αβ 仍拒绝任何 structured placement，直到
structured autograd capability 通过。

## 3. 已验证的非平凡行为

合成三 barrier 案例中：

```text
Local Greedy:
  全选最快 dense
  → 总 peak 超预算
  → requires re-plan

Global Exhaustive:
  联合选择 1 dense + 2 structured
  → peak 满足预算
  → 不需要缩 batch
```

这证明 Global API 已不再等价于单一 Memory-Threshold。两 ReLU MLP 的 mixed runtime path 与
all-dense bounds 对齐，并通过 materialization trace 证明只有指定 barrier 产生 persistent dense。

## 4. 验证结果

- multi-barrier 专项：7 passed；
- PR-11/CROWN/α/αβ 相关回归：41 passed；
- profiler 完成后全量：208 passed、1 skipped；
- 新 Planner/runner 模块 Mypy 通过；
- Pylint 10.00/10；
- runtime 文件已撤销误触发的全文件格式化，只保留语义 diff。

## 5. Barrier-level measured profile

新增 `scripts/profile_phase7a_pr11_barrier_placements.py`，对 fixed query 枚举所有 barrier action
组合，并分离 latency、peak、trace 与 dense correctness。

有效工件：

```text
artifacts/phase7a-pr11/pr11-barrier-calibration-shuffled-s32-d8-20260712/
artifacts/phase7a-pr11/pr11-barrier-mini-heldout-shuffled-s32-d8-20260712/
```

- calibration/validation：MLP、CNN、residual block、add+concat DAG、独立 2-block mini-ResNet，
  共 56 个组合；
- final held-out：3-block mini-ResNet，7 barriers、128 个组合；
- `spec=32/domain=8`，warmup 2、repeats 5、deterministic shuffled order；
- 184/184 rows ok，0 correctness failure。

held-out 代表数据：

- all dense：5.57 ms、38,653,952 peak bytes；
- all structured：38.86 ms、31,049,728 peak bytes；
- 全组合 peak 范围约 30.7–47.3 MB，latency 范围约 5.57–38.86 ms；
- 多个 mixed plan 位于 dense/structured 两端之间，但 peak 对 action 数量并不单调。

初始 lexicographic profile 暴露了组合顺序污染 latency，因此已增加 deterministic shuffle/seed，
旧的 lexicographic latency 不进入 cost model。基于 shuffled calibration 的第一版交互模型在
held-out 上仍无法同时满足“0 unexpected failure”和“median regret ≤20%”：提高 peak safety
inflation 可消除 unexpected failure，但会迫使过多 structured barriers，median regret 约 1.56×。
这项负结果尚需固化为独立 evaluator/JSONL，当前不得写成最终 Planner 结果。

## 6. 尚未完成

- barrier-level latency/peak 目前使用显式输入，尚未由真实 profile 校准；
- exhaustive 只适合小图，尚未实现 scalable heuristic/DP；
- 已有 mini-ResNet measured Oracle 原始组合，但 calibration→prediction→Local/Global/Oracle 的
  独立 evaluator 尚未固化；
- 当前 peak model 使用 `sum persistent + max ephemeral`，需要 allocator/profile 验证；
- 尚未把 placement plan 纳入 compile/cache key；
- scheduler reduce-batch 仍只是 signal。

因此本切片证明 nontrivial planning mechanism、mixed execution semantics 和 measured exhaustive
数据链已成立，但 cost model/heuristic 仍未通过门禁，不构成论文性能 claim。下一步是冻结
barrier placement evaluator，并针对 peak uncertainty 与 latency tail 设计 scalable heuristic。
