# Phase 7A PR-18：Hybrid Planner / Capability Table

**日期**: 2026-07-01

## 背景

PR-15 到 PR-17 已经把 Phase 7A 从局部优化推进到可观测闭环：

- PR-15 增加 opt-in operator attribution，并把 `unknown_materialization` 清零。
- PR-16 增加 run-local dense cache，证明 `concat_relu_mlp` 存在一部分可复用的 `RightMatmulLinearOperator.to_dense()`。
- PR-17 增加 final concretization policy，使 `structured` 与 `dense_barrier` 能在同一 benchmark 口径下比较。

PR-18 的目标不是再手工优化一个 operator，而是把这些观测结果收口成一个最小 hybrid planner：

```text
structured when useful
cached dense when exact sign split requires materialization
dense_barrier when final concretization is cheaper as dense
future lowering only after policy boundary is visible
```

## 主要改动

### 1. 新增 planner / capability table

新增 `boundflow/runtime/bound_planner.py`：

- 定义 `BoundOpCapability`。
- 定义 `PHASE7A_CAPABILITY_TABLE`。
- 定义 `Phase7APlannerDecision`。
- 新增 `plan_phase7a_shared_crown(...)`。
- 新增 `phase7a_capability_table_jsonable()`。

当前 capability table 明确记录：

- `RightMatmulLinearOperator`
  - `relu_pullback = exact_requires_dense_sign_split`
  - `split_pos_neg = exact_dense_fallback`
  - `dense_cache = eligible`
  - `planner_action = cached_dense_do_not_fake_structured_sign_split`
- `SliceInputLinearOperator`
  - `split_pos_neg = exact_structured_delegation`
  - `relu_pullback = exact_embedding_materialization`
  - 后续可做 exact fast path，但 PR-18 不做数学改写。
- `AddLinearOperator` / `ScaledInputLinearOperator`
  - 适合继续用 cache；若 wrapper/Python dispatch 主导，再转向 planner-level fusion。

### 2. benchmark 支持 auto final policy

更新 `scripts/bench_phase7a_shared_crown_path_attribution.py`：

- `--final-concretization-policy` 新增 `auto`。
- 顶层 schema version 仍保持 `phase7a_shared_crown_path_attribution.v1`。
- `meta.capability_table` 写入当前 capability table。
- 每个 row 新增 `planner_decision`。

当前 auto 规则：

- `relu_barrier` workload：
  - final policy 选择 `structured`
  - 原因：PR-17 smoke 显示 dense final barrier 没有解决 ReLU-heavy 的主成本，当前仍应保留 structured final path，同时依赖 PR-16 run-local dense cache 处理 exact `RightMatmul` 成本。
- `layout_only` workload：
  - final policy 选择 `dense_barrier`
  - 原因：PR-17 smoke 显示 layout-only final concretization 在 dense barrier 下有明确改善信号。

### 3. 测试

更新 `tests/test_phase7a_pr11_shared_crown_bench.py`：

- 验证 capability table schema。
- 验证 `RightMatmulLinearOperator` 的 planner action 不会伪 structured sign split。
- 验证 auto planner：
  - `relu_barrier -> structured`
  - `layout_only -> dense_barrier`
- 验证 benchmark JSON 中包含 `planner_decision` 与 `meta.capability_table`。

## 结果摘要

PR-18 后，Phase 7A 的 shared CROWN path 已形成完整闭环：

```text
PR-15 attribution
PR-16 run-local dense cache
PR-17 final concretization policy
PR-18 hybrid planner / capability table
```

这使 BoundFlow 的主张从“structured path 一定更快”收缩为：

```text
BoundFlow can observe, explain, and plan dense / structured / backend lowering boundaries for verification workloads.
```

## 验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7a_pr11_shared_crown_bench.py tests/test_phase7a_linear_operator_concretize.py
```

结果：

```text
20 passed, 1 skipped
```

后续还需要跑完整 Phase 7A 回归与 all-workload smoke，作为阶段收口证据。

## 下一步

Phase 7A 已具备收口条件。下一阶段建议转向：

1. 用 PR18 auto planner 复跑 CPU/CUDA bench，形成 structured / cached dense / dense barrier 的表格。
2. 如果 ReLU workload 仍由 `RightMatmul` exact sign split 主导，不继续伪 structured，而是设计 cached dense + hybrid policy 的论文叙事。
3. 如果 layout-only / final concretization 已有稳定 crossover，可把它作为 selective lowering 的第一个候选 hot path。
4. Phase 7B 再做成本模型与更正式的 planner policy；Phase 7C 再做 selective TVM lowering。
