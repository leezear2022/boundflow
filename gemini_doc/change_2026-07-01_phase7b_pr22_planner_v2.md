# Phase 7B PR-22：Planner v2 High-Confidence Promotions

**日期**: 2026-07-01

## 背景

PR-21 的 CPU matrix 生成了 `phase7b_cost_model_v1`，其中只有两条规则达到 high confidence：

```text
cpu + permute_reshape_linear + small -> structured
cpu + permute_reshape_linear + bench -> structured
```

PR-22 将这两条规则推进到 planner v2。其他 medium / low confidence 规则继续保留为 evidence，不改变 runtime planner。

## 主要改动

### 1. planner v2

更新 `boundflow/runtime/bound_planner.py`：

- 新增 `Phase7BCostModelRule`。
- 新增 `PHASE7B_COST_MODEL_RULES`。
- 新增 `phase7b_cost_model_rules_jsonable()`。
- 新增 `plan_phase7b_shared_crown(...)`。

`plan_phase7b_shared_crown(...)` 行为：

- manual request：`structured` / `dense_barrier` 继续尊重用户指定。
- `auto` request：
  - 若存在 high-confidence cost-model rule，使用该 rule。
  - 否则 fallback 到 PR-18 `plan_phase7a_shared_crown(...)`。

当前仅提升：

```text
cpu + permute_reshape_linear + small -> structured
cpu + permute_reshape_linear + bench -> structured
```

### 2. benchmark 接入 planner v2

更新 `scripts/bench_phase7a_shared_crown_path_attribution.py`：

- `_collect_row()` 改用 `plan_phase7b_shared_crown(...)`。
- `planner_decision` 中可出现：
  - `planner = phase7b_cost_model_v1`
  - `confidence = high`
  - `evidence = {...}`
- meta 新增 `cost_model_rules`，便于审计当前 runtime 内置规则。

### 3. planner-v2 audit 脚本

新增 `scripts/report_phase7b_planner_v2_candidates.py`。

输入：

```text
phase7b_cost_model_v1
```

输出：

```text
phase7b_planner_v2_candidates.v1
```

报告：

- `promoted_rules`
- `missing_promotions`
- `held_back_rules`

当前最终 audit：

```text
promoted_count = 2
missing_promotion_count = 0
held_back_count = 10
```

promoted rules：

```text
permute_reshape_linear + small -> structured
permute_reshape_linear + bench -> structured
```

### 4. 测试

更新 / 新增：

- `tests/test_phase7a_pr11_shared_crown_bench.py`
  - 验证 planner v2 对 `permute_reshape_linear small` 选择 `structured` 且 `confidence == high`。
  - 验证 smoke scale 仍 fallback 到 PR-18 的 `dense_barrier`。
- `tests/test_phase7b_planner_v2_report.py`
  - 验证 high-confidence embedded rule 被标记为 promoted。
  - 验证 low-confidence rule 被 held back。

## 验证

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr11_shared_crown_bench.py \
  tests/test_phase7b_planner_v2_report.py
```

结果：

```text
12 passed
```

完整相关回归：

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_linear_operator_concretize.py \
  tests/test_phase7a_pr10_relu_barrier_structured.py \
  tests/test_phase7a_pr9_dag_linear_operator.py \
  tests/test_phase7a_pr11_shared_crown_bench.py \
  tests/test_phase7b_crossover_matrix.py \
  tests/test_phase7b_cost_model.py \
  tests/test_phase7b_planner_v2_report.py
```

## 下一步

Phase 7B 的 CPU 证据闭环已完成。后续应做：

1. 在有 CUDA 的机器上重复 PR-21 matrix。
2. 若 CUDA 上也出现 high-confidence rules，再加入 device-specific planner v2。
3. 对 `permute_reshape_linear small/bench -> structured` 的 high-confidence 规则，可作为 selective lowering 之前的稳定 policy 边界。
