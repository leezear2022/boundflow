# 变更记录：Phase 5A PR#2（interval_v2 最小 partition 输出多 task DAG）

## 动机

在 Phase 5A PR#1 已经钉住 TaskGraph/PlanBundle/scheduler 的基础上，PR#2 的目标是：

- 让 planner 产出 **多 task + TaskGraph**（DAG）
- 严格保证语义不变：输出与 Phase 4 的单 task 执行完全一致（作为后续 5B/5E 优化的回归基线）

## 本次改动

### 1) interval_v2 planner（baseline partition）

- 新增：`boundflow/planner/interval_v2.py`
  - 复用 `plan_interval_ibp_v0` 获取 canonical 的 TaskOp 列表与 StoragePlan
  - baseline partition 规则：
    - `permute/transpose/layout_only` 单独成段（layout-only region）
    - 其它算子累积成 compute 段
    - 若仍不足 `min_tasks`，按 op 数量做一次确定性的二分
  - 将 segment lowering 成多个 `BoundTask`，并生成 `TaskGraph`（buffer 级依赖）
  - 为每个 task 明确填充 TaskIO：`input_buffers` / `output_buffers`（对齐 `StoragePlan.value_to_buffer`）

### 2) planner 导出

- 修改：`boundflow/planner/__init__.py`
  - 导出 `plan_interval_ibp_v2`

### 3) scheduler 默认输出推断增强

- 修改：`boundflow/runtime/scheduler.py`
  - 当 `output_value` 未指定时：
    - 若可推断唯一 sink task 且其 `output_values` 也唯一，则自动返回该 value
    - 否则要求调用者显式指定 `output_value`

## 测试

- 新增：`tests/test_phase5a_pr2_partition_multitask_equivalence.py`
  - MLP/CNN：验证
    - `plan_interval_ibp_v2 + run_ibp_scheduled(...)`
    - 与 `plan_interval_ibp_v0 + PythonTaskExecutor.run_ibp(...)`
    - 输出上下界 allclose
  - 新增手工构造的 branch+merge primal graph 回归：确保 cross-segment 依赖在 buffer 级上被正确连边

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5a_pr2_partition_multitask_equivalence.py
conda run -n boundflow python -m pytest -q
```
