# 变更记录：Phase 5C PR#5（Planner Pipeline 统一入口 + config_dump 可复现消融）

## 动机

Phase 5 进入 “planner 计划空间搜索/系统化消融” 阶段后，必须先把两件事钉死：

1) **统一 planner 入口（pipeline）**：避免不同实验走不同 code path，导致数据不可比；  
2) **可序列化的配置快照（config_dump）**：bench 输出必须能记录“本次运行到底用了什么配置”，否则无法复现。

本 PR 先做骨架：interval-IBP 的 v0/v2 lowering + 可选 storage reuse，并把 config_dump 放到 `PlanBundle.meta`。

## 本次改动

### 1) PlannerConfig 结构化选项（占位但稳定）

- 新增：`boundflow/planner/options.py`
  - `PartitionOptions/PartitionPolicy`
  - `LifetimeOptions/LifetimeModel`（占位）
  - `LayoutOptions/LayoutPolicy`（占位）
  - `PlannerDebugOptions`
- 修改：`boundflow/planner/core.py`
  - `PlannerConfig.partition/lifetime/layout/debug`

### 2) Pipeline 统一入口 + config_dump

- 新增：`boundflow/planner/pipeline.py`
  - `plan(program, *, config) -> PlanBundle`
  - 内置 `_to_jsonable()`：把 dataclass/Enum 转为 JSON 友好的 dict/value
  - `PlanBundle.meta["config_dump"]`：可直接 `json.dumps(...)`
  - `PlanBundle.meta["planner_steps"]`：记录执行了哪些 planner step

### 3) bench：最小 pipeline config dump

- 新增：`scripts/bench_planner_pipeline.py`
  - 输出 JSON：包含 `num_tasks/num_edges/config_dump/planner_steps/reuse_stats`

### 4) 测试：不同 config 下 task 数变化但语义等价

- 新增：`tests/test_phase5c_pr5_pipeline_config_dump.py`
  - v0 single-task vs v2 multi-task：输出 bounds allclose
  - `config_dump` 必须 JSON 可序列化

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5c_pr5_pipeline_config_dump.py
conda run -n boundflow python scripts/bench_planner_pipeline.py --min-tasks 2 --policy v2_baseline
```

