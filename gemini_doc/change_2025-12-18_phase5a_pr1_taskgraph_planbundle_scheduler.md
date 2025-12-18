# 变更记录：Phase 5A PR#1（TaskGraph / PlanBundle / scheduler）

## 动机

Phase 5 的后续工作（task 切分、cache/reuse、batching、部分 TVM、系统化消融）都需要一个“可调度的多 task DAG”作为基础抽象。

因此先落地 PR#1：只做骨架与最小可回归闭环，不引入复杂 partition/cost model。

## 本次改动

### 1) TaskGraph IR

- 新增：`boundflow/ir/task_graph.py`
  - `TaskGraph(task_ids, edges)` + `validate()` + `topo_sort()`（基于 value 级依赖）

### 2) Planner 输出骨架

- 新增：`boundflow/planner/core.py`
  - `PlannerConfig`
  - `PlanBundle`
  - `PlannerPass`（Protocol）+ `run_planner_passes()`

### 3) BFTaskModule 扩展

- 修改：`boundflow/ir/task.py`
  - `BFTaskModule.task_graph: Optional[TaskGraph]`
  - 新增 `get_task(task_id)`，`get_entry_task()` 基于它实现
  - `validate()` 中如果存在 `task_graph` 则一并校验

### 4) 串行调度器（Topo schedule）

- 新增：`boundflow/runtime/scheduler.py`
  - `run_ibp_scheduled(module, input_spec, executor, output_value)`
  - 如果 `module.task_graph` 为空则回退到旧的单 task 执行（Phase 4 行为不变）

### 5) PythonTaskExecutor 增加 task 级执行单元

- 修改：`boundflow/runtime/task_executor.py`
  - 新增 `run_ibp_task(task, env, params)`：在共享 env 上执行单个 task 的 ops

## 测试

- 新增：`tests/test_phase5a_pr1_taskgraph_and_scheduler.py`
  - 手工构造 2-task relu chain 的 TaskGraph
  - 断言 scheduler 输出 == 单 task（拼接 ops）输出

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5a_pr1_taskgraph_and_scheduler.py
conda run -n boundflow python -m pytest -q
```

