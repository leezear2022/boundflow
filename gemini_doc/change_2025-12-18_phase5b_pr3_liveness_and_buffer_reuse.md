# 变更记录：Phase 5B PR#3（Liveness + Physical Buffer Reuse v0）

## 动机

Phase 5A 已经把 `TaskGraph/TaskIO（buffer contract）/scheduler` 钉住，下一步进入 5B 需要把“内存生命周期 + 复用”从 executor 里抽出来，变成 planner 可插拔的分析与计划：

- 为后续 5B.2（更激进复用/更细粒度 lifetime）、5C（cache/reuse）、5E（Relax lowering 签名）提供稳定接口；
- 让“同一张 multi-task DAG”具备可度量的 memory footprint（physical buffers / bytes）。

本 PR 先实现 **task 粒度**的保守 liveness 与复用：保证语义不变、接口可扩展。

## 本次改动

### 1) StoragePlan 扩展：Logical vs Physical

- 修改：`boundflow/ir/task.py`
  - `StoragePlan.physical_buffers: Dict[str, BufferSpec]`
  - `StoragePlan.logical_to_physical: Dict[str, str]`
  - helper：`StoragePlan.to_physical()` / `num_logical_buffers()` / `num_physical_buffers()`
  - 兼容行为：当 `logical_to_physical` 为空时，默认 `physical == logical`（Phase 4/5A 行为不变）。

### 2) Liveness IR（task 粒度，保守）

- 新增：`boundflow/ir/liveness.py`
  - `BufferLifetime` / `LivenessInfo`
  - `compute_liveness_task_level(module)`
  - 基于 `TaskGraph` topo order + 扫描 task ops（value→buffer）来计算：
    - `producer_index` / `last_use_index`（task 粒度）
    - reuse key（保守：scope/dtype/shape/device/layout/strides/alignment 全匹配）

### 3) Planner passes：liveness + reuse（骨架 + 可复用函数）

- 新增：`boundflow/planner/passes/liveness_pass.py`
  - `LivenessPass`：把 `LivenessInfo` 写入 `PlanBundle.meta["liveness_info"]`
- 新增：`boundflow/planner/passes/buffer_reuse_pass.py`
  - `apply_conservative_buffer_reuse(module)`：产出 `logical_to_physical + physical_buffers`
  - `BufferReusePass`：PlannerPass 形式（为后续 pass pipeline 准备）
- 修改：`boundflow/planner/passes/__init__.py` 导出上述 pass 与 helper

### 4) Runtime env 统一使用 physical buffer id

- 修改：`boundflow/runtime/scheduler.py`
  - input/output buffer 在 env 中使用 `StoragePlan.to_physical(logical_id)`
- 修改：`boundflow/runtime/task_executor.py`
  - `PythonTaskExecutor.run_ibp_task()` 内部 `value_name -> logical_id -> physical_id`

### 5) interval_v2 可选开启复用（默认关闭）

- 修改：`boundflow/planner/interval_v2.py`
  - `IntervalV2PartitionConfig.enable_storage_reuse: bool = False`
  - 开启后调用 `apply_conservative_buffer_reuse(module)`（不改变语义，仅改变存储计划）

## 测试

- 新增：`tests/test_phase5b_pr3_buffer_reuse.py`
  - MLP：`plan_interval_ibp_v2(enable_storage_reuse=True)` 输出对齐 `plan_interval_ibp_v0`
  - 手工多 task chain：复用前后输出一致，并断言 `physical_buffers/bytes` 明显下降

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5b_pr3_buffer_reuse.py
conda run -n boundflow python -m pytest -q tests/test_phase5a_pr1_taskgraph_and_scheduler.py tests/test_phase5a_pr2_partition_multitask_equivalence.py
```

