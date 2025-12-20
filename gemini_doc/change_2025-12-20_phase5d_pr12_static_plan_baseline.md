# 变更记录：Phase 5D PR#12（StaticPlanBlockMemory baseline × BoundFlow reuse：可控开关 + memory stats + 四象限 bench）

## 动机

Phase 5D 进入“系统论文消融”阶段后，审稿人一定会问：

> BoundFlow 的 logical→physical reuse，和 TVM Relax 自带的 `StaticPlanBlockMemory` 有什么差异？是不是 TVM 自己就能做掉？

要回答这个问题，必须把两种机制放进同一套实验矩阵：

- **BoundFlow reuse（跨 task DAG / inter-task）**：planner 层面 logical→physical 映射。
- **TVM StaticPlanBlockMemory（单个 Relax function 内 / intra-function）**：编译器内静态内存规划。

因此 PR#12 做三件事：

1) 让 TVM 侧的 static memory planning **可控开关**（ON/OFF）；
2) 把 TVM 侧的“内存规划效果”变成 **可计量的 stats**（alloc_storage bytes 等）；
3) 提供一个 **四象限 bench**，直接输出一张表所需字段。

---

## 本次改动

### 1) TVMTaskExecutor：memory planning 开关（12A）

- 修改：`boundflow/runtime/tvm_executor.py`
  - 新增 `MemoryPlanMode`：
    - `DEFAULT`：使用 TVM 默认 build pipeline（包含 `StaticPlanBlockMemory`）
    - `DISABLE_STATIC_PLAN`：移除 `StaticPlanBlockMemory`（保留其它必要 pass，确保 VM codegen 能跑）
    - `FORCE_STATIC_PLAN`：占位（当前等价于 DEFAULT 的“显式开启”语义）
  - `TVMExecutorOptions.memory_plan_mode`：纳入 task-level compile cache key，避免不同模式误命中。
  - task-level 编译不再完全依赖 `tvm.relax.pipeline.default_build_pipeline()` 的黑箱：
    - 直接按 TVM 源码里的 pass 列表组合出“等价默认 pipeline”，但允许跳过 `StaticPlanBlockMemory`。

### 2) TVM compile_stats：memory stats（12B）

- 修改：`boundflow/backends/tvm/relax_analysis.py`
  - 新增 `collect_relax_memory_stats(ir_mod)`：统计
    - `relax.vm.alloc_storage` / `relax.vm.alloc_tensor`（及兼容 op 名）
    - `alloc_storage_total_bytes/max_bytes`（按 shape×dtype 估算；用于相对消融，不追求精确峰值）
- 修改：`boundflow/runtime/tvm_executor.py`
  - task-level `compile_stats` 增加：
    - `memory_plan_mode`
    - `memory_stats`

### 3) 四象限 bench（12C）

- 新增：`scripts/bench_static_plan_baseline.py`
  - 默认跑 4 组：
    - BoundFlow reuse OFF/ON × TVM static plan OFF/ON
  - 输出 JSON（可直接转 CSV/画图）：
    - planner：`num_tasks/num_edges/storage_plan(logical/physical)/reuse_stats/timings/config_dump`
    - tvm：`memory_plan_mode` + task-level `alloc_storage_total_bytes` 汇总
    - runtime：`run_ms_avg`（post-compile）

---

## 测试

- 新增：`tests/test_phase5d_pr12_static_plan_modes.py`
  - `MemoryPlanMode.DEFAULT` / `DISABLE_STATIC_PLAN` 两种模式均可编译执行，并与 `PythonTaskExecutor` allclose。

---

## 如何验证

```bash
# 单测
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr12_static_plan_modes.py

# 全量回归
conda run -n boundflow python -m pytest -q

# 四象限 bench（建议加 --no-capture-output 观察输出）
conda run --no-capture-output -n boundflow python scripts/bench_static_plan_baseline.py --min-tasks 2 --warmup 1 --iters 20
```

---

## 备注/已知限制

- `collect_relax_memory_stats` 的 byte 统计是“IR 侧估算”（按 `alloc_storage` shape×dtype），用于趋势对比；不等价于真实峰值显存。
- `MemoryPlanMode.FORCE_STATIC_PLAN` 目前主要是语义占位（方便后续 pipeline 插拔/对齐不同 TVM 版本行为）。
