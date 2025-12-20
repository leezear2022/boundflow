# 变更记录：Phase 5D PR#12.1（接入 estimate_memory_usage + 明确 DEFAULT pipeline 边界 + 预留 tir_var_upper_bound）

## 动机

PR#12 已经能做四象限对照（BoundFlow reuse × TVM StaticPlanBlockMemory），但为了让证据链更“论文友好”，需要：

1) 同时提供 TVM 官方口径的 memory estimate（作为佐证/对照）；
2) 明确 `MemoryPlanMode.DEFAULT` 的实现边界：默认行为尽量走 TVM 官方 `default_build_pipeline()`，减少“我们手写 pipeline 漂移”的风险；
3) 为 dynamic shape 的 memory planning（`tir_var_upper_bound`）预留可控变量，并写入 bench/config（哪怕暂不启用），避免后续引入导致历史数据不可比。

## 本次改动

### 1) compile_stats 增加 TVM 官方估算（estimate_memory_usage）

- 修改：`boundflow/runtime/tvm_executor.py`
  - task-level compile 产物 `compile_stats[...]["memory_stats"]` 结构调整为：
    - `by_scan`：基于 `relax.vm.alloc_storage/alloc_tensor` 的结构化统计（PR#12 方案）
    - `by_tvm_estimator`：`tvm.relax.analysis.estimate_memory_usage(...)` 的原文字符串（best-effort）

### 2) DEFAULT pipeline 边界更清晰

- 修改：`boundflow/runtime/tvm_executor.py`
  - `MemoryPlanMode.DEFAULT/FORCE_STATIC_PLAN`：直接使用 TVM 官方 `tvm.relax.pipeline.default_build_pipeline()`
  - `MemoryPlanMode.DISABLE_STATIC_PLAN`：使用“等价默认 pipeline 但移除 StaticPlanBlockMemory”的自定义 pass 列表

### 3) 预留 tir_var_upper_bound（暂不启用）

- 修改：`boundflow/runtime/tvm_executor.py`
  - 新增 `TVMExecutorOptions.tir_var_upper_bound: Optional[Dict[str,int]] = None`
  - 纳入 task-level compile cache key 与 compile_stats（即使为空也记录）
- 修改：`scripts/bench_static_plan_baseline.py`
  - 输出字段中增加 `tir_var_upper_bound`（当前为 null）

### 4) 测试与 bench 兼容更新

- 修改：`tests/test_phase5d_pr12_static_plan_modes.py`
  - 适配 `memory_stats` 的新结构（检查 `by_scan/by_tvm_estimator`）
- 修改：`scripts/bench_static_plan_baseline.py`
  - 汇总字段改为 `compile_ms_total`

## 如何验证

```bash
conda run -n boundflow python -m pytest -q
conda run --no-capture-output -n boundflow python scripts/bench_static_plan_baseline.py --iters 1 --warmup 1
```

## 备注

- `estimate_memory_usage` 是 TVM 官方字符串输出，作为“趋势/口径对照”；由于我们最终 module 经 `LowerAllocTensor` 进入 `relax.vm.alloc_storage` 形态，结构化统计仍以 `by_scan` 为主。
- `tir_var_upper_bound` 目前仅做 schema+记录，占位用于后续 dynamic shape/planning 消融。
