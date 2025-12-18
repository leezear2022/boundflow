# 变更记录：Phase 5D PR#9（TVMTaskExecutor：compile cache + run_ibp_task 对齐 scheduler）

## 动机

PR#8 已经把 “interval linear task → Relax IRModule（RELAX_OPS/CALL_TIR）” 的 lowering 骨架打通，并验证了 IRModule 可 build。

PR#9 的目标是把这条链路接到 runtime：在 **Phase 5 TaskGraph/scheduler（physical env）** 语义下，让 TVM executor 能执行 task，并与 Python reference executor 对齐（allclose）。

## 本次改动

### 1) key 驱动的 Relax lowering helper（避免 runtime 依赖 BFTaskModule/BoundTask）

- 修改：`boundflow/backends/tvm/relax_task_lowering.py`
  - 新增 `build_interval_linear_relax_ir_module(key, mode, relax_func_name)`
  - `CALL_TIR` 路径为 TIR PrimFunc 设置唯一 `global_symbol`，避免与 Relax 函数名冲突（TVM build 要求 global_symbol 唯一）

### 2) TVMTaskExecutor 实现 run_ibp_task（对齐 scheduler 的 physical env contract）

- 修改：`boundflow/runtime/tvm_executor.py`
  - 新增 `run_ibp_task(task, env, params, storage_plan)`：
    - env 为 **physical buffer id → IntervalState**（与 Phase 5B hardening 一致）
    - `linear`：使用 Relax VM（kernel_style=`relax` 或 `call_tir`）加速
    - `relu/add/mul`：回退到 IntervalDomain（保持语义一致）
  - 新增 `_compile_interval_linear_executable()`：对 `(kernel_style, IntervalLinearKey)` 做编译缓存（避免每次运行都编译）

### 3) 测试：scheduler 下 Python vs TVM allclose

- 新增：`tests/test_phase5d_pr9_tvm_executor_linear_equiv.py`
  - 在 `plan_interval_ibp_v2(min_tasks=1)` + `run_ibp_scheduled()` 下，对比：
    - `PythonTaskExecutor` vs `TVMTaskExecutor(kernel_style=relax|call_tir)`
  - 断言 `lower/upper` allclose

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr9_tvm_executor_linear_equiv.py
conda run -n boundflow python -m pytest -q
```

