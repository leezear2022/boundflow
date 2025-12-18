# 变更记录：Phase 5D PR#8（Task → Relax IRModule lowering skeleton：interval linear）

## 动机

在 Phase 5C（PR#5~PR#7）已经把 planner pipeline 的“统一入口 + config_dump + verifier/instrument + determinism/dump”钉住后，Phase 5D 的主线是把 **编译后端链路**拉起来。

本 PR 先实现最小 lowering skeleton：把一个 **只含 `linear` 的 interval-IBP task** lower 成 **Relax IRModule**，并提供两种风格：

- `RELAX_OPS`：只用 Relax 高层算子（由 TVM 内部完成后续 legalize/lower）
- `CALL_TIR`：显式生成 TIR `PrimFunc`，并用 `relax.call_tir` 调用（mixed module 的典型形态）

该 skeleton 为 PR#9（TVMExecutor v0：compile + execute 与 PythonTaskExecutor 对齐）预留稳定接口。

## 本次改动

### 1) 新增：Task-level Relax lowering

- 新增：`boundflow/backends/tvm/relax_task_lowering.py`
  - `RelaxLoweringMode`：`RELAX_OPS | CALL_TIR`
  - `lower_interval_linear_task_to_relax_ir(task, module, target, mode)`：
    - v0 仅支持 **single-op task**：`linear(x, w, b) -> y`
    - interval lane 显式拆成 `(x_l, x_u) -> (y_l, y_u)`（与当前 TVMExecutor 的 data model 一致）
    - 输出 `RelaxLoweringResult(ir_mod, relax_func_name, mode)`

### 2) 新增：可复用的 TIR PrimFunc（用于 call_tir）

- 修改：`boundflow/backends/tvm/interval_linear.py`
  - 新增 `build_interval_linear_primfunc(key)`：返回 TIR `PrimFunc`（用于 `relax.call_tir` lowering）
  - 保留现有 `build_interval_linear_module(key)`（legacy/TE demo）

### 3) 测试：IRModule 可构建 + 可编译

- 新增：`tests/test_phase5d_pr8_relax_lowering_skeleton.py`
  - `RELAX_OPS`/`CALL_TIR` 两种 mode 均能构建 Relax IRModule
  - `CALL_TIR` mode 的 IRModule 可被 `relax.build(..., target="llvm")` 编译

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr8_relax_lowering_skeleton.py
conda run -n boundflow python -m pytest -q
```

