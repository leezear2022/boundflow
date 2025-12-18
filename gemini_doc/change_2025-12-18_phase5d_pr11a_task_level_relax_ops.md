# 变更记录：Phase 5D PR#11A（compute task 全 TVM：RELAX_OPS 覆盖 linear+relu(+add/mul)）

## 动机

PR#9 已经跑通 “lowering → build（带 cache）→ scheduler 以 physical env 执行 task” 的后端链路闭环，但 task 内仍是 “linear→TVM，其它 op 回退 Python/IntervalDomain”，会产生：

- task 内多次 Python↔TVM 往返与解释器开销；
- 难以在后续做“call_tir 数量下降 / fusion 收益”这类系统消融。

因此先落地 PR#11A：把一个 compute task 内的 interval ops 用 **Relax 高层算子（RELAX_OPS）**统一表达为一个 Relax function，让 task 内尽量 “一次 VM 调用完成”。

该路径定位为 reference/debug-friendly：语义更好对齐，后续 PR#11B 再引入 fusion pipeline（LegalizeOps/Annotate/FuseOps/FuseTIR）减少 `call_tir` 数量。

## 本次改动

### 1) 新增：task-level RELAX_OPS lowering（lane 拆分契约）

- 新增：`boundflow/backends/tvm/relax_interval_task_ops.py`
  - `build_interval_task_relax_ops_ir_module(task, storage_plan, target, func_name)`
  - v0 支持 op：`linear/relu/add/mul/permute/reshape`
  - interval lane contract：每个 value 表示为 `(lower, upper)` 两个 tensor；函数签名为：
    - inputs：对每个 `task.input_values` 传 `(v_l, v_u)`
    - params：对每个 `task.params` 传 plain tensor
    - outputs：按 `task.output_values` 返回扁平 tuple `[o0_l,o0_u,o1_l,o1_u,...]`

### 2) TVMTaskExecutor：新增“整 task 执行”开关（默认不影响旧行为）

- 修改：`boundflow/runtime/tvm_executor.py`
  - `TVMExecutorOptions.enable_task_relax_ops: bool = False`
  - 当 `enable_task_relax_ops=True` 且 `kernel_style="relax"` 时：
    - 尝试将整个 task 编译成一个 Relax function 并执行
    - 编译缓存使用 task-level signature hash（避免重复编译）
    - 若 lowering 不支持则自动回退到旧的 per-op 执行（保持鲁棒）

### 3) 测试：scheduler 下 TVM vs Python allclose

- 新增：`tests/test_phase5d_pr11a_task_relax_ops_equiv.py`
  - 模型：`Linear + ReLU`
  - `plan_interval_ibp_v2(min_tasks=1) + run_ibp_scheduled(...)`
  - `PythonTaskExecutor` vs `TVMTaskExecutor(enable_task_relax_ops=True)` 输出 allclose

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11a_task_relax_ops_equiv.py
conda run -n boundflow python -m pytest -q
```

