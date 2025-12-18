# 变更记录：Phase 5D PR#11B（可控 fusion pipeline + call_tir 数量统计）

## 动机

PR#11A 已经把 compute task 以 “整 task RELAX_OPS lowering → 一次 VM 调用执行” 跑通并对齐 Python reference。

但要写系统消融，需要把 “RELAX_OPS → call_tir 数量下降（fusion）” 变成一个 **可控编译开关**，并且在编译统计里记录：

- Legalize 后产生的 `call_tir` 数量（每 op 一个 PrimFunc 的 baseline）
- FuseOps/FuseTIR 后的 `call_tir` 数量（融合后调用次数下降）

这样才能把 “planner timing vs TVM compile per-pass timing vs runtime overhead” 的证据链补齐。

## 本次改动

### 1) 新增 Relax IR 统计工具

- 新增：`boundflow/backends/tvm/relax_analysis.py`
  - `count_relax_call_tir(ir_mod)`
  - `collect_relax_ir_stats(ir_mod)`：输出 `{relax_funcs, tir_funcs, call_tir}`

### 2) TVMTaskExecutor：task-level compile pipeline 可控 + IR stats 落盘

- 修改：`boundflow/runtime/tvm_executor.py`
  - `TVMExecutorOptions` 新增：
    - `enable_task_fusion_pipeline: bool = False`
    - `task_fuse_opt_level: int = -1`
  - task-level `RELAX_OPS` compile 时使用自定义 pipeline（仍由 `relax.build(..., relax_pipeline=...)` 执行）：
    - `Normalize → FoldConstant → LegalizeOps → (AnnotateTIROpPattern → FuseOps → FuseTIR) → DCE → RemoveUnusedOutputs → LambdaLift → Normalize`
    - fusion 通过开关控制
  - `compile_stats` 增加 `ir_stats`（best-effort）：
    - `before`
    - `after_legalize`
    - `after_fuse_ops`
    - `after_fuse_tir`

### 3) 回归测试：fusion 开启仍对齐 + 统计满足单调性

- 修改：`tests/test_phase5d_pr11a_task_relax_ops_equiv.py`
  - 开启 `enable_task_fusion_pipeline=True`
  - 断言 allclose 仍成立
  - 若 `after_legalize/after_fuse_tir` 统计可用，则断言 `call_tir_after_fuse_tir <= call_tir_after_legalize`

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11a_task_relax_ops_equiv.py
conda run -n boundflow python -m pytest -q
```

