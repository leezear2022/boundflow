# 变更记录：Phase 5D PR#11C.1（save_function bench + 修复 task-level pipeline：组合 default_build_pipeline）

## 动机

在 PR#11A/11B/11C 基础上，为了进一步做“VM overhead”消融与稳定 fusion：

1) 增加一个 **save_function closure** 的 micro-bench：对比 `vm[func](*args)` / cached packedfunc / `save_function` closure 的调用开销；
2) 修复 task-level 自定义 relax_pipeline：此前用纯自定义 pass 序列会绕开 TVM 默认 build pipeline，导致 VM codegen 可能报错（例如 `CodeGenVM cannot handle relax.builtin.alloc_tensor`）。正确做法应 **组合官方 `default_build_pipeline()`**，在其前面插入我们可控的 fusion/额外 pass。

## 本次改动

### 1) task-level pipeline 修复：pre-pass + default_build_pipeline

- 修改：`boundflow/runtime/tvm_executor.py`
  - task-level 编译使用：
    - pre-pass（可选）：`Normalize/FoldConstant/LegalizeOps/ConvertToDataflow/Annotate/FuseOps/FuseTIR/...`
    - + `tvm.relax.pipeline.default_build_pipeline()`（LowerAllocTensor/StaticPlanBlockMemory/AttachGlobalSymbol/...）
  - 避免 “绕开默认 pipeline 导致 VM codegen 不支持 alloc_tensor” 的问题
  - fusion stats 的 `after_fuse_ops/after_fuse_tir` 统计链路也补上 `ConvertToDataflow()`

### 2) task-level RELAX_OPS lowering 修复：避免把 param 当作 interval 输入

- 修改：`boundflow/backends/tvm/relax_interval_task_ops.py`
  - 从 `StoragePlan` 的 buffer `scope` 推断 param/const 值，确保：
    - `task.input_values` 中的 param 不会生成 `(p_l,p_u)` 两 lane 输入
    - param 只以 plain tensor 形式出现在函数参数列表

### 3) save_function micro-bench + 回归测试

- 新增：`scripts/bench_relax_vm_overhead.py`
  - 输出 JSON：对比三种调用方式的平均耗时：
    - `vm_lookup_each_time`
    - `vm_cached_packedfunc`
    - `vm_save_function_closure`
- 新增：`tests/test_phase5d_pr11c1_save_function_closure.py`
  - 断言 `vm.save_function(...); vm[saved]()` 的输出与直接调用一致

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11c1_save_function_closure.py
conda run -n boundflow python -m pytest -q

# micro-bench（输出 json）
conda run -n boundflow python scripts/bench_relax_vm_overhead.py --iters 200 --warmup 20
```

