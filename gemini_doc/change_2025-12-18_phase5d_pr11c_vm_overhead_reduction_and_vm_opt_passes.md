# 变更记录：Phase 5D PR#11C（降低 Relax VM 调用开销：VM/PackedFunc 缓存 + VM-level passes 插槽）

## 动机

PR#11A/11B 已经实现：

- task-level `RELAX_OPS` lowering（整 task 一次 VM 调用）
- 可控 fusion pipeline（LegalizeOps/Annotate/FuseOps/FuseTIR）+ `call_tir` 数量统计

下一步需要进一步降低执行侧开销，并为后续试验（如 tuple 展开、删无用参数、inline private funcs 等）预留 “不改架构即可插 pass” 的接口。

注意：本仓库使用的 TVM 版本里，`VirtualMachine.save_function(func, saved_name, *inputs)` 用于创建 **固定输入的 closure**（主要用于测试/benchmark）；因此这里更通用且更稳的方式是：

- 缓存 `VirtualMachine` 实例
- 缓存 `vm[func_name]` 取得的 PackedFunc（避免每次字典查找/dispatch）

## 本次改动

### 1) VM/PackedFunc 缓存（减少重复构建与 lookup）

- 修改：`boundflow/runtime/tvm_executor.py`
  - 新增 `TVMExecutorOptions`：
    - `enable_vm_cache: bool = True`
    - `enable_vm_packed_func_cache: bool = True`
  - `TVMTaskExecutor` 内部新增 `_vm_cache`：
    - key：`(cache_key_hash, dev.type, dev.index)`
    - value：`{"vm": VirtualMachine, "fn": PackedFunc}`
  - task-level 与 linear-level 两条路径都复用 `_get_vm_callable(...)`

### 2) VM-level passes 插槽（可插拔，不改主流程）

- 修改：`boundflow/runtime/tvm_executor.py`
  - `TVMExecutorOptions.task_vm_opt_passes: Tuple[str,...] = ()`
  - task-level compile pipeline 末尾可追加这些 pass（字符串表示，便于 JSON/bench 记录）
  - 常用候选（TVM Relax 内置）：`ExpandTupleArguments/RemoveUnusedParameters/InlinePrivateFunctions/CallTIRRewrite`

### 3) 测试：开启缓存与 pass 插槽仍保持语义等价

- 新增：`tests/test_phase5d_pr11c_vm_cache_and_opt_passes.py`
  - 开启 `enable_vm_cache/enable_vm_packed_func_cache`
  - 开启 `task_vm_opt_passes`
  - scheduler 下跑两次，断言与 Python reference allclose，且两次 TVM 输出一致

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11c_vm_cache_and_opt_passes.py
conda run -n boundflow python -m pytest -q
```

