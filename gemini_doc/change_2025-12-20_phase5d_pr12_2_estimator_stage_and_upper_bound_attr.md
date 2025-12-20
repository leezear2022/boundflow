# 变更记录：Phase 5D PR#12.2（estimator_stage 固定 + 写入 tir_var_upper_bound attrs + dynamic 回归用例）

## 动机

在 PR#12.1 已经引入 `by_scan` 与 `by_tvm_estimator` 双口径后，还需要两颗“闭环钉子”，避免后续 pass 顺序/动态形状引入导致数据漂移且不可解释：

1) **固定 estimator 的调用阶段并记录**：`estimate_memory_usage` 在不同 pipeline 阶段看到的 alloc 形态不同，若不记录 stage，会把“阶段变化”误判为“优化效果”。
2) **让 tir_var_upper_bound 真正生效**：不仅记录 options，而是把它写入 Relax function attrs，并提供一个最小 dynamic 回归用例证明它会影响 memory planning 的可测统计。

## 本次改动

### 1) compile_stats 增加 `by_tvm_estimator_stage`

- 修改：`boundflow/runtime/tvm_executor.py`
  - `compile_stats[...]["memory_stats"]` 增加字段：`by_tvm_estimator_stage`（当前固定为 `"pre_static_plan"`）
  - `by_tvm_estimator` 的调用时机保持不变（Dispatch/Legaize/.../CallTIRRewrite 之后、StaticPlanBlockMemory 之前），但显式记录，避免未来漂移。

### 2) 写入 `tir_var_upper_bound` 到 Relax function attrs

- 修改：`boundflow/runtime/tvm_executor.py`
  - 当 `TVMExecutorOptions.tir_var_upper_bound` 非空时：在 task-level IRModule 上对 `main` Relax function 写入 `tir_var_upper_bound` attrs（best-effort，不阻塞编译）。

### 3) 新增 dynamic-shape 回归用例（证明 upper bound 会影响可测 bytes）

- 新增：`tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py`
  - 构造一个带动态维度 `n` 的 Relax module（`alloc_tensor(shape=[n,16])`）
  - 对比：
    - 无 upper bound：`collect_relax_memory_stats` 的 `alloc_storage_total_bytes == 0`（符号 size 无法折算）
    - 有 upper bound：`alloc_storage_total_bytes > 0` 且 `alloc_storage_nonconst_bytes` 下降
  - 用 TVM 官方 `default_build_pipeline()` 触发 memory planning 与 lowering（走 `relax.vm.alloc_storage` 形态）。

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py
conda run -n boundflow python -m pytest -q
```

## 备注

- 该 dynamic 用例的目标是证明：`tir_var_upper_bound` 能把部分 alloc_storage 从“符号 size”变成“可折算常量 bytes”（趋势可观测），并不等价于真实峰值显存。
