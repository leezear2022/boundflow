# 2026-03-19：Phase 6G 零 split-state detector 短路修正

## 背景

在整理并验证剩余未提交改动时，`tests/test_phase6g_branch_pick_reuses_forward_trace.py` 失败：

- 期望：alpha-beta oracle 在 root 节点只跑一次 `_forward_ibp_trace_mlp(...)`
- 实际：跑了两次

根因不是 branch picking 本身重复做 forward，而是：

- `run_alpha_beta_crown_mlp(...)` 的 first-layer infeasible detector 在 `split_state` 全为 0 时也会触发
- detector 内部为了收集 halfspace 会先跑一次 `_forward_ibp_trace_mlp(...)`
- root node 的 `ReluSplitState.empty(...)` 是“字典非空但所有条目都为 0”，因此被误判成“有 split”

## 修正

更新：

- `boundflow/runtime/alpha_beta_crown.py`

新增：

- `_has_nonzero_split_state(...)`

调整：

- `_collect_first_layer_split_halfspaces(...)`
  - 当 `relu_split_state` 为空，或者所有 split 都为 0 时，直接返回空列表，不再先跑 forward trace
- `run_alpha_beta_crown_mlp(...)`
  - `do_infeasible_check` 从 `batch_size == 1 and bool(split_state)` 改成
    `batch_size == 1 and _has_nonzero_split_state(split_state)`

## 结果

- root node 的空 split 不再额外触发 detector
- `branch_choices` 继续复用 alpha-beta oracle 自己的 forward trace
- Phase 6G 的“branch pick 不重复 forward”回归恢复通过

## 验证

已执行：

```bash
conda run -n boundflow python -m pytest -q tests/test_phase6g_branch_pick_reuses_forward_trace.py
conda run -n boundflow python -m pytest -q tests/test_env.py tests/test_phase4d_onnx_frontend_matches_torch.py tests/test_phase5d_pr8_relax_lowering_skeleton.py tests/test_phase5d_pr9_tvm_executor_linear_equiv.py tests/test_phase5d_pr10_tvm_compile_instruments.py tests/test_phase5d_pr11a_task_relax_ops_equiv.py tests/test_phase5d_pr11c1_save_function_closure.py tests/test_phase5d_pr11c_vm_cache_and_opt_passes.py tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py tests/test_phase5d_pr12_static_plan_modes.py tests/test_phase6c_crown_ibp_multispec_batch.py tests/test_phase6g_bab_node_batch.py tests/test_phase6g_bab_node_eval_cache.py tests/test_phase6g_branch_pick_reuses_forward_trace.py tests/test_phase6g_node_batch_grad_isolation.py tests/test_phase6g_node_batch_partial_infeasible_prune.py tests/test_phase6h_artifact_runner_smoke.py tests/test_phase6h_bench_e2e_schema.py tests/test_phase6h_plot_smoke.py tests/test_phase6h_report_csv_schema.py tests/test_phase6h_workload_suite_smoke.py
```

结果：

- `tests/test_phase6g_branch_pick_reuses_forward_trace.py`: `1 passed`
- 剩余 Phase4D/5D/6C/6G/6H 验证集合：`26 passed`
