# 变更记录：PR#14A（Phase 5）mnist_cnn workload + Relax task-level interval conv2d/flatten

## 动机

Phase 5 的矩阵消融如果只覆盖 MLP，论文/AE 容易被质疑“只在 toy GEMM 上成立”。本次补齐一个最小 CNN 类 workload（conv2d/relu/flatten/linear），并让 `TVMTaskExecutor` 的 **task-level RELAX_OPS** lowering 覆盖 `conv2d/flatten`，以便：

- CNN 也能进入 `bench_ablation_matrix.py` 的同一套 schema/产线；
- multi-task DAG 的 schedule + reuse/static_plan/fusion 等 knob 在 CNN 上也能被系统化消融；
- TVM compile/run/memory/call_tir 统计链路保持一致（不落回 Python-only）。

## 本次改动

- 更新：`boundflow/backends/tvm/relax_interval_task_ops.py`
  - Relax task lowering 追加支持 `conv2d` 与 `flatten`：
    - `conv2d` 采用 NCHW/OIHW（v0），并按 IBP 公式用 `w_pos/w_neg` 做 4 次 conv 组合（参考现有 `boundflow/backends/tvm/relax_interval_conv2d.py` 的写法）。
    - `flatten` 直接按 `storage_plan` 的输出 shape 做 `reshape`（保持语义与 planner/storage plan 一致）。
  - 文档注释同步更新 “supported ops” 列表。

- 更新：`scripts/bench_ablation_matrix.py`
  - 新增 workload：`mnist_cnn`（conv2d + relu + flatten + linear）。
  - `--batch/--eps` 改为对所有 workload 生效（不再仅限 mlp）。

- 更新：`docs/bench_jsonl_schema.md`
  - Workload 参数化章节增加 `mnist_cnn` 并同步 `--batch/--eps` 适用范围。

- 新增测试：
  - `tests/test_phase5d_pr14a_cnn_workload_no_tvm_smoke.py`：无 TVM 环境下的 bench 冒烟（`--no-tvm`），确保 CNN workload 可跑且仍按 schema 写行。
  - `tests/test_phase5d_pr14a_cnn_workload_tvm_smoke.py`：有 TVM 环境时的 bench 冒烟（`pytest.importorskip("tvm")`），确保 TVM 模式下 `runtime.compile_first_run_ms/run_ms_p50` 等字段被填充。

## 与 Phase 5E 产线收口的衔接

- multi-workload artifact runner（`--workload all`）与 workload 分色图在后续 PR#14B 中落地。

## 如何验证

- 无 TVM 环境：
  - `python -m pytest -q tests/test_phase5d_pr14a_cnn_workload_no_tvm_smoke.py`
- 有 TVM 环境：
  - `python -m pytest -q tests/test_phase5d_pr14a_cnn_workload_tvm_smoke.py`
