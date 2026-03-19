# 变更记录：Phase 6A 起步——引入 InputSpec + LpBallPerturbation（L∞/L2）与线性 concretize

## 动机

Phase 5 的 IBP 主线把输入扰动固定成 `LinfInputSpec(center, eps)`（逐元素 box）。为推进 Phase 6 的 CROWN/αβ-CROWN/BaB，需要先把“扰动集合”从 IBP 管线中解耦出来，让：

- 扰动类型（L∞/L2/…）成为 running context；
- 线性形式的数值化（`concretize`）由扰动集合统一提供；
- 在不破坏 Phase 5 测试与接口的前提下，先把最小可行的 `L2` 输入支持跑起来（至少对线性层给出正确 deviation 公式）。

## 主要改动

- 新增：`boundflow/runtime/perturbation.py`
  - `PerturbationSet` 协议（`bounding_box`、`concretize_matmul`）。
  - `LpBallPerturbation(p∈{∞,2,1}, eps)`：实现对偶范数形式的线性 concretize（当前优先使用 `p=∞/2`）。
  - `InputPerturbationState(center, perturbation)`：用于把“输入=中心+集合”作为一种 DomainState 挂进 executor。

- 更新：`boundflow/runtime/task_executor.py`
  - 新增 `InputSpec(value_name, center, perturbation)`，并保留 `LinfInputSpec` 兼容旧测试。
  - `PythonTaskExecutor.run_ibp` 支持 `InputSpec`：
    - 对 `linear`：若输入为 `InputPerturbationState`，走 `perturbation.concretize_matmul(...)` 得到输出区间。
    - 对 `conv2d` 等：当前先把输入降级到 box（`bounding_box`）再走现有 interval 传播（sound，但不 tight）。

- 更新：`boundflow/runtime/scheduler.py`、`boundflow/runtime/tvm_executor.py`、`boundflow/runtime/executor.py`
  - 允许接受 `InputSpec` 类型，但 **scheduler/TVM 路径当前仍只支持 L∞（box）**；非 L∞ 会抛 `NotImplementedError`（避免 silent 的错误口径）。

- 新增测试：`tests/test_phase6a_inputspec_lpball_linear.py`
  - `L2` 输入扰动下的线性层 soundness（采样验证）。
  - `InputSpec.linf` 与 legacy `LinfInputSpec` 在线性层上一致性回归。

## 如何验证

```bash
python -m pytest -q tests/test_phase6a_inputspec_lpball_linear.py
```

## 备注与后续

- 这是 Phase 6 的“接口地基”改动：为后续 CROWN-IBP 的 `PerturbationSet.concretize(A, x0)`（可能需要 `LinearOperator`）留出了位置，但当前只实现了最小的 matmul 形态。
- conv2d 的 `L2` tighten 暂未做；目前采用 sound 的 box 外包策略（见设计文档 `gemini_doc/perturbation_support_design.md`）。
- 对 `IBP/CROWN/αβ/BaB` 的工程落地评审与避坑清单（无外链版）：`gemini_doc/phase6_review_three_axis_stage_pipeline.md`。
