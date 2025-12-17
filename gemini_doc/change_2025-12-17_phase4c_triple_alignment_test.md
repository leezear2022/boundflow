# 变更记录：Phase 4C 三方对齐测试（auto_LiRPA / PythonTaskExecutor / TVMTaskExecutor）

## 背景与动机

目前已有测试 `tests/test_phase4c_tvmexecutor_matches_python.py`，它验证了 `TVMTaskExecutor` 的输出与 `PythonTaskExecutor` 一致。

但这个链路缺少 “对外部权威实现” 的对齐点：我们还需要把 auto_LiRPA 拉进来，形成：

- auto_LiRPA（IBP 参考实现）
- BoundFlow PythonTaskExecutor（reference 执行器）
- BoundFlow TVMTaskExecutor（TVM kernel demo + fallback）

三方闭环后，能更清晰回答两个问题：

1. `PythonTaskExecutor` 的 IBP 语义是否与 auto_LiRPA 一致（端到端正确性）
2. `TVMTaskExecutor` 是否真正做到 “加速路径输出 == reference 输出”（后端正确性）

## 本次改动

新增测试用例：`tests/test_phase4c_tvmexecutor_against_auto_lirpa.py`

- 网络：MLP（`Linear → ReLU → Linear`）
- 扰动：`L∞`（`eps=0.05`）
- 对齐断言：
  - `PythonTaskExecutor` 输出 `lower/upper` 与 auto_LiRPA `compute_bounds(method="IBP")` 的 `lb/ub` allclose
  - `TVMTaskExecutor` 输出 `lower/upper` 与 `PythonTaskExecutor` allclose
- 运行条件：
  - 若环境中缺少 `tvm` 或 `auto_LiRPA`，测试会自动 skip
  - 若 `tvm.runtime.enabled("llvm")` 为 false（没有可用的 CPU target），测试会 skip

## 如何验证

在 conda 环境 `boundflow` 下运行：

```bash
conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_against_auto_lirpa.py
```

期望结果：`1 passed`（若缺少依赖则为 `skipped`）。

