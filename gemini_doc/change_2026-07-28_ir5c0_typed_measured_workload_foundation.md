# 变更记录：IR-5C0 typed measured workload foundation

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`e9df1d8`（IR-5B fair policy evaluator）
> 状态：真实测量 workload 基础完成；measurement/held-out artifact 仍 pending

## 改动

- 新增正式 `typed_benchmark_workloads`，不再依赖测试私有 helper；
- 可按 seed/dimensions/device 构造确定性 plain-CROWN MLP；
- 同一语义可生成 reference、dense/chunked/TVM fused typed
  BoundModule→PlanTemplate→PlanInstance→TaskIR→ScheduleIR；
- workload 对外返回所有 compiler/runtime typed inputs，供 cold/warm/peak 测量复用；
- evaluator 将 `predicted_compile_ms` 与 `measured_compile_ms` 分离，global selection 不再
  偷看 held-out measured compile。

## 验证与边界

- reference 与 PyTorch dense typed candidate 实际执行 final lower/upper 完全一致；
- 两条候选具有不同 PlanInstance hash；
- workload/evaluator Mypy 0 issues；
- 定向 3 passed。

本切片没有计时结果，不是 held-out artifact。下一步继续实现 calibration-only prediction、
CUDA synchronization/peak measurement、frozen split 和 fresh-process semantic replay。

