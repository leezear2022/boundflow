---
status: implemented-formal-run-pending
updated: 2026-08-26T12:38:00+08:00
type: change
topic: boundflow
slug: r3-3-isolated-attribution-implementation
stage: s01
---

# R3-3 Isolated Attribution 实现变更记录

## 1. 变更范围

本轮只实现已预注册的只读归因与 replay 基础设施，没有修改 production TIR、schedule、
runtime 数学语义或 optimizer：

- `r3_3_isolated_attribution.py`：规范化 profiler marker/CUDA event、互斥区间账本、
  守恒/校准/扰动门禁和 fail-closed route；
- 单 worker：5 个 capture 各自独立进程，10 warmup、30 个无 profiler CUDA-event sample、
  1 个带 marker 的 diagnostic capture；
- artifact/replay：冻结 source blob、capture digest、原始 event、逐 worker ledger 和 route；
- tamper：预注册 12 类 fully re-signed 篡改；
- worker stdout/stderr 在写入 artifact 前确定性替换 repo/Python prefix，replay 对任何遗留本机
  路径 fail closed；
- 单元测试：event round-trip、互斥守恒、profiler 扰动拒绝、route 优先级及本机路径泄漏。

## 2. 预正式运行观察

单 worker smoke 中，无 profiler candidate median 约 `1.40 ms`，带 profiler 的 CUDA-event
wrapper 约为其 `2.77x`；同时 calibration event 与归因 kernel union 的残差超过冻结阈值。
因此该 smoke 的 ledger 正确输出：

`attribution_admitted=false`，failure=`calibration-residual, profiler-perturbation`。

这不是性能结论。其 bucket share 只能作为不具准入资格的诊断投影，不能形成 KERNEL、BRIDGE、
AUTOGRAD 或 CUMULATIVE route。正式 five-fresh 必须使用干净、已提交 source 重跑；若同样失败，
冻结结果应为 `STOP-attribution-quality`。

## 3. Claim 边界

- `performance_claimed=false`；
- `r3_4_open=false`；
- `same_solver_open=false`；
- 本轮 share 不是 query/queue share；
- raw 后不得放宽 `1.20x` profiler 扰动门槛，也不得新增组合 bucket。

## 4. 验证记录

预提交验证：归因单元测试 `5 passed`；artifact 测试在正式 artifact 生成前按设计 skip；
mypy/pylint/正式 replay/tamper/full regression 将在 source commit 与 formal raw 后记录。
