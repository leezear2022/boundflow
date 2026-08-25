---
status: implemented-formal-run-pending
updated: 2026-08-26T13:30:00+08:00
type: changelog
topic: boundflow
slug: mr0-low-perturbation-event-budget
stage: s01
---

# BoundFlow MR0 低扰动显式事件预算变更记录

## Summary

- R3-3 profiler attribution按冻结门禁 STOP 后，没有放宽阈值或采信非准入 share；
- 新建独立 MR0，只校准17-op显式 CUDA-event record 的物理扰动；
- 最大预算通过也只开放 internal-boundary correctness，不直接开放 share/same-solver/优化。

## Frozen decisions

- 主 workload 沿用已批准 CIBC ResNet2B graph，不使用 R3 wrapper 投影选方向；
- event budget=`1/4/8/17`，正式决策只看17；
- five-fresh、20 paired group、100 replay/group、CI/IC交错；
- GO=`geomean<=1.05/bootstrap upper<=1.05/worst<=1.08`；
- `performance_claimed=false`，MR1/R2/same-solver 默认关闭。

## Validation pending

实现已新增：

- `boundflow/runtime/mr0_explicit_event_budget.py`：budget row、geomean、bootstrap upper 与机械
  verdict重算；
- fresh worker：真实CIBC 17-op CUDA Graph、预分配36个event object、1/4/8/17对逐次记录；
- artifact/replay：source blob、input digest、portable log、5 raw与summary绑定；
- 12类fully re-signed tamper框架与专项测试。

提交前单worker smoke保持semantic exact、36 tensor/235992元素、pointer/contract/stream稳定；非正式
overhead ratio随budget=`1.0618/1.1615/1.3226/2.1307x`。这些数字只证明runner可执行并预示
formal可能NO-GO，不参与最终结论。正式raw必须从clean implementation commit的worker 0重新生成。

当前 targeted/typing/lint 后冻结source；formal raw、tamper、full regression与最终 verdict 尚待后续提交。
