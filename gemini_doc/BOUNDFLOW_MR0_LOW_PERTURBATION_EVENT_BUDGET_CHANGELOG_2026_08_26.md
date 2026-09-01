---
status: validated-no-go-closed
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

首次 clean-source formal 在 source=`3080af6` 得到预期 NO-GO，但第一类 tamper 只改一个非中位
样本，无法改变派生summary，因而被replay接受。该 artifact 不作为最终证据。修正攻击为整组20个
latency样本的全重签改写，不修改门槛、worker或统计协议；修正source后必须从worker 0重跑formal。

最终source=`651e432`已从worker 0重跑：17对geomean/bootstrap upper/worst=
`2.137191/2.153191/2.163574x`，三门禁均失败；12/12 tamper通过。最终状态=
`VALIDATED-NO-GO-MR0-EXPLICIT-EVENT-BUDGET`，MR1/same-solver/R2关闭。专项=`10 passed`，
全量=`1677 passed,3 skipped`。
