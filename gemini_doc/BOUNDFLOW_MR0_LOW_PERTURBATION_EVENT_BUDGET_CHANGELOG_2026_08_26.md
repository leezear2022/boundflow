---
status: preregistration-recorded
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

本记录仅冻结协议。实现、formal raw、replay、tamper、full regression与最终 verdict 尚待后续提交。
