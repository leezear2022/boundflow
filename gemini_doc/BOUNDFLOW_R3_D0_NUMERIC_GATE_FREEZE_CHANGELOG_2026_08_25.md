---
status: completed
updated: 2026-08-25T07:05:00+08:00
type: changelog
topic: boundflow
slug: r3-d0-numeric-gate-freeze
stage: s01
---

# R3-D0 数值门禁冻结修改记录

## 修改

- 在profiler实现与formal运行前冻结host marker、CUDA event/activity envelope的校准残差阈值；
- 冻结containment fallback上限为`5%`、unattributed kernel为`0`；
- 明确CUDA event elapsed、kernel activity union和host residual的时钟所有权，禁止跨域相减；
- 接纳外部评审的残留风险：same-solver query必须按op type重测share与可达`G`，不能把独立IBP图
  `2.45631x`直接代入query feasibility。

## 边界

本提交只冻结门禁，不实现profiler、不运行formal、不形成性能claim。R3-3、CUDA Graph和kernel
schedule/fusion仍关闭。

## 验证

- `git diff --check`
- `dol lint --soft`
