---
status: preregistered-r3-2a
updated: 2026-08-25T05:12:28+08:00
type: changelog
topic: boundflow
slug: r3-2a-optimizer-trajectory-preregistration
stage: s01
---

# R3-2A Optimizer Trajectory 预注册修改记录

## 修改

- 冻结 P-anchor 10 evaluation/9 Adam mutation/scheduler/clamp/termination语义；
- 冻结每步动态plan/trace rebind，而不削弱R3-1b1/b2 content-hash admission；
- 冻结独立native/candidate five-fresh、逐步lower/dα/α/Adam-state比较和immutable copy-in门禁；
- 冻结完整10/9 wrapper的memory/ownership门禁，但继续禁止读取latency；
- 冻结formal replay和中间轨迹fully re-signed tamper集合；
- 记录用户授权：R3-1b2/b3外部审计延后合并，不作为本轮人工暂停点。

## Claim边界

本文只是执行前协议冻结，不证明R3-2A已实现或通过，也不形成性能、whole-core、query或ASPLOS claim。
