---
status: implemented-smoke-passed-formal-pending
updated: 2026-08-26T00:50:00+08:00
type: changelog
topic: boundflow
slug: r3-d2b-timing-worker
stage: s01
---

# R3-D2-B Timing Worker 修改记录

- 新增 native/D1-C/D2-B 三模式独立进程 worker；
- 每个 worker 固定 3 warmup + 30 个完整 10/9 host-wall sample；
- headline 后另跑一次 10-event coefficient-sign parent measurement，明确禁止进入 headline；
- 冻结 terminal、execution、D1-C/D2-B ownership、arena pointer 与 memory receipt；
- black、mypy、pylint `10.00/10` 通过。

单 triplet smoke：native/D1-C/D2-B median=`97.608/393.543/56.634 ms`，candidate/native=
`1.72349x`，D1-C recovery=`6.94886x`，raw/staged coefficient-sign=`55.7662x`；D2-B 与 D1-C peak
allocated/reserved相同。该结果只证明 worker 与量级可执行，`performance_claimed=false`，必须由 five-fresh
formal 重算。

