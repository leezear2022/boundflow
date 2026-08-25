---
status: implemented-correctness-smoke-passed-timing-pending
updated: 2026-08-25T19:04:00+08:00
type: changelog
topic: boundflow
slug: r3-d1c-cumulative-implementation
stage: s01
---

# R3-D1-C Cumulative Wrapper 实现修改记录

## 实现边界

- 新增独立 D1-C module，固定复用 D1-B winner：256 threads、serial reduction、vector width 1；
- 不修改冻结的 R3-2B、D1-A 或 D1-B 源码/artifact；
- 在原 10/9 candidate forward 内只拦截 residual11/residual6 两个 v1 symbol；其他 forward、custom
  backward、Adam、scheduler、α state 和 termination 完全复用原实现；
- residual11 scratch 使用第二个既有 arena 的 `[6144:12288]`；residual6 scratch 使用第一个既有
  arena 的 `[12288:18432]`；无新 persistent tensor 或 global workspace；
- stage-2 明确移除 `tir.noalias`，仅允许 6-element bias accumulator 作为 input/output 原位别名，
  从而避免额外 copy kernel；
- 每 evaluation 由原 15 个 forward launch 变为 17 个，其中 D1-C 恰为 4 launch；scratch region
  恰为 2，bias alias 恰为 2。

## 当前验证

- 原 R3-2B candidate 与 D1-C candidate 的完整 10 evaluation/9 Adam/9 scheduler terminal lower、
  sign、terminal α 在冻结 production state 上通过 tolerance；
- execution counter 为 10/9/9，custom forward/backward 为 10/10；
- ownership/claim drift 负路径通过；
- targeted `2 passed in 8.88s`；
- mypy clean，pylint `10.00/10`。

## Claim 边界和下一步

当前只证明 cumulative 接线的单次轨迹语义与 ownership；没有计时、没有性能 claim。下一步先运行
三方 smoke（native / frozen R3-2B / D1-C）确认物理传播，再冻结 5 fresh NC/CN formal protocol。
