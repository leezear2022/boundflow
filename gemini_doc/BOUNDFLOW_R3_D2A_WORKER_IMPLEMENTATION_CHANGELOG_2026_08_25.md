---
status: implemented-smoke-passed-formal-pending
updated: 2026-08-25T20:41:00+08:00
type: changelog
topic: boundflow
slug: r3-d2a-worker-implementation
stage: s01
---

# R3-D2-A Attribution Worker 修改记录

## 实现

- 同一 fresh process 内运行两个独立 10/9 wrapper；
- phase-only wrapper 只插入 forward/backward/sign/effective/recompute 五层 CUDA event，用于 headline
  share 和 Amdahl；
- symbol-only wrapper 只在 B1/B2 launch 周围插 event，用于 dominant symbol ledger，明确
  `symbol_profile_headline_forbidden=true`；
- 两个 wrapper 都在 3 warmup 后运行，分别重置同一冻结 α 初值；
- terminal lower/α/sign 与对应 D1-C formal run 比较；
- terminal compressed-gradient 采用 backward parent 减三个互斥 child 的守恒 residual，不伪造直接 marker。

## 被拒绝的首版测量

首版把 phase 和每 symbol event 放进同一 wrapper，host=`490.601 ms`，相对 D1-C formal 约高
`24.6%`，因此不得形成 share。拆分后 phase-only host=`393.013 ms`，相对对应 formal
`393.705 ms` 仅差 `0.18%`。

## Pre-formal smoke

- backward=`369.380 ms`；coefficient-sign=`342.899 ms`；effective=`25.856 ms`；
  recompute=`0.536 ms`；terminal residual=`0.090 ms`；
- symbol-only host=`415.550 ms`，只作 ledger；
- raw residual6=`257.383 ms`，raw residual11=`105.018 ms`，effective-pre23=`25.725 ms`；
- phase/symbol 两次 terminal 均通过；mypy clean、pylint `10.00/10`。

## 门禁修正

formal 尚未生成前，基于 D1-B 已验证 worst `56.8625x` 的同构 residual signature，把该精确映射路径
的 required-speedup admission 上限从通用 `10x` 修正为 `15.50x`。未知区域仍保持 `10x`。

## 首轮 formal replay 加固

首轮 raw 生成后，tamper 探针发现：若同时修改 terminal tensor 与其 tensor hash，replay 只检查了 worker
已存的 diff 字段，没有从冻结 D1-C raw 重新计算，导致该 tamper 被接受。replay 已改为按 run index 加载
D1-C formal terminal，独立重算 lower/α diff 与 sign；加固后必须重新提交源码并重跑全部 five-fresh，
首轮 raw 不进入正式 artifact。

第二轮 raw 暴露另一项 scope 问题：两个 profile worker 比对应 formal D1-C 慢约`13%–14%`，直接用
`phase_cuda/formal_host` 得到最多`1.0126`的非物理 share。summary 已改为先在同一 worker 计算
`phase_cuda/profile_host`，再乘 `formal_host/profile_host` calibration scale 投影 formal duration；
required-speedup 只使用投影后的同 scope duration。第二轮 raw 同样不进入正式 artifact，必须再次重跑。
