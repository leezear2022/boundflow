---
status: implemented-awaiting-five-fresh-formal
updated: 2026-08-25T18:05:00+08:00
type: changelog
topic: boundflow
slug: r3-d1b-formal-artifact-implementation
stage: s01
---

# R3-D1-B Formal Artifact 实现修改记录

## 修改范围

- 新增 fixed-winner fresh worker：只执行已由 calibration 选定的 `256 threads / serial reduction /
  vector width 1 / two-kernel`；
- 每个 worker 独立编译 v1 baseline 与 candidate，使用同一 stream 和同一 production tensors；
- 每个 worker 固定 2 warmup、10 组 baseline/candidate 交错 CUDA-event 样本；
- 新增 artifact generator/replay：从 raw 数组独立重算 median、pair speedup、5-run geomean/worst，
  并绑定 D1-A 两个 manifest、calibration、source/code blobs 与 compiler receipt；
- 新增 formal pytest gate，严格区分 isolated performance 与尚未运行的 wrapper performance。

## Smoke

- 单 worker：v1 `34.088959 ms`，candidate `0.596480 ms`，`57.1502x`；
- maximum diff `9.5367431640625e-07`、sign exact；
- mypy clean、pylint `10.00/10`。

## Claim 边界

- five-fresh 完成前不形成 formal claim；
- 通过后只允许 claim isolated residual6+11 固定 schedule qualification；
- `wrapper_performance_claimed=false`，D1-C 仍需独立 cumulative 10/9 wrapper protocol。

## 预生成 tamper 修正

首版 duration tamper 只改一个非中位数样本，统计上不会改变 median，因而被正确接受。该探针已改为
重写完整 10 样本数组；同时 replay 新增强制：五个 fresh compiler receipt 完全一致、environment receipt
完全一致、protocol 的 winner/sample/gate/claim 字段精确匹配。修正后 10/10 fully re-signed tamper 均拒绝。
