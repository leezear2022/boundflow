---
status: implemented-awaiting-five-fresh-formal
updated: 2026-08-25T16:40:00+08:00
type: changelog
topic: boundflow
slug: r3-d1a-residual6-implementation
stage: s01
---

# R3-D1-A Residual6 Staged 实现修改记录

## 实现

- 新增v2 stage-1：`incoming --conv4^T--> scratch[6,1024]`；
- 新增v2 stage-2：同一ReLU19 slope/intercept后执行stride-2 `conv2^T`，并融合1×1 shortcut
  `conv5^T`与bias；
- output为`[6,8,16,16]`，严格复现v1偶数坐标shortcut与output-padding语义；
- 新增zero-copy non-default-stream runtime：2 launch、1 caller-owned scratch、15/15 DLPack pointer、
  fallback=0、无persistent dense A、无计时；
- 生产状态比较v1 raw TIR、v2 staged TIR与独立PyTorch oracle；
- shape/dtype/nonfinite/device/alias/default-stream全部fail closed。

## 当前验证

- targeted生产correctness初始`2 passed`，补负路径后待最终重跑；
- 没有修改v1/residual11 formal artifact；
- `timing_recorded=false`、`performance_claimed=false`。

## 下一步

提交clean source，生成5 fresh raw，以CPU float64 replay重算oracle并执行fully re-signed tamper。通过前D1-B
timing、D1-C wrapper与R3-3关闭。
