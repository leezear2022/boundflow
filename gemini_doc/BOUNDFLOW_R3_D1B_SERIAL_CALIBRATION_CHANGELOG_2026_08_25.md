---
status: calibration-winner-selected-formal-pending
updated: 2026-08-25T17:42:00+08:00
type: changelog
topic: boundflow
slug: r3-d1b-serial-calibration
stage: s01
---

# R3-D1-B Serial Schedule Calibration 修改记录

## 修改

- 新增隔离的 D1-B backend，不改动或重签 D1-A residual6/residual11 已冻结源码与 artifact；
- 编译 v1 raw residual6+11 baseline，以及 two-kernel staged 的 `64/128/256` threads 固定候选；
- reduction 保持 serial reference、vector width 固定 1、2 scratch、4 launch；
- calibration 使用同一 production tensors、同一 non-default stream、2 warmup、10 组交错 CUDA-event
  样本；所有候选先做冻结 reference correctness，再计时。

## Calibration 结果

| threads | v1 median(ms) | candidate median(ms) | isolated speedup | max diff |
|---:|---:|---:|---:|---:|
| 64 | 34.100225 | 0.600016 | 56.8322x | 9.53674e-7 |
| 128 | 34.086912 | 0.589824 | 57.7917x | 9.53674e-7 |
| 256 | 34.077696 | 0.581120 | 58.6414x | 9.53674e-7 |

固定 calibration winner 为 `256 threads / serial reduction / vector width 1 / two-kernel`。其单次
calibration 已超过 D1-B isolated opportunity `15.50x`，但它不是 formal claim。

## Claim 边界与下一步

- `calibration_only=true`、`formal_performance_claimed=false`；
- 不因单 capture calibration 开放 D1-C；
- 下一步仅用冻结 winner 生成 5 fresh formal correctness/timing artifact 与 fully re-signed tamper；
- shared-memory、warp-shuffle、vector 2/4 和 producer-consumer fusion 暂不进入 formal，因为 serial
  winner 已显示有资格先接受稳定性门禁。
