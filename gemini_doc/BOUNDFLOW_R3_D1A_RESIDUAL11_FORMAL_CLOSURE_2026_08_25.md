---
status: validated-r3-d1a-residual11-correctness
updated: 2026-08-25T16:15:00+08:00
type: closure
topic: boundflow
slug: r3-d1a-residual11-formal-closure
stage: s01
---

# R3-D1-A Residual11 Staged 正式关闭

## 判定

`VALIDATED-R3-D1A-RESIDUAL11-CORRECTNESS`成立。formal source=`8fc15be`，artifact=
`artifacts/r3-structured-owner/r3-d1a-residual11-staged-v1`，summary hash=
`f7e0d6a75229f594449eecec47ff405d757338978c2ca3f04dbc9fd41503cf1b`。

这只开放residual6 staged correctness；D1-B schedule timing、D1-C wrapper、R3-3和same-solver保持关闭。

## 证据

- 5 fresh独立process，合计61,500个output/bias比较元素；
- candidate-v1 output/bias max diff=`1.0430813e-7/3.5762787e-7`；
- candidate-CPU-float64-oracle output/bias max diff=`3.0207744e-8/4.4692896e-7`；
- v1-oracle output/bias max diff=`9.1501499e-8/8.0455683e-7`；
- 全局最大diff=`8.0455683e-7 ≤ 2e-4`，sign exact；
- unscheduled/scheduled/device-source hash 5/5一致：`bc554be6…` / `64cfd923…` /
  `796446ee…`；
- 每run恰2 launch、1 caller-owned scratch、13/13 DLPack pointer、fallback=0、无global workspace、无
  persistent dense A；
- shape/dtype/nonfinite/device/alias/default-stream负路径fail closed；
- 10/10 fully re-signed tamper拒绝；targeted=`7 passed`；
- `timing_recorded=false`、`performance_claimed=false`。

## 语义变化

v1在每个最终output系数内重复计算`conv10^T`中间值；v2把完全相同的中间值先写入evaluation-local
scratch，再由stage-2应用同一ReLU slope/intercept、`conv8^T`、skip与bias。scratch不进入
Plan/Instance/optimizer，不跨evaluation保存。

## 下一动作

按同一合同实现residual6 v2：`conv4^T -> scratch -> slope -> stride2 conv2^T + 1x1 shortcut`。完成5
fresh correctness前禁止任何D1 timing或性能措辞。
