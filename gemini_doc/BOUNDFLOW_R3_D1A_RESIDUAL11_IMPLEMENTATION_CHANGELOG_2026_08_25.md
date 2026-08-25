---
status: implemented-awaiting-five-fresh-formal
updated: 2026-08-25T15:55:00+08:00
type: changelog
topic: boundflow
slug: r3-d1a-residual11-implementation
stage: s01
---

# R3-D1-A Residual11 Staged 实现修改记录

## 实现

- 新增独立v2两symbol TIR：stage-1执行`conv10^T`到caller-owned scratch，stage-2执行ReLU slope/
  intercept、`conv8^T`、skip与bias；
- 固定128线程仅作correctness qualification，未进入D1-B schedule sweep；
- 新增zero-copy current-stream runtime：2 launch、1 scratch、13个DLPack pointer、无global workspace、
  fallback=0、timing/performance=false；
- 不改v1 backend/runtime/artifact，不把中间dense coefficient写入plan/state或跨evaluation保存；
- 新增生产状态三方对照：v1 raw TIR、v2 staged TIR、独立PyTorch闭合公式；
- 新增shape/dtype/nonfinite/device/arena-alias/default-stream负路径；
- 新增5 fresh worker/artifact/replay，raw保存输入，replay以CPU float64重新计算oracle。

## 当前验证

- targeted=`4 passed`；
- 单worker smoke最大candidate-v1 output/bias差约`1.04e-7/3.58e-7`，candidate-f64-oracle
  output/bias差约`3.02e-8/4.47e-7`；
- black、py_compile、mypy clean；pylint修正后待最终重跑；
- 没有读取latency，`performance_claimed=false`。

## 下一步

提交干净source后运行5 fresh correctness artifact与fully re-signed tamper。通过前residual6、D1-B timing和
D1-C wrapper保持关闭。
