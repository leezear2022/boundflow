---
status: implemented-awaiting-formal
updated: 2026-08-25T07:35:00+08:00
type: changelog
topic: boundflow
slug: r3-d0-profiler-implementation
stage: s01
---

# R3-D0 Profiler 与 Replay 实现修改记录

## 实现

- 新增diagnostic-only worker，复用冻结R3-2B对象和10/9 schedule；30次unprofiled sanity后仅profile
  一个完整wrapper；
- candidate通过实例级临时wrapper标注forward/backward/optimizer和compiled symbol，退出后恢复原method，
  未修改冻结runtime；
- 从PyTorch CUPTI profiler raw重建marker、correlation-parent/containment归属、kernel union/sum、phase、
  symbol family、host residual和校准receipt；
- 新增整段compiled recurrence闭合区域账本：R3-2A语义所有权、无persistent dense A、2 scratch；
- 新增5-pair formal artifact生成/replay，绑定R3-2B raw sanity、数值语义、事件、ledger、Amdahl route
  和manifest digest；
- 新增synthetic event/union/calibration/tamper单测及formal artifact replay入口。

## Smoke（非formal）

- native：median约`97–100 ms`，kernel union约`16.0 ms`，联合`id + linked_correlation_id`
  恢复后fallback `400/8516`、unattributed `0`、校准通过；
- candidate：median约`735 ms`，kernel union约`721 ms`，host residual约`14 ms`，fallback
  `7/606`、unattributed `0`、校准通过；
- candidate最重symbol是`boundflow_r31b1_residual6`（单次约`23.9 ms`）和
  `boundflow_r31b1_residual11`（单次约`10.5 ms`）。

这些只用于验证实现和决定能否启动formal，不形成性能claim。正式数字必须由5 fresh artifact replay给出。

## 边界

- 没有修改R3-2B runtime、TIR kernel或schedule；
- 没有实施CUDA Graph、fusion或调优；
- `performance_claimed=false`，R3-3仍关闭。

## 验证

- `6 passed`：`tests/test_r3_d0_microphysics_attribution.py`
- native/candidate各1个真实GPU smoke；
- `black`、`py_compile`、`git diff --check`。
