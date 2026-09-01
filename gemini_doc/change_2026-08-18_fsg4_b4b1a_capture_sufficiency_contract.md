# FSG4/B4-B1a Capture Sufficiency Contract 变更记录

日期：2026-08-18
状态：`IMPLEMENTED-B4-B1A-CAPTURE-CONTRACT-PENDING-FIVE-FRESH`

## 代码变更

- 新增 `B4B1RegionLiveObserverV1`，在显式 opt-in 路径捕获 incoming lower bias、真实 operator
  bias，以及 region `output_lower_a` / `output_bias` 的 production adjoints；
- B4-B1 observer 将 materialized lower-A 重新接回 diagnostic autograd graph，以保证捕获的是实际
  被上游继续消费的 tensor；B4-B0 observer 返回 `None`，旧路径不替换；
- 新增 `ProductionDifferentiableReferenceCaptureV1`，在 B4-B0 base capture 上追加 bias、output
  adjoints、全部 sparse α/β mapping raw、presence bitmap、logical shape 与 Conv output padding；
- 新 payload 从 raw 重建 base/amendment，并逐个 mapping tensor digest 对齐 B4-B0 lineage；
- `crown_ibp` 与 terminal schedule 只把 observer 类型收窄为 duck-typed protocol，默认无 observer
  与 B4-B0 v1 observer 行为保持。

## 验证

- CPU production schedule：两个锚点 bias/output adjoints 捕获通过；
- real CUDA：旧 B4-B0 dual capture 通过；新 B4-B1 amendment payload 两锚点生成、raw replay通过；
- related=`26 passed`；
- full=`1378 passed, 3 skipped, 6 warnings`；
- Black clean、scoped Mypy clean、Pylint=`10.00/10`、`git diff --check`通过。

## 边界与下一步

本轮只实现 capture contract/mechanism，尚未生成 5-fresh formal artifact，尚未实现 typed IR 或
pure-PyTorch reference。因此没有 B4-B1 correctness/gradient claim，更没有 B4-B2/TIR/performance
claim。下一步是提交冻结代码后实现独立 worker/runner，生成 B4-B1a five-fresh raw、root replay
与 bias/adjoint/layout 完整性负例。
