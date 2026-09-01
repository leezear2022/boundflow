---
status: formal-executed-invalid
updated: 2026-08-26T00:25:00+08:00
type: changelog
topic: boundflow
slug: mr7-launch-materialization-implementation
stage: s01
---

# MR7 Launch / Materialization 归因实现修改记录

## 实现

- 新增互斥嵌套 host ledger，以 outer host clock 闭合 admission/handoff、layout/materialization、
  FFI/DLPack/stream、post-output guard 和 optimizer/residual；
- 新增 C0/C1/C2 × forward/backward × ordinal 显式 profiler marker；
- CUDA kernel 仅允许通过 CPU parent/correlation 绑定 marker，不以 temporal fallback 形成 headline；
- 单列 CUDA kernel sum 与 record-function device envelope，总量闭合误差门禁为 `<=2%`；
- control/profile 使用同一 MR6 diagnostic guard policy，保留 30/27 launch、cache/module/stream 和
  semantic receipt；
- profiler 使用 `acc_events=True`，避免外部 solver profiler cycle 清除已有 CUPTI event；
- 新增 host ledger、marker attribution、unattributed rejection 和 Amdahl fail-closed CPU 测试。

## 观测纪律

- host critical path 与 CUDA device time 分账，不相加；
- control 提供 host category share，profile只提供 device kernel attribution；
- profile/control CUDA event ratio仍需`<=1.10`；
- 本实现不改 TIR、schedule、allocator、solver 或 production default，且`performance_claimed=false`。

## 待完成

- 已完成三组 fresh counterbalanced formal artifact与独立replay；
- 已完成11/11 fully re-signed tamper；
- 定向`7 passed`，全量`1815 passed, 3 skipped, 6 warnings`；
- profile/control perturbation有一组`1.239399 >1.10`，MR7按冻结阈值以INVALID关闭；
- 后继仅开放MR7-R unprofiled host recovery，未开放MR7-A/B/C实现。
