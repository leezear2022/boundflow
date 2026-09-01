---
status: prepared-r3-d0-and-combined-audit
updated: 2026-08-25T06:20:00+08:00
type: changelog
topic: boundflow
slug: r3-d0-and-combined-audit-preparation
stage: s01
---

# R3-D0 与合并外审准备修改记录

- 在R3-2B NO-GO后冻结只读microphysics归因，而非直接选择CUDA Graph/TIR调优；
- 冻结host/CUPTI/NVTX时钟校准、symbol-family ledger和5-pair profile/sanity协议；
- 冻结dispatch与kernel两支Amdahl可达性公式及`required<=10x`准入；
- 冻结整体NO-GO条件和fully re-signed tamper集合；
- 新增覆盖R3-1b2/b3、R3-2A、R3-2B的一次性外审交接；
- 无代码或性能claim变更。

