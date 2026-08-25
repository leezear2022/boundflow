---
status: implemented-pending-formal
updated: 2026-08-26T22:35:00+08:00
type: changelog
topic: boundflow
slug: mr6-guard-attribution-worker
stage: s01
---

# MR6 Guard Attribution Worker 修改记录

- 新增独立provider/full/diagnostic worker，不修改MR5 production default或已冻结TIR；
- full模式对既有value guards只计数不改语义；diagnostic只跳过270次输入finite/range与30次handoff
  content同步，保留60次output-finite检查；
- 三种模式均复用MR5完整outer measurement、semantic state、30/27、cache/module/stream receipts；
- diagnostic envelope强制`production_admitted=false`、`performance_claimed=false`；
- CPU tests覆盖structural validator的shape/layout/gradient拒绝和`360→60`账本。

## Informal GPU preflight

在clean full regression结束后顺序运行`provider/full/diagnostic`各一个fresh process：

- provider/full/diagnostic host=`107.843/121.729/120.760 ms`；
- full/diagnostic=`1.008021x`，远低于正式路由门槛`1.10x`；
- provider/diagnostic=`0.893036x`，低于`0.98x`；
- CUDA event方向一致，guard receipt=`0/360/60`。

这是一组preflight，不是formal claim；但它否定了“300个被移除同步guard足以恢复parity”的乐观假设。
按预注册仍需clean source后运行3 triplet/9 fresh；若正式结果一致，MR6-B安全guard fusion关闭，下一步
只能做57次独立launch、DLPack、permute/contiguous与临时buffer的互斥归因。
