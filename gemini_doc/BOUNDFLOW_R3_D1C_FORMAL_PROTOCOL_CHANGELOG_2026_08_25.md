---
status: implemented-awaiting-formal-generation
updated: 2026-08-25T19:36:00+08:00
type: changelog
topic: boundflow
slug: r3-d1c-formal-protocol
stage: s01
---

# R3-D1-C Formal Protocol 修改记录

## 协议

- 五个 fresh triplet、15 个独立 worker；native/D1-C 相对顺序严格为 `ND/DN/ND/DN/ND`，B3
  以平衡位置插入；
- 每 worker 3 warmup + 30 host-wall samples；worker 间固定 30 秒 cooldown；
- 三方共享 source capture、model、plan、trace、environment；
- native/B3/D1-C terminal lower、sign、α 与 10/9/9 counters 逐 triplet 比较；
- D1-C receipt 强制 17 forward launch，其中 D1-C 4 launch；2 arena/2 tail scratch、2 bias alias、
  persistent dense A/global workspace/fallback/eager/native shadow 全为零；
- 同时报告 native→D1-C wrapper speedup 与 B3→D1-C cumulative recovery，不跨 scope 代入。

## 冻结门禁

- wrapper geomean `≥1.20x`、worst `≥1.00x`；
- B3→D1-C cumulative worst `≥9.3181x`；
- D1-C allocated/reserved peak 不高于 B3；
- 全部通过才开放 R3-3；失败则关闭为
  `VALIDATED-NO-GO-R3-D1C-CUMULATIVE-WRAPPER`，只开放 backward attribution。

## 预生成检查

单 triplet smoke 可被 validator 独立重算：wrapper=`0.254035x`、B3 recovery=`1.867601x`、
lower/α/sign/memory 全通过。该数字仅验证协议实现，不是 formal 结果。
