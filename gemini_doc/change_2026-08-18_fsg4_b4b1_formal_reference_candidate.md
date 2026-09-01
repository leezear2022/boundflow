---
status: validated-b4b1-reference-pending-integrity-report-and-regression
updated: 2026-08-18T15:34:00+08:00
type: change
topic: boundflow
stage: s01
---

# FSG4/B4-B1 formal reference artifact 候选

## Formal artifact

`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v1`绑定source=`2bc9638`及B4-B1a
manifest=`67ace9e4...25f6`。root replay从5个原始PT重编译IR/instance并重跑pure-PyTorch
reference：

- 5 fresh / 10 captures；
- 60 metrics / 196,380 elements；
- maximum absolute difference=`1.9073486328125e-06`；
- allclose=true、sign exact=true；
- S native β gradient=5/5、P incoming-A gradient=5/5；
- S/P静态IR hash分别为`f5085dde...a08`与`f781e56c...f67`；
- summary hash=`9489c70f...4cc2`；
- `performance_claimed=false`、`tir_admitted=false`。

## 协调篡改候选

新增两类all-run rewrite：incoming bias与output adjoint。两案都同步重签内部capture digest、source
summary、source manifest及derived protocol，旧capture-sufficiency replay仍可通过，但新的数值reference
均以`numerical semantics differ`拒绝。probe正式report需在probe源码提交后生成。

## 边界与下一步

当前仍是pending external audit；下一步生成hash-bound integrity report、跑相关/全量/static/DocOps，
形成内部关闭与exchange。B4-B2/TIR/performance仍关闭。
