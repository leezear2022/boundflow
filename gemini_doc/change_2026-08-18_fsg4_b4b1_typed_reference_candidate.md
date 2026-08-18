---
status: implemented-b4b1-typed-reference-pending-five-fresh
updated: 2026-08-18T15:14:00+08:00
type: change
topic: boundflow
stage: s01
---

# FSG4/B4-B1 typed IR 与 PyTorch reference 候选

## 改动

- 新增typed `DifferentiableLowerRegionIRV1`与instance，分离静态语义和raw tensor digest；
- 冻结α sparse reconstruction与β `-value*split-sign`语义、lower-only ReLU、Linear/Conv affine
  contraction、bias、stream、alias与fanout合同；
- 新增只依赖公开PyTorch算子的独立reference及hash-bound parity receipt；
- 新增双锚点前向/VJP、identity/policy/tensor/attribute/input-hash拒绝、动态bias/adjoint数值拒绝、
  S incoming-A micro-gradient和P empty-beta ownership测试。

## 验证

- related tests：`28 passed`；
- formal run 0双锚点全部metric allclose且sign exact，最大误差=`1.9073486328125e-06`；
- Black `--fast --check`通过；
- scoped Mypy `--follow-imports=skip`通过；
- scoped Pylint=`10.00/10`。

## 边界与下一步

当前只证明实现候选在一个formal run上成立，不构成B4-B1关闭。下一步生成5 fresh × 2 anchors
reference artifact、root semantic replay与协调篡改负例。B4-B2/TIR、性能、显存与ASPLOS-ready
继续关闭。
