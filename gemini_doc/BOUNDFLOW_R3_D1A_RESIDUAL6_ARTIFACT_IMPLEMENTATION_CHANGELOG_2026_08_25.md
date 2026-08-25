---
status: implemented-awaiting-formal-generation
updated: 2026-08-25T17:05:00+08:00
type: changelog
topic: boundflow
slug: r3-d1a-residual6-artifact-implementation
stage: s01
---

# R3-D1-A Residual6 Artifact 实现修改记录

## 修改范围

- 新增独立进程 worker，冻结 residual6 production inputs、v1 reference、v2 candidate、
  tensor hashes 与 runtime receipt；
- 新增 five-fresh artifact generator/replay，绑定 clean source、code blobs、上游 residual11
  artifact、模型与 source capture；
- replay 使用 CPU float64 闭合公式独立重算 stride-2 main path、1×1 shortcut、ReLU
  slope/intercept 与 bias，不调用仓库 v1 reference；
- 新增 10 类 fully re-signed tamper：input、candidate/reference output、schedule hash、launch、
  scratch、timing、tolerance、D1-B admission 与 performance claim；
- 新增 pytest replay/admission/tamper 三条 formal gate。

## 预生成验证

- 单进程 smoke：candidate-reference 最差 `9.5367431640625e-07`；
- candidate-float64-oracle 最差 `1.6777612468210634e-06`；
- reference-float64-oracle 最差 `1.916179825922626e-06`；
- 以上均小于冻结 `2e-4`，sign exact 由 replay 强制；
- targeted 在 artifact 生成前为 `3 passed, 3 skipped`；
- mypy clean，pylint `10.00/10`。

## Claim 边界

- 本提交只建立 correctness evidence machinery；
- `timing_recorded=false`、`performance_claimed=false`；
- D1-B 仅在 five-fresh、replay 与 10/10 tamper 完成后开放；D1-C/R3-3 继续关闭。
