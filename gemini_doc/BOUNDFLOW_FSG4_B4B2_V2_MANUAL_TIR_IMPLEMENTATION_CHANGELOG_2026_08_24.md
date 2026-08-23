---
status: implemented-formal-run-pending
updated: 2026-08-24T02:15:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4b2-v2-manual-tir-implementation
stage: s01
---

# FSG4/B4-B2 v2 Manual TVM TIR Implementation Changelog

## Scope

按Triton formal GO开放的V2-2，把P-anchor CIBC horizontal fusion等价下沉为manual TVM TIR。
不修改production capture、typed IR、compressed-state ownership或public-PyTorch/Triton oracle；
B4-B3仍未接通。

## Compiler and Runtime

- manual TIR module仅导出forward/backward两个PrimFunc；
- forward以128 threads/block、32 outputs/block、4-thread cooperative reduction执行bound selection、
  compressed-alpha gather、Conv contraction和bias reduction；
- backward在单kernel中直接生成incoming-A与compressed-alpha gradient；
- 两个方向均使用单一combined output/gradient buffer，不物化v1的三个全局workspace；
- generated CUDA source与profiler均验证exact `1 forward + 1 backward`；
- compiled PackedFunc、DLPack views、mapping tensors和combined buffers由PlanInstance持有并复用；
- 默认stream在instance admission时与TVM FFI stream核对；hot call只在forward核对一次stream，
  backward复用同一已准入call context；non-default/mismatch fail closed；
- 关键运行时修复：禁止每次launch按symbol字符串重新查询Executable PackedFunc。将两个PackedFunc
  常驻PlanInstance后，消除了约百微秒级Python/FFI间隙。

## Deterministic Validation

- 5 raw public-PyTorch parity通过，sign exact，开发probe最大差=`5.96e-08`；
- generated-source symbol inventory与profiler launch inventory均exact `1+1`；
- global intermediate workspace bytes=`0`；
- targeted manual-TIR tests=`5 passed`，Triton+TIR合计=`10 passed`；
- black、mypy clean、pylint `10.00/10`。

开发probe三方median约为PyTorch/Triton/TIR=`0.500/0.153/0.093 ms`：TIR对PyTorch约
`5.38x`，Triton/TIR时间比约`1.64x`。这些数字只决定formal run值得执行，不形成performance claim。

## Next

提交clean source，运行5 correctness workers与六全排列顺序的6个三方timing workers。只有正式
baseline/TIR geomean≥`1.20x`、Triton/TIR geomean≥`0.90x`、correctness/kernel/workspace/memory
门禁全部通过，才开放B4-B3 exact-call integration。
