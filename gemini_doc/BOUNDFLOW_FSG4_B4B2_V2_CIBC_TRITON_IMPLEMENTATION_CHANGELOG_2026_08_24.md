---
status: implemented-formal-run-pending
updated: 2026-08-24T01:45:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4b2-v2-cibc-triton-implementation
stage: s01
---

# FSG4/B4-B2 v2 CIBC-parity Triton Implementation Changelog

## Scope

本轮实现预注册v2的Triton horizontal-fusion oracle，不修改B4-B0/B1 production capture、typed IR、
compressed alpha ownership或public-PyTorch oracle，也不进入manual TVM TIR port/B4-B3。

## Implementation

- 新增`fsg4_b4b2_cibc_triton.py`；
- forward把slope/intercept、compressed-alpha gather、ConvTranspose contraction、operator-bias
  reduction融合为一个真实CUDA kernel；
- backward把adjoint convolution、incoming-A VJP与compressed-alpha VJP融合为一个真实CUDA
  kernel；
- forward只分配一个combined output buffer，backward只分配一个combined gradient buffer；不物化
  `relu_lower_a`、`output_bias_delta`或`adjoint_conv`全局中间workspace；
- 保持P-anchor beta absent，不伪造zero beta tensor；
- 固定12项`BLOCK_M/BLOCK_K/num_warps`搜索空间；
- compile/admission移出timed hot path，hot path保留不发GPU kernel的dtype/device/contiguous/alias
  检查；public入口仍在实例构建时执行shape/range/nonfinite fail-closed；
- 新增fresh-process worker与formal artifact/replay生成器；每个compiled specialization冻结
  source/TTIR/TTGIR/LLVM IR/PTX/cubin hash、register/spill及profiler kernel inventory。

## Deterministic Validation

- frozen config inventory：12/12；
- one-raw all-config direct parity：12/12 allclose且sign exact；
- five-raw winner-shape direct parity：5/5，最大差不超过`2e-4`；
- profiler inventory：exact `1 forward + 1 backward`，global intermediate workspace=`0`；
- targeted：`5 passed`；
- mypy（三个实现/runner文件）：clean；
- pylint（实现、runner、artifact、test）：`10.00/10`；
- black与`git diff --check`通过。

开发阶段非正式probe观察到public-PyTorch约`0.54 ms`、candidate约`0.15–0.27 ms`。该数字仅用于
确认formal run值得执行，不形成performance claim；正式结论必须来自提交后clean source上的12个
fresh calibration workers、5个correctness workers和6个AB/BA timing workers。

## Next

提交clean source并生成formal artifact。若minimum GO全部通过，才开放manual TVM TIR等价port；
否则以`VALIDATED-NO-GO-B4-B2-V2-TRITON-PHYSICS`关闭，B4-B3继续关闭。
