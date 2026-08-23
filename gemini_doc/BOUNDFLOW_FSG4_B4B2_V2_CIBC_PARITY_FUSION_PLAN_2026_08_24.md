---
status: preregistered-implementation-open
updated: 2026-08-24T01:20:00+08:00
type: plan
topic: boundflow
slug: fsg4-b4b2-v2-cibc-parity-fusion
stage: s01
---

# FSG4/B4-B2 v2 CIBC-parity Fusion Plan

## 1. Motivation and Boundary

B2-5已以`VALIDATED-NO-GO-B4-B2-V1-PHYSICS`关闭：v1虽将peak allocated降至baseline的
`0.474638`，但真实执行为3 forward + 3 backward CUDA kernels，shared/vector/half优化均为0，
wrapper-inclusive geomean仅`0.424842x`。

v2不修改B4-B0/B1的production capture、typed IR、α/β ownership或PyTorch oracle。它只替换P-anchor
lowering/schedule/runtime，复现CIBC的两个核心物理机制：

1. horizontal multi-output fusion；
2. target-hardware autotuning。

本阶段不宣称用户报告的40x、whole-core/query、B0 parity或ASPLOS-ready。Triton首先作为CUDA融合和
schedule oracle；只有等价TVM TIR port单独通过时，才允许写TIR performance claim。

## 2. Fused ABI

P-anchor输入保持：incoming A、lower/upper、compressed alpha、incoming bias、weight/operator bias、
output adjoints及86项静态坐标。

forward只允许一个真实CUDA kernel：

- 主grid计算`output_lower_a[6,1,16,8,8]`；
- 同一kernel的尾部programs计算`output_bias[6,1]`；
- slope/intercept、compressed-alpha gather、ConvTranspose contraction和bias reduction全部融合；
- 输出可使用单一contiguous combined buffer的typed views，不得物化`relu_lower_a`或
  `output_bias_delta`全局workspace。

backward只允许一个真实CUDA kernel：

- 主grid直接生成incoming-A gradient；
- 尾部programs直接生成compressed-alpha gradient；
- adjoint Conv在register/local reduction中重算并共享表达式，不得物化`adjoint_conv`；
- P-anchor beta继续absent，不得伪造zero gradient。

## 3. Frozen Autotune Space

首轮Triton oracle只允许以下12项配置，写timing raw后不得追加：

| ordinal | BLOCK_M | BLOCK_K | num_warps |
|---:|---:|---:|---:|
| 0 | 32 | 16 | 4 |
| 1 | 32 | 32 | 4 |
| 2 | 32 | 64 | 4 |
| 3 | 64 | 16 | 4 |
| 4 | 64 | 32 | 4 |
| 5 | 64 | 64 | 4 |
| 6 | 64 | 32 | 8 |
| 7 | 64 | 64 | 8 |
| 8 | 128 | 16 | 4 |
| 9 | 128 | 32 | 4 |
| 10 | 128 | 64 | 4 |
| 11 | 128 | 64 | 8 |

允许Triton warm autotune选择winner；compile/tune时间排除，winner config和generated IR/ASM hash必须冻结。

## 4. Gates

### V2-0 correctness and structure

- 同一5份P raw，四路output/gradient均`atol/rtol=2e-4`、sign exact；
- 真实kernel inventory exact=`1 forward + 1 backward`；
- global intermediate workspace bytes=`0`；
- fallback/eager=`0/0`；higher-order、shape/dtype/device/alias/stream均fail closed。

### V2-1 physical oracle

沿用B2-5 public-PyTorch wrapper baseline与6-worker AB/BA协议：

- minimum GO：geomean `>=1.20x`、bootstrap lower `>1.00x`、worst `>=0.98x`；
- research target：geomean `>=2.00x`；
- stretch evidence：任一qualified production shape `>=5.00x`；
- allocated/reserved ratio均`<=1.05`；
- module/kernel count、thermal/power、semantic与autotune receipt全部通过。

未过minimum GO则`VALIDATED-NO-GO-B4-B2-V2-TRITON-PHYSICS`，不做TIR port。

### V2-2 TVM TIR port

Triton oracle通过后，冻结winner block/reduction组织并等价下沉为manual TIR。TIR必须独立重复V2-0/V2-1，
不得借Triton执行结果冒充TIR。TIR geomean至少达到Triton winner的`0.90x`且自身对PyTorch`>=1.20x`，
才允许进入B4-B3 exact-call integration。

## 5. Execution Order

1. 实现Triton fused forward/backward及custom autograd；
2. 5-raw correctness、kernel/ASM/workspace receipt；
3. 冻结12项autotune结果与6-worker formal timing；
4. 若minimum GO，实施manual TIR port并重复formal gate；
5. 只有TIR port通过才开放B4-B3；否则保留Triton mechanism/physics证据并关闭TIR integration。

本轮按用户指令连续执行，不在各子阶段暂停等待外审，但每步仍生成DocOps change/validation与可重放产物。
