---
status: validated-b4-b2-v2-triton-physics
updated: 2026-08-24T01:55:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4b2-v2-cibc-triton-formal-closure
stage: s01
---

# FSG4/B4-B2 v2 CIBC-parity Triton Formal Closure

## Verdict

`VALIDATED-B4-B2-V2-TRITON-PHYSICS`。

在同一P-anchor production capture、compressed-state ABI与public-PyTorch oracle下，CIBC式横向融合
通过预注册minimum GO，并达到`2x` research target。因此只开放manual TVM TIR等价port；B4-B3
exact-call integration仍关闭。

## Frozen Identity

- source=`77a15ebec0b0deb3798ad35553bde53c8295616b`；
- artifact=`artifacts/fsg4-b4b2-v2-cibc-formal/resnet2b-prop0-v1`；
- manifest hash=`8214e3f99400c55dcc31dcea60baba5b16e8b8f3602a3983d3be16b8cc5b9aed`；
- summary hash=`5784a8c738c7887047e435a99f5d1ac199f52d8e411e5b921293b7158f7006a0`；
- winner ordinal=`1`，`BLOCK_M/BLOCK_K/warps=32/32/4`；
- forward/backward PTX hash=`4438f5f4…e616`/`2df315a8…688f`；
- forward/backward cubin hash=`f781b970…b705`/`71746e0f…5e9a`；
- register=`114/80`，spill=`0/0`。

## Correctness and Structure

- 12个fresh calibration workers，winner只按raw median选择；
- 5个fresh correctness workers，四路共`12,810`元素/worker；
- maximum absolute difference=`1.9073486328125e-06`；
- allclose/sign exact=`true/true`；
- profiler kernel inventory exact=`1 forward + 1 backward`；
- global intermediate workspace bytes=`0`；不物化`relu_lower_a`、`output_bias_delta`或
  `adjoint_conv`；
- root replay从raw重算所有median/pair speedup/geomean/bootstrap/memory ratio/配置，并在独立进程
  重新编译winner；TTIR/TTGIR/LLVM IR/PTX/cubin逐项一致；
- 10/10 outer-resigned semantic tamper rejected。

## Timing and Memory

- 6 fresh workers，order=`AB/BA/AB/BA/AB/BA`；每侧10 warmup、30 measured pairs；
- worker speedups=
  `[2.8475445648,2.7687974429,2.9150637482,2.8948009132,2.8644301612,2.7399968435]`；
- geomean=`2.837719484336988x`；
- bootstrap 95% lower=`2.785754654624944x`；
- worst worker=`2.739996843548485x`；
- maximum allocated ratio=`0.3633369923161361`；
- maximum reserved ratio=`1.0`。

预注册minimum GO(`1.20x`/lower>1/worst≥0.98/memory≤1.05)全部通过，research target
`2.00x`也通过。`performance_claimed=false`仍保持：该结果是内部正式物理准入，不冒充外部审计或
whole-core/query claim。

## Diagnosis

v1失败、v2通过的因果差异已经被隔离：v1真实执行`3+3` kernels，并在hot path反复同步准入；v2把
准入移到instance construction，执行期只做零kernel metadata/alias检查，同时把bound selection、
Conv contraction、bias reduction和两路VJP压成`1+1` kernels，消除三个全局workspace。这个结果支持
用户关于BoundConv可出现数量级算子收益的判断，但当前P-anchor含forward+backward及wrapper，正式值是
`2.84x`，不是40x。

## Next

按预注册V2-2把winner的block/reduction组织等价下沉为manual TVM TIR。TIR必须重新通过5 raw、exact
`1+1` kernel、零workspace与6-worker formal timing；只有TIR达到Triton的`0.90x`且对PyTorch
`>=1.20x`，才开放B4-B3。
