---
status: validated-b4-b3-cibc-exact-call
updated: 2026-08-24T04:50:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4b3-cibc-five-fresh-formal-closure
stage: s01
---

# FSG4/B4-B3 CIBC Five-fresh Formal Closure

## Verdict

`VALIDATED-B4-B3-CIBC-EXACT-CALL`。

5个独立CUDA进程均完成B3/B4-B3 10-evaluation/9-update semantic pair，terminal lower与全部
α/β state allclose/sign exact。B4-B3机制关闭，只开放累计core timing；native-value bridge移除及
whole-query仍未开放。

## Frozen Identity

- source=`1d06aab80614c2da3822ab949570d2b324dc12d0`；
- artifact=`artifacts/fsg4-b4b3-cibc-five-fresh/resnet2b-prop0-v1`；
- manifest hash=`ff93b594fb0eb674ae1c62f71c56c98a8fbd623239b880b807605d67c2ba51e7`；
- summary hash=`74145ab9126eda7543ac6f58952b111a97958f4a6e4f79353fb94cc03da38ce6`；
- dense TIR module hash=`29e108a33c468bc9d3ee8b287f40374d7ccdaf78bf08caf6e23990abe3660d20`；
- tamper report hash=`a12c2593b159e928c024b54d37ef09e5602cc95955565bac42dacd69e0522b3e`。

## Semantic and Structural Results

- 5/5 fresh pairs、13 metrics/worker；
- maximum terminal/state absolute difference=`3.5762786865234375e-07`；
- evaluation-0 local output-A/bias max diff=
  `2.2351741790771484e-08/1.1920928955078125e-07`；
- allclose/sign exact=`true/true`；
- provider activation=`50`，TIR forward/backward=`50/45`；
- S-anchor unsupported=`5`并保持B3；
- fallback/eager/adjoint-materialization=`0/0/0`；
- native-value bridge=`50`，由receipt强制，不隐藏；
- root semantic replay通过；8/8 outer-resigned tamper rejected。

## Timing Boundary

raw中的单次wall ratios为
`[5.80,0.090,5.90,0.089,5.95]`，与BC/CB首调用顺序完全相关，直接证明其受cuDNN first-call
warmup支配；这些值冻结为`timing_diagnostic_only=true`，不得用于任何speedup结论。

## Next

实现累计core formal timing：每个fresh worker必须先分别预热B3与B4-B3，再用多组交错BC/CB执行；
对照仍为B3。先测量含native-value bridge的真实累计成本并给出NO-GO/GO；若bridge导致回退，则进入
CROWN lower-path provider ownership改写，在不预先执行native lower Conv的前提下解决Adam数值轨迹。
