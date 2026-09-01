---
status: validated-b4-b2-v2-manual-tir
updated: 2026-08-24T02:35:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4b2-v2-manual-tir-formal-closure
stage: s01
---

# FSG4/B4-B2 v2 Manual TVM TIR Formal Closure

## Verdict

`VALIDATED-B4-B2-V2-MANUAL-TIR`。

手写TVM TIR已把CIBC式P-anchor横向融合压成真实CUDA `1 forward + 1 backward`，并在相同
production capture、public-PyTorch oracle及已冻结Triton winner下通过三方正式门禁。因此
B4-B3 exact-call integration现已开放；whole-core/query claim仍关闭。

## Frozen Identity

- source=`5b2c9ba53e20a95ccd85ddae13e0d481f226c902`；
- artifact=`artifacts/fsg4-b4b2-v2-cibc-tir-formal/resnet2b-prop0-v1`；
- manifest hash=`b309f6e85d9eede4c852f637ad0628055c8abae25bb7fd96d2b11485bfa536c4`；
- summary hash=`dd58b77b0cb9bce3724ea251568f7c6b12be427ea5ccd8b3ef0be2dac95c9e6b`；
- TIR module hash=`681def0238eae5861af69673488afacc05415717c394ad9a39c9f21c2369d072`；
- CUDA device-source hash=`ee24f60b8efe385ea7de52974c6be0612792003b766fe9fbd4f10c8139971d21`；
- exported symbols恰为`boundflow_cibc_horizontal_forward_v2`与
  `boundflow_cibc_horizontal_backward_v2`。

## Correctness and Structure

- 5个fresh correctness workers，三方四路输出各`12,810`元素/worker；
- maximum absolute difference=`1.9073486328125e-06`，allclose/sign exact；
- profiler kernel inventory exact=`1 forward + 1 backward`；
- global intermediate workspace bytes=`0`；
- PlanInstance复用编译后的PackedFunc、DLPack views、mapping tensors及combined output/gradient
  buffers，编译与stream admission均不进入双launch hot path；
- root replay从raw独立重算median、三方ratio、geomean、bootstrap、memory ratio与receipt；
- 10/10 outer-resigned semantic tamper rejected，report hash=
  `1fa49a8a4b9e3287455f93777f0d7858581127ce58e2f11f7e4048e74851699c`。

## Three-way Timing and Memory

6个fresh workers，顺序为`BTR/BRT/TBR/TRB/RBT/RTB`，每方案10 warmup、30 groups：

- PyTorch/TIR speedups=
  `[5.2120187,4.7223099,4.6860109,5.1601616,4.9395157,4.6987477]`；
- PyTorch/TIR geomean=`4.898339978572916x`；
- bootstrap 95% lower=`4.737707934038285x`；
- worst worker=`4.686010915822886x`；
- Triton/TIR ratios=
  `[1.8200419,1.5688838,1.5950139,1.7860364,1.7705454,1.5763333]`；
- Triton/TIR geomean/lower/worst=
  `1.6827270318064584x/1.6069511060812625x/1.5688838046634832x`；
- maximum allocated/reserved ratio=`0.45088566827697263/1.0`。

全部预注册门禁通过：TIR不仅保留至少`0.90x` Triton性能，实际比Triton再快`1.68x`；相对
public-PyTorch oracle则快`4.90x`。`performance_claimed=false`继续保持，因为这是region-level内部
物理准入，不是whole-core、query或论文最终claim。

## Diagnosis

这一结果确认此前慢的不是“TIR天然慢”，而是v1的六kernel拆分、global intermediates以及每次launch
重复做PackedFunc查找/stream admission。v2同时完成算子横向融合、schedule重写和PlanInstance运行时
常驻，因而在同一P-anchor上超过Triton。这也解释了用户原型BoundConv能获得几十倍局部收益：融合边界
越宽、被消掉的Python/launch/workspace越多，局部倍数就越大；本轮正式比较包含forward+backward与
wrapper，可靠数值为`4.90x`。

## Next

实现B4-B3 exact-call：以现有live observer把P-anchor production tensors直接交给该TIR executor，
保留S-anchor走B3并明确记为unsupported；验证10/10 optimizer evaluations、9/9 mutation、terminal
parity、provider/fallback/eager计数和5 fresh B3/B4-B3语义对。此阶段通过后才允许累计core timing。
