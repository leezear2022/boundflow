---
status: validated-no-go-v1-physics
updated: 2026-08-24T01:05:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-5-formal-microphysics
stage: s01
---

# FSG4/B4-B2 B2-5 Formal Microphysics Closure

## Verdict

`VALIDATED-NO-GO-B4-B2-V1-PHYSICS`。

当前B4-B2 v1以6个标量化CUDA kernels实现sparse-source Conv forward/backward，正式
wrapper-inclusive结果未通过`1.05x`局部门禁，因此B4-B3 exact-call integration保持关闭。

该结论只关闭当前12项冻结schedule与6-kernel v1 lowering，不否定CIBC式horizontal fusion、
shared-memory、多级tiling或auto-tuning v2。

## Frozen Identity

- source=`bf1c8b75230ebfa33706e1ce0436d1f8199cd5fe`；
- artifact=`artifacts/fsg4-b4b2-b2-5-formal-microphysics/resnet2b-prop0-v1`；
- manifest=`c2d3d30d…f518`；
- summary=`f873d471…b163`；
- winner ordinal=`11`；schedule=`2b2d6518…0e28`；module=`24995e01…8d03`；
- ledger=`1660edca…07c6`，没有第13项。

## Correctness and Replay

- S/P各5个独立进程，共10 correctness workers；
- maximum absolute difference=`2.384185791015625e-06`，allclose/sign exact；
- root replay重新派生summary并独立重编译winner，TIR/device-source/module/kernel inventory一致；
- 8/8 outer-resigned tamper rejected；首次7/8失败暴露的module cross-binding缺口已在新source修复，
  旧artifact仅保留于`/tmp/boundflow-b2-5-pre-module-binding-20260824`作为可恢复失败证据。

## Timing and Memory

- 6 fresh workers，order=`AB/BA/AB/BA/AB/BA`；
- 每worker每侧10 warmup、30 measured pairs；
- worker speedups=
  `[0.4149406421,0.3776925294,0.4341507837,0.4300251100,0.4423875933,0.4542608734]`；
- paired geomean=`0.42484238749783887x`；
- bootstrap 95% lower=`0.4031569161542472x`；
- worst worker=`0.3776925294408135x`；
- candidate/baseline peak allocated max ratio=`0.4746376811594203`；
- reserved max ratio=`1.0`。

结论：v1 active allocation约减少52.5%，但wrapper-inclusive时延约为baseline的2.35倍，物理门禁
明确失败。`performance_claimed=false`，不得把memory ratio升级为system memory claim。

## Kernel Diagnosis

- module calls=`1 forward + 1 backward`；
- real CUDA kernels=`3 forward + 3 backward`；
- shared-memory/vector/half token=`0/0/0`；
- materialized workspace含`adjoint_conv[6,1,16,8,8]`与`output_bias_delta[6,1]`。

这说明v1完成了语义lowering与compressed-state输入，但没有完成CIBC意义上的多输出横向融合和
硬件级调优。

## Next

下一工程动作是另行冻结并实现`B4-B2-v2 CIBC-parity`：以同一typed IR/oracle/capture为语义根，
目标是消除`adjoint_conv`全局物化、减少真实kernel数、加入shared/local cache、多级tiling与实际
硬件搜索。v2通过前不启动B4-C coverage或B4-D whole-query。
