---
status: validated-no-go-b4-c0-native-value-bridge
updated: 2026-08-24T06:15:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4c0-cumulative-core-formal-closure
stage: s01
---

# FSG4/B4-C0 Cumulative Core Formal Closure

## Verdict

`VALIDATED-NO-GO-B4-C0-NATIVE-VALUE-BRIDGE`。

含native-value bridge的exact-call候选语义可靠，但稳定回退约6%，未通过no-regression门禁。因此该
bridge只能作为correctness scaffold，不得作为累计性能候选；B4-C1 provider-owned lower path
rewrite现已开放。

## Frozen Identity

- source=`d1db31e693def0aa2b3eda26352eb40f951f64cc`；
- artifact=`artifacts/fsg4-b4c0-cumulative-core/resnet2b-prop0-v1`；
- manifest hash=`ea54377b7b792b50f7413c31c45dc18884ad11da3d3d82bdcff016895f16258a`；
- summary hash=`7a0b0505cfdd26f02e41cdaa6ebc8b69ea85f74499e41f5ac01a42925e2db9f9`；
- tamper report hash=`c24eddc83084d687c83160bc5d72179f86014d118cb28b2ac2fca128a739e2a9`。

## Timing

- B3 medians ms=
  `[81.802,77.698,79.235,78.230,77.898,82.094]`；
- candidate medians ms=
  `[86.860,82.486,83.959,83.104,82.949,87.878]`；
- worker speedups=
  `[0.94177,0.94195,0.94373,0.94135,0.93910,0.93418]`；
- geomean=`0.9403411451305688x`；
- bootstrap 95% lower=`0.9377792526994357x`；
- worst worker=`0.9341801911997806x`；
- maximum allocated/reserved ratio=`1.0481830750500403/1.0`。

no-regression与research gate均失败。`performance_claimed=false`保持。

## Semantics and Integrity

- 180 paired groups全部terminal lower/α/β allclose/sign exact；
- maximum absolute difference=`7.152557373046875e-07`；
- root replay从raw重算全部median/geomean/bootstrap/memory；
- 8/8 outer-resigned tamper rejected。

## Diagnosis and Next

局部manual TIR对public-PyTorch为`4.90x`，而累计bridge为`0.94x`；差值来自同一区域先执行native
lower Conv取得bitwise值，再执行TIR取得gradient，属于确定的重复计算而非算子性能不足。

B4-C1必须让observer在Conv backprop前取得provider ownership并跳过native lower branch。为避免
TIR约`1e-7` reduction-order差被Adam放大，先实现lower-only provider接口与逐evaluation数值轨迹
测量；若直接TIR值仍导致10-step漂移，则引入预注册的optimizer-stability策略，而不是恢复双算。
