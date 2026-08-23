---
status: validated-no-go-b4c1-materialization-frontier
updated: 2026-08-24T11:25:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4c1-provider-owned-lower-formal-closure
stage: s01
---

# FSG4/B4-C1 Provider-owned Lower Formal Closure

## Verdict

`VALIDATED-NO-GO-B4-C1-MATERIALIZATION-FRONTIER`。

provider ownership、custom backward和数值语义均成立，但6 fresh累计 core稳定回退约5.2%。单一
P-anchor不再允许继续调参，也不形成production speedup claim。B4-C2只允许把融合边界移动到真实
materialization frontier，并扩展到完整lower operator tree与14-call coverage。

## Frozen Evidence

- source=`01bb21586ddfcfb1f5e04c31fd2958090a6b7d5f`；
- artifact=`artifacts/fsg4-b4c1-provider-owned-lower/resnet2b-prop0-v1`；
- manifest hash=`bbdd585209252348701e052affd64fe230b0713cbcb8ef7285d6533e46b3627c`；
- summary hash=`b8d163e578910ed064ad9cb527f9fd8b7433b02839f7e1e103312ac79135b1d0`；
- tamper report hash=`5cb9936ab0a55e547033c30f964a4d9115e5ffdd44a9c1e3700a25a9cbd7d669`。

## Six-fresh Timing

- B3 median ms=`[77.5631,78.2369,81.7899,81.5514,82.0596,81.2914]`；
- B4-C1 median ms=`[81.8493,82.4154,85.7164,86.2545,86.7178,85.9276]`；
- paired speedup=`[0.94763,0.94930,0.95419,0.94547,0.94628,0.94605]`；
- geomean=`0.9481500115566288x`；bootstrap lower=`0.9461590425689989x`；
- worst worker=`0.9454748158005997x`；
- allocated/reserved ratio=`0.9946132774437887/1.0`。

因此 no-regression与research门禁均失败，`performance_claimed=false`保持。

## Semantics and Integrity

- 180组terminal lower与全部α/β allclose/sign exact；
- max abs diff=`7.152557373046875e-07`；
- 每worker receipt均为10 forward/9 backward、provider-owned=10、bridge=0、reuse=9、
  fallback/eager=0；
- root replay从raw重算通过；8/8 outer-resigned tamper rejected。

## Root Cause

局部`4.90x`来自将 native structured operator在observer边界强制`to_dense()`后再比较。production
B3在这里保留`SignSplit(Conv2d(...))`并把materialization延迟到更远的消费点，所以该局部基准的
native分母不是same-path production成本。B4-C1虽然消除了lower算术，却提前生成dense tensor并引入
provider/autograd边界，收益无法传播到core。

## Next Gate

B4-C2必须先冻结真实materialization sites、消费者和operator-tree lineage，然后在该边界一次性lower
并融合完整子树。不得继续用P-anchor局部microbenchmark挑schedule。完成optimizer 10 calls后再扩展
KFSB 3 calls与剩余call；14-call覆盖前B4-D和query claim保持关闭。
