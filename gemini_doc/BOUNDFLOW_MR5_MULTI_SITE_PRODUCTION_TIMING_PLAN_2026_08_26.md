---
status: preregistered-active
updated: 2026-08-26T20:40:00+08:00
type: plan
topic: boundflow
slug: mr5-multi-site-production-timing
stage: s01
---

# MR5 Multi-Site Production Bridge Timing 预注册

## 1. 前置与唯一问题

MR5 correctness formal与clean full=`1787 passed,3 skipped`均已通过，本协议已激活。本阶段只回答：

> 在冻结ResNet2B property-0的同一个αβ-CROWN beta-split optimized exact call中，同时替换
> C2→C1→C0三条lower ReLU+Conv region后，完整10-evaluation/9-mutation outer exact call是否比
> 原生provider更快，并且不增加absolute peak memory？

这是MR4发现`4.5P`静态机会后第一次测试收益能否穿过三site wrapper/runtime/materialization传播。
不改TIR数学，不做schedule调优，不把CIBC独立IBP graph的`2.45631x`代入本结果，不测complete query。

## 2. Baseline与Candidate

### Provider P

原αβ-CROWN exact call；计时包含10次三个native region、其余全部图、optimizer、clip和返回。

### Multi-Bridge M

同一exact call中只替换MR5冻结的C2/C1/C0。计时必须包含每evaluation三site admission、full α重建、
contiguous/materialization、DLPack、30/27 launch、custom backward和provider optimizer；不得运行native shadow、
formal trajectory、owner snapshot或failure injection。

三个module必须在计时前compile/cache hit，并分别做一次untimed fixed-shape forward/backward warmup。compile、
dummy tensor和首次module load不计时；该口径只允许形成warm-cache exact-call claim，不外推cold query。

## 3. Fresh协议

- 6 pair/12 fresh process，顺序=`PM/MP/PM/MP/PM/MP`；
- model/property、三个外部repo、seed=`100`与MR5 correctness完全相同；
- 每worker只接受一个outer optimized exact call；
- headline=`P_host_ns/M_host_ns`，host `perf_counter_ns`包住完整outer call；
- 同current stream CUDA event只作诊断，不替换host headline、不作overlap adjustment；
- 记录device/stream、GPU/driver/temperature/power/clock、base/peak allocated/reserved；
- raw-first、partial拒绝、不得resume或丢弃慢pair。

## 4. Hot-path观测合同

timing region内只保留O(1) counters：

- candidate per-site forward/backward=`10/9`，累计=`30/27`；
- 每site cache=`10 hit/0 miss`（compile和dummy warm发生在region外）；
- β absent、order exact、pending/fallback/eager/native shadow=`0`；
- CPU tensor copy/JSON/hash/trajectory/profiler/NVTX/compile/memory reset=`0`；
- provider/candidate verdict、visited domains、final lower/α/module state按MR5冻结容差保持；
- 六个candidate module/signature receipt必须稳定；
- host/event方向6/6一致，只披露event/host ratio。

## 5. Memory

outer前`synchronize/reset_peak_memory_stats`，记录base/peak allocated/reserved，outer后同步。headline使用
absolute peak candidate/provider ratio；同时披露incremental peak但不升级系统memory claim。precompiled modules
与persistent cache均计入各自base。

## 6. 冻结门禁

### GO：`VALIDATED-MR5-MULTI-CONV-PRODUCTION-BRIDGE-PHYSICS`

全部满足：

- 6/6 correctness/structure通过；
- pair speedup geomean `>=1.05x`；
- 固定seed=`20260826`、10,000 bootstrap 95% lower `>=1.00x`；
- worst pair `>=0.98x`；
- absolute peak allocated/reserved worst ratio均`<=1.05x`；
- host/event方向6/6一致；
- 至少18类fully re-signed tamper、targeted/full regression通过。

通过只开放另行预注册的same-solver complete-query on/off timing，不自动claim queue/B0/B3 parity或
ASPLOS-ready。

### NO-GO

任一失败即`VALIDATED-NO-GO-MR5-MULTI-CONV-PRODUCTION-BRIDGE-PHYSICS`。保留MR5 correctness与MR4 census；
不得事后调阈值、换kernel-only headline或删除慢pair。若失败，下一结构路线必须先归因
compile/cache已排除后的materialization、DLPack、launch与kernel share，不能直接添加第四site。

## 7. 顺序与边界

1. MR5 full regression通过后激活本预注册（已满足）；
2. O(1) timing worker与negative tests；
3. clean-source 12 fresh formal raw/replay/tamper；
4. closure传播；仅GO时才允许complete-query预注册。

当前无正式timing raw，不存在speedup/memory claim。
