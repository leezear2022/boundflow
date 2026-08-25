---
status: preregistered
updated: 2026-08-26T15:45:00+08:00
type: plan
topic: boundflow
slug: mr3-single-site-production-bridge-timing
stage: s01
---

# MR3 Single-Site Production Bridge Timing 预注册

## 1. 准入与唯一问题

前置MR3 correctness已以`VALIDATED-MR3-P-PRODUCTION-BRIDGE-CORRECTNESS`关闭。本阶段只回答：

> 在冻结ResNet2B property-0的同一个αβ-CROWN beta-split optimized exact call中，只替换P-anchor
> `/49: /input-24 → /input-20` lower region后，完整10-evaluation/9-mutation outer exact call是否比
> 原provider调用更快，并且不增加absolute peak memory？

不改TIR/template/schedule，不扩S-anchor或第二site，不测完整query/queue，不读取formal worker里受CPU
copy污染的日志时间。MR3 correctness artifact必须先replay。

## 2. Baseline与Candidate

### Provider P

原αβ-CROWN exact call。provider拥有全部site、10/9 optimizer、split/history、termination。计时region
从outer optimized call进入到返回，包含10次native P lower region与其余全部工作。

### Bridge B

同一exact call，仅P lower region由已关闭correctness的CIBC TIR forward/custom backward替换。计时
region包含每evaluation的admission、DLPack view、plan buffer、10/9 module launch、autograd与provider
optimizer。不得运行native P lower shadow、formal tensor-to-CPU轨迹、owner snapshot或failure injection。

candidate module必须在计时前compile/cache hit，并以一个固定shape dummy forward/backward完成untimed
module/driver warmup；compile、dummy tensor与首次module load均不计时。provider在进入本exact call前已由
同一solver的initial CROWN自然warm。两侧均披露该非对称来源，不得把compile排除外推为cold-query claim。

## 3. Fresh协议

- 6 pair/12独立进程，顺序=`PB/BP/PB/BP/PB/BP`；
- 每进程从冻结model/property和seed=`100`启动，只执行一个solver query、只接纳一个outer exact call；
- outer call只记录一次host `perf_counter_ns`，headline ratio=`P_host_ns/B_host_ns`；
- 同时用同一current stream的一对预分配CUDA event包住outer call，作为diagnostic而非headline；
- 每进程记录call前后current device/stream、GPU name/driver、temperature/power/clock/power-limit；
- raw-first，12个worker任一失败、缺失或已有partial raw时整体拒绝，不允许resume；
- pair按同一position配对，不能跨pair重排或丢弃慢样本。

## 4. 观测剥离与结构门禁

timing worker只保留O(1) counter与最终semantic digest：

- bridge forward/backward=`10/9`、fallback/eager/native shadow=`0/0/0`；
- provider和bridge solver verdict/visited domains exact；final lower/α/module owner state用MR3冻结容差复核；
- timing region内CPU tensor copy、JSON、hash、trajectory、profiler/NVTX、compile与memory reset=`0`；
- candidate precompile/dummy warm receipt与module hash六次稳定；
- event bracket不得改变host headline；host/event ratio仅披露，不做overlap adjustment。

## 5. Memory口径

每个worker在outer call前`synchronize/reset_peak_memory_stats`，记录base/peak allocated与reserved；call后
再次synchronize。headline只使用absolute peak ratio `B/P`，同时披露incremental peak但不据此升级
system memory claim。precompiled module与prepared solver state已在两侧各自base内。

## 6. 冻结门禁

### GO：`VALIDATED-MR3-P-PRODUCTION-BRIDGE-PHYSICS`

必须同时满足：

- 6/6 correctness与结构门禁通过；
- pair speedup geomean `>=1.05x`；
- 固定seed=`20260826`、10,000次bootstrap 95% lower bound `>=1.00x`；
- worst pair `>=0.98x`；
- candidate/provider absolute peak allocated与reserved worst ratio均`<=1.05x`；
- host与CUDA-event方向6/6一致；
- 至少14类fully re-signed tamper拒绝；targeted/full regression通过。

通过只开放另行预注册的same-solver complete-query bridge-on/off timing。不得直接claim query/queue
speedup、B0/B3 parity、multi-site或ASPLOS-ready。

### NO-GO

任一门禁失败则关闭为`VALIDATED-NO-GO-MR3-P-PRODUCTION-BRIDGE-PHYSICS`。保留MR3 correctness，
不得调阈值、改用kernel-only时间或删除慢pair；complete-query timing继续关闭。下一步只能做无扰动
O(1) counter账本或重新预注册新的结构路线。

## 7. 提交顺序

1. 本预注册；
2. timing worker + synthetic/negative tests；
3. clean source formal raw/replay/tamper；
4. closure与claims传播。

预注册提交前零正式样本；实现提交不得携带formal raw。

## 8. Claim边界

本文件冻结实验，不代表实验已运行。当前仍只能claim MR3 production bridge correctness，不能claim任何
新增性能或memory收益。
