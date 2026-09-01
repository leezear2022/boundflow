---
status: d1c-validated-no-go-superseded-by-d2a
updated: 2026-08-25T15:10:00+08:00
type: plan
topic: boundflow
slug: r3-d1-compiled-region-schedule
stage: s01
---

# R3-D1 Compiled Region Schedule/Fusion 预注册

## 1. 目标与边界

D1只验证：在保持R3-2A 10/9 optimizer语义、无跨evaluation persistent dense A、最多2个既有arena scratch
的条件下，能否把R3-D0的whole compiled recurrence至少加速到5-pair worst required `9.3181x`，并让完整
candidate wrapper达到`candidate <= native / 1.20`。

禁止改native oracle、减少10/9 action、改变α/β/split/history、删optimizer、用CUDA Graph掩盖kernel、
只报isolated kernel headline、复活R3-3或same-solver claim。

## 2. 冻结证据与算法诊断

- D0 formal source=`4232069`，compiled share=`99.6627%–99.7257%`；
- residual6=`66.06%–67.02%`，residual11=`28.27%–29.10%`，effective_pre23约`3.5%`；
- 当前`_residual6_primfunc`/`_residual11_primfunc`直接保留raw TIR，没有进入通用TE schedule；
- 两个PrimFunc在每个目标系数内重复执行前一层conv-transpose reduction，再乘ReLU slope和下一层权重；
- residual6+11只优化时worst required=`15.4733x`，whole region统一优化时worst required=`9.3181x`。

## 3. D1-A：两阶段语义factorization

只新增v2 symbol，不原地替换v1：

```text
residual11:
  stage-1: incoming --conv10^T--> scratch0[6,1024]
  stage-2: scratch0 * selected_relu25_slope --conv8^T--> output + skip

residual6:
  stage-1: incoming --conv4^T--> scratch0[6,1024]
  stage-2: scratch0 * selected_relu19_slope --stride2 conv2^T--> output
           + incoming --1x1 conv5^T--> output
```

bias/intercept必须使用同一中间值和同一slope所有权；允许第二scratch做partial reduction，禁止把scratch写入
plan/state/optimizer并跨evaluation存活。必须输出scheduled TIR、buffer inventory、symbol/module hash与
zero-fallback receipt。

D1-A correctness：5 fresh process，每个同时比较v1 symbol、独立PyTorch/R3-2A oracle和v2 staged输出；
lower/gradient绝对误差`≤2e-4`、sign exact、bias tolerance同上；所有shape/dtype/device/hash/nonfinite/
stream篡改fail closed。D1-A不计时、不claim speedup。

## 4. D1-B：冻结schedule搜索空间

只允许以下预注册候选：

- threads per block：`64/128/256`；
- reduction：serial reference、shared-memory tiled、warp-shuffle partial；
- stage-1/stage-2：two-kernel materialized scratch；只有two-kernel通过后才允许producer-consumer融合；
- vector width：`1/2/4`，仅在alignment和tail exact证明后准入；
- residual skip和bias epilogue允许融合进stage-2；不得融合跨越下一个ReLU owner。

固定一个calibration capture选择每个shape/signature winner；5 fresh correctness/timing只验证winner，不得在
formal pair间重选。compile/cold/warm cache与module receipt全部冻结。

isolated opportunity gate：residual6+11组合5 fresh worst speedup应`≥15.50x`，否则它们单独无法满足最终
目标；若未达，不立刻宣告whole-region NO-GO，而是先量化effective_pre23/其余region加入后的新Amdahl界。

## 5. D1-C：whole compiled region累计门禁

只有D1-A/B correctness全过后，才允许在R3-2B同一10/9 wrapper中替换compiled region：

- 5 fresh pair，`NC/CN/NC/CN/NC`，每worker 3 warmup + 30 host-wall sample，固定cooldown 30秒；
- 同R3-2B reference `±15%` sanity；terminal lower、α、10 eval/9 Adam/9 scheduler、sign/counter等价；
- candidate fallback/eager/native shadow为0，stream/device/arena pointer保持；
- whole compiled region 5-pair worst speedup `≥9.3181x`；
- 完整wrapper geomean `≥1.20x`且worst `≥1.00x`；
- peak allocated/reserved不高于R3-2B candidate，scratch仍为2且不跨sample存活。

全部通过才可关闭为`VALIDATED-R3-D1-P-LOCAL-WRAPPER`并重新评估R3-3。任一语义失败直接NO-GO；性能
不足则关闭为`VALIDATED-NO-GO-R3-D1-SCHEDULE`，不得以isolated数字代替wrapper。

## 6. Artifact与防篡改

formal绑定D0 manifest、code blob、TIR/device-source hash、schedule choice、raw timings、terminal tensors、
arena pointer与compile/cache receipt。replay至少拒绝schedule、thread/vector/reduction、scratch lifetime、
symbol order、duration、terminal、optimizer counter、fallback、target、required speedup、route和performance
claim的fully re-signed tamper。

## 7. 当前动作

**历史准入（已被下方closure取代）**：residual11与residual6 D1-A correctness已关闭；D1-B固定
256-thread winner以5 fresh isolated geomean/worst=`58.0619x/56.8625x`、10/10 tamper关闭；当时
只开放D1-C cumulative wrapper。R3-3与same-solver仍关闭；不得修改v1或D1-A/B formal artifact。

**2026-08-25 closure**：D1-C formal wrapper geomean/worst=`0.249369x/0.243233x`，相对B3
recovery=`1.879305x/1.855758x`，以`VALIDATED-NO-GO-R3-D1C-CUMULATIVE-WRAPPER`关闭。
本计划不再开放任何D1实现；唯一后继为D2-A只读backward attribution。
