---
status: validated-mr5-correctness-preregistration-open
updated: 2026-08-26T17:30:00+08:00
type: plan
topic: boundflow
slug: mr4-production-conv-site-census
stage: s01
---

# MR4 Production CROWN Conv Site Census 预注册

## 1. 前置与问题

MR3已以`VALIDATED-NO-GO-MR3-P-PRODUCTION-BRIDGE-PHYSICS`关闭：只替换
`/49: /input-24 → /input-20`一个P-anchor时，完整outer exact call host geomean仅`0.979727x`，
不能传播到complete-query。

本阶段不复活MR3、不改TIR、不计时，只回答新的结构问题：

> 同一个真实beta-split optimized exact call中，除P-anchor外是否还存在由provider实际执行、具有
> 稳定10/9轨迹、absent β和直接ReLU→Conv consumer closure的同构Conv sites，使“多site累计替换”
> 值得进入独立correctness预注册？

MR2只在既有P/S证据中选最近site，不是全图动态census；MR4因此必须重新从真实provider调用观察，
不能把MR2的“两候选”误写成“生产图只有两个site”。

## 2. 冻结测量对象

- model/property、seed、batch、5 α steps、10 β steps与MR3完全相同；
- 只运行provider，不执行任何candidate/TIR，不记录latency或peak memory；
- hook仅在一个outer beta-split optimized call内启用；
- 对`start_node=/49`的每次ReLU backward和相邻直接Conv backward只记录O(1) metadata/counters；
- outer call返回后才生成final solver/result digest；
- 5个独立fresh process，顺序固定为run 0…4，不允许resume或丢run。

## 3. 冻结候选拓扑

基于已冻结ResNet2B provider graph，预注册三条直接ReLU→Conv边：

1. C0：`/input-4 ← /input`；
2. C1：`/input-12 ← /input-8`；
3. C2/P：`/input-24 ← /input-20`。

`/input-16 ← /39`与`/45 ← /44`的predecessor为Add，`/48 ← /input-28`为Linear，均不准伪装为
Conv site。若运行时发现第四条直接Conv边或三条中任一缺失，census fail closed，先修schema而不是
事后改候选集合。

## 4. 每site账本

每个site、每个evaluation必须记录并由raw重算：

- ReLU/Conv identity、direct predecessor与call ordinal；
- incoming lower-A shape/dtype/device/contiguity；
- preactivation lower/upper shape/dtype/device、finite与`lower<=upper`；
- compressed α shape、requires-grad、reconstructed full-α shape；
- β tensor count/shape/numel，必须明确区分absent与active；
- Conv weight/bias shape/dtype/device；
- ReLU handoff与Conv input的shape/content/pointer identity；
- provider Conv输出A/bias shape；
- grad-enabled evaluations与final no-grad evaluation；
- 由shape机械计算的forward MAC units和candidate最低materialization bytes。

不保存完整tensor payload，不调用profiler/NVTX/CUDA event，不在hook中`synchronize`。

## 5. 静态机会量

以P-anchor C2的单evaluation forward MAC units为`1.0 P-unit`：

- `site_mac_ratio_to_p = site_mac_units / P_mac_units`；
- `eligible_total_mac_ratio_to_p = Σ eligible site ratios`；
- `new_site_mac_ratio_to_p = eligible_total - 1.0`；
- 同时披露若每site保持独立TIR wrapper时的projected forward/backward launch=`30/27`，它是风险，
  不能被MAC ratio当作收益抵消。

MAC只是结构机会量，不是时间share、speedup或Amdahl claim。

## 6. 机械准入

只有全部满足才输出
`OPEN-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS-PREREGISTRATION`：

1. 5/5 fresh solver verdict与visited domains exact；
2. 三条冻结edge恰好出现，无额外direct Conv edge；
3. 每site恰10次evaluation，其中9次grad、1次final no-grad；
4. 每site start-node、shape/dtype/device/contiguity在5 fresh内稳定；
5. 每次β均为一个`[6,0]` tensor、numel=0；
6. ReLU→Conv handoff content 10/10 exact，fallback/repeated/pending=`0`；
7. 三site均可重建full α，provider输出结构稳定；
8. `eligible_total_mac_ratio_to_p >= 1.75`且`new_site_mac_ratio_to_p >= 0.75`；
9. replay及至少14类fully re-signed tamper拒绝；targeted/full regression通过。

通过只开放MR5 **correctness预注册**。不得直接实现timing、宣称multi-site speedup、把MAC ratio写成
GPU share，或改写MR3 NO-GO。

任一项失败则状态=
`VALIDATED-NO-GO-MR4-PRODUCTION-CONV-SITE-CENSUS`，multi-site当前路线关闭。

## 7. MR5边界（仅在MR4通过后）

MR5最多可以预注册：三site provider vs generalized bridge five-fresh correctness、逐site10/9轨迹、
atomic rollback与完整outer result等价。MR5必须先证明三site cumulative ownership，不得直接计时；
即使correctness通过，后续timing仍需新预注册，且对照基线仍是原provider。

## 8. Claim边界

本文件只冻结census协议。当前没有MR4 raw、没有multi-site implementation，也没有新增性能或memory
claim。MR3 single-site NO-GO与CIBC full-graph IBP reduced claim均保持原样。

## 9. 事后结果注记（不修改原门禁）

5 fresh/150 rows全部通过；C0/C1/C2 MAC ratio=`1.5/2.0/1.0P`，total=`4.5P`、new sites=`3.5P`；
global semantic max diff=`3.516674041748047e-06`，16/16 tamper。状态=
`OPEN-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS-PREREGISTRATION`；timing仍关闭。见
`BOUNDFLOW_MR4_PRODUCTION_CONV_SITE_CENSUS_FORMAL_CLOSURE_2026_08_26.md`。
