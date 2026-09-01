---
status: validated-no-go-b4c2-dense-retention
updated: 2026-08-24T11:40:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4c2-materialization-frontier-kill
stage: s01
---

# FSG4/B4-C2 Materialization Frontier Kill Changelog

## Verdict

`VALIDATED-NO-GO-B4-C2-DENSE-RETENTION`。本候选在3个fresh GPU worker上触发预注册kill gate，
不进入6 fresh或B4-D。

## 实现与证据

候选接管production optimizer每次evaluation的6个真实lower ReLU materialization sites：
`31/28/25/23/19/17`，共10×6=`60`次。它复用bias reduction已经生成的dense coefficient，避免
下一消费点重新物化同一operator tree。receipt严格要求60/60、逐site 10/10、fallback=0。

三个30组interleaved worker结果：

- paired speedup=`[0.348761,0.337448,0.346003]`；
- terminal lower/α/β max diff三轮均为`4.768e-7`，sign exact；
- peak allocated ratio三轮均=`1.3401085408885496`。

## Root Cause

该候选消除了算子树的重复forward materialization，却把每层lower coefficient变成跨层存活的dense
autograd tensor。native structured path只保存轻量operator tree并在最终消费点组合求值；dense retention
则让6层大tensor及其autograd history同时存活，导致约2.9×回退和34%显存增加。这个问题不能靠
schedule tuning修复，必须用自定义forward/backward kernel在每个边界释放中间态。

## B4 Closure

B4-C0 bridge、B4-C1单anchor provider、B4-C2全frontier dense retention三条累计候选均未达到
no-regression。按kill纪律，本轮alpha-CROWN纵向ReLU→affine融合以`VALIDATED-NO-GO-B4`关闭；
B4-D same-solver timing不开放，避免对已确定回退的候选追加正式实验。

下一路线回到CIBC论文真正验证过的目标：IBP/forward bound operator的**水平融合**（同shape的
lower/upper、center/deviation、正负权重分支）与schedule autotuning，而不是继续扩大alpha-CROWN
纵向dense autograd链。该路线必须独立建立operator与whole-model门禁，不继承B4性能claim。
