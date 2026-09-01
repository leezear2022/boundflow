---
status: implemented-pre-formal
updated: 2026-08-26T16:05:00+08:00
type: changelog
topic: boundflow
slug: mr3-single-site-production-bridge-timing-worker
stage: s01
---

# MR3 Single-Site Production Bridge Timing Worker 修改记录

## 1. 目的

按已提交的MR3 timing预注册，实现一个独立、低观测扰动的production exact-call worker。它只测冻结
ResNet2B property-0的一次beta-split optimized outer call；provider保留原路径，bridge只替换P-anchor
`/49: /input-24 → /input-20` lower region。

## 2. 修改

- 新增`scripts/run_mr3_production_bridge_timing_worker.py`；
- bridge侧在计时前完成TIR compile/cache与固定shape dummy forward/backward warm；
- bridge侧冻结module/device-source hash、TVM版本、exported symbols与dummy `1/1/0/0`
  launch/fallback/eager receipt，供6个fresh candidate进程做稳定性门禁；
- CUDA event在tracker初始化时预分配，避免event构造进入计时region；
- headline记录host `perf_counter_ns`，同一current stream event只作为diagnostic；
- outer call前同步并reset peak，记录absolute base/peak allocated/reserved；
- timing region内不启用formal observer，不做tensor-to-CPU trajectory、hash、JSON或failure injection；
- outer call返回后才生成solver verdict、final lower/alpha/module state与bridge receipt；
- 输入model/property、三个外部仓库commit、设备与stream全部fail closed并进入worker hash。

## 3. 非正式预检

在未提交source上只运行一对provider/bridge用于排错，不计入formal raw：

- solver verdict与visited domains exact；
- outer/final alpha/module共9,540个数值元素通过冻结容差，最大绝对差
  `1.7695128917694092e-06`；
- bridge forward/backward=`10/9`，fallback/eager/native shadow=`0/0/0`；
- provider/bridge host分别为`108.010533 ms`/`109.432906 ms`，单对ratio=`0.987002x`；
- absolute peak allocated/reserved ratio分别为`1.032240x`/`1.032258x`。

这些数字只证明worker可运行并帮助发现实现偏差，不是正式性能或memory claim；预检目录不提交，正式
决策仍必须来自后续clean source的冻结6 pair/12 process artifact。

## 4. 验证

- Black target py312：通过；
- mypy（worker）：通过；
- pylint（worker）：`10.00/10`；
- RTX 4060 Laptop GPU真实provider/bridge预检：均成功闭合。

## 5. Claim边界

本提交不得声明MR3加速或memory收益，也不得以预检失败直接关闭路线。下一步是对clean implementation
commit建立机械summary/replay/tamper门禁，再运行预注册formal协议。
