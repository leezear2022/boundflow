---
status: implemented-r3-2b-runtime-awaiting-formal
updated: 2026-08-25T05:48:00+08:00
type: changelog
topic: boundflow
slug: r3-2b-timing-runtime-implementation
stage: s01
---

# R3-2B Timing Runtime 实现修改记录

- 新增persistent compiled candidate owner，在setup阶段建立TIR module、scratch和DLPack views；
- 新增capture-free compiled forward/custom VJP，移除timed path的SHA、CPU copy、receipt构造、sync和
  memory-stat reset；
- 新增native/candidate共同10/9 wrapper，Adam/scheduler/clamp均在wrapper内部；
- candidate每evaluation保留O(1) tensor identity、immutable version、ordinal、lr和stream门禁；
- 新增terminal lower/α parity与timed-source隔离测试；
- 新增fresh worker，冻结3次warmup、30个host-wall样本、untimed terminal/memory capture与原始样本输出；
- 新增atomic five-fresh artifact runner、逐raw semantic/statistical replay、protocol/code revision与exact
  inventory验证；
- 一对非正式diagnostic得到native/candidate median约`99.64/754.64 ms`，speedup=`0.13203x`，
  correctness与memory仍通过。该值不替代5-pair formal，不据此修改协议；
- 尚未运行formal timing，当前无performance claim。

formal已完成采样且初步replay为NO-GO；新增10类fully re-signed tamper probe，覆盖latency raw、terminal
lower/α、counter、memory、sample count、clock、protocol threshold和summary verdict/order。全部拒绝后才
形成正式NO-GO closure。
