---
status: preregistered-after-environment-rejection
updated: 2026-08-25T08:35:00+08:00
type: changelog
topic: boundflow
slug: r3-d0-fixed-cooldown-protocol-addendum
stage: s01
---

# R3-D0 固定 Cooldown 协议补充

## 触发事实

`add014b`后的首批formal中，10个worker均完成且校准通过，但pair 3/4 native median分别约
`157.7/159.8 ms`，相对冻结R3-2B reference超出`±15%`。artifact在summary admission处整批拒绝，
没有生成formal目录，也没有裁剪pair。

## 冻结补充

- 每个fresh worker结束后固定idle `30 s`，最后一个除外；
- 等待不根据mode、GPU温度、中间latency或route动态变化；
- worker顺序、3 warmup、30 unprofiled samples、1 profile、±15% sanity、校准与10x路由门槛全部不变；
- 新批次必须从pair 0开始，失败批次raw不复用。

这是笔记本共享散热条件下的环境稳定控制，不形成性能claim。若固定cooldown后仍有任一worker失败，整批
继续fail closed，不再放宽sanity。
