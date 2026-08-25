---
status: validated-r3-d1b-isolated-schedule-opportunity
updated: 2026-08-25T18:24:00+08:00
type: closure
topic: boundflow
slug: r3-d1b-schedule-formal-closure
stage: s01
---

# R3-D1-B 固定 Schedule 资格测试正式关闭

## Verdict

`256 threads / serial reduction / vector width 1 / two-kernel materialized scratch` 以
`VALIDATED-R3-D1B-ISOLATED-SCHEDULE-OPPORTUNITY` 关闭。五个 fresh process 的 residual6+11
isolated geomean/worst speedup 为 `58.061911x / 56.862515x`，超过冻结 opportunity gate
`15.50x`；production correctness、compiler receipt、ownership 与 10/10 fully re-signed tamper
均通过。

因此 D1-C cumulative wrapper 现在开放。这里没有测量完整 10/9 optimizer wrapper，不能 claim
wrapper、query、queue 或 ASPLOS end-to-end speedup。

## 冻结证据

- source：`06f87650901b67f2e7e37d467b52e723b2a246ec`；
- artifact：`artifacts/r3-structured-owner/r3-d1b-schedule-formal-v1`；
- protocol hash：`f6457ec447b1571e00fca747c1d1380d566f9cc94576f56cff323c2d1e4d03d0`；
- summary hash：`3a13ad4db7746f0dba98dcac0b84df930fd470c4f0f78475f71227f29660b95f`；
- manifest hash：`1ea112e70fd0d82af4165bb541b2c125b6332d2923f43f0fc760a28e9f667b1b`；
- 5 fresh speedup：`56.8625x / 58.5772x / 59.7686x / 57.2103x / 57.9368x`；
- maximum correctness diff：`9.5367431640625e-07`，sign exact；
- 4 launch、2 scratch、无 persistent dense A；
- 10/10 fully re-signed tamper，targeted `11 passed`。

## 为什么是 58x

主要收益不是 128→256 threads 的微调，而是 D1-A factorization 改变了计算复用边界：v1 raw TIR
在多个输出位置内部重复重算上游转置卷积；staged candidate 先把该结果计算一次写入 caller-owned scratch，
再由下游 kernel 消费。calibration 中 64/128/256 分别为 `56.83x/57.79x/58.64x`，说明线程选择只
贡献小幅增益，结构化复用才是数量级收益来源。

## D1-C 唯一下一动作

在 R3-2B 同一 10 evaluation/9 Adam/9 scheduler wrapper 中，仅替换 residual6/residual11 两个 forward
symbol；保持其他 forward、custom backward、optimizer、state ownership、NC/CN 顺序、3 warmup+30 sample
和 30 秒 pair cooldown 不变。必须同时报告：

1. terminal lower/alpha、10/9/9 counters 与 sign 等价；
2. whole compiled-region speedup `≥9.3181x`；
3. complete wrapper geomean `≥1.20x`、worst `≥1.00x`；
4. allocated/reserved 不高于 R3-2B candidate；
5. fallback/eager/native shadow 为零，scratch 不跨 sample 存活。

isolated 58x 不能替代以上任何一条。
