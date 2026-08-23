---
status: implemented-b4-c0-cumulative-core-runner-pending-formal
updated: 2026-08-24T05:30:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4c0-cumulative-core-runner
stage: s01
---

# FSG4/B4-C0 Cumulative Core Runner Changelog

新增双方各3 warmups、30 interleaved groups的fresh cumulative-core worker。每个candidate call新建
exact observer但复用clean-source编译module；compile、模型导入与PlanTemplate准备不计时，完整10/9
optimizer schedule、observer/PlanInstance runtime、TIR launches及native-value bridge均计时。每组
重复核对terminal lower与全部α/β allclose/sign exact，并独立测peak allocated/reserved。

timing模式同时关闭evaluation-0 correctness capture，receipt以
`correctness_capture_enabled=false/unsupported_semantic_anchor_count=0`显式冻结；正确性模式默认行为
不变。

单worker pilot在充分预热后B3/B4-C0 median=`81.918/87.519 ms`，ratio=`0.9360x`。该值只用于验证
runner物理合理性，不形成正式结论；它预示native-value bridge累计候选可能NO-GO。

下一步：构建6 fresh、BC/CB全排列artifact/replay并按预注册no-regression与`1.05x`research gate关闭。
