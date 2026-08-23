---
status: implemented-pending-formal
updated: 2026-08-24T12:10:00+08:00
type: changelog
topic: boundflow
slug: cibc-ibp-horizontal-fusion-implementation
stage: s01
---

# CIBC IBP Horizontal Fusion Implementation Changelog

## 目标纠正

CIBC论文的`up to 40x`来自forward-bound operator的**水平融合**，不是此前B4的纵向
alpha-CROWN ReLU→Conv替换。本阶段按论文公式实现：一个TIR kernel同时计算
`mid/radius -> center/deviation -> lower/upper`，取代PyTorch的weight正负拆分、4次Conv和多次
elementwise调用。

## 实现

- 新增typed Conv signature，冻结NCHW/weight/stride/padding/dilation/groups；
- 一个manual CUDA/TIR kernel同时输出lower与upper，共享输入、权重和reduction traversal；
- schedule候选固定为64/128/256 threads，不在计时后追加候选；
- plan-owned DLPack views、输出buffer和packed function，hot path保持1 launch；
- `IntervalDomain`通过显式context opt-in，默认PyTorch路径不变；
- 新增prepared CUDA graph，把6个真实ResNet2B Conv、ReLU、residual add、flatten、2个Linear纳入
  同一静态地址执行计划；baseline也用CUDA graph，避免把graph replay收益冒充TIR收益。

## 当前诊断（非正式claim）

- 固定`(6,16,8,8)×(16,16,3,3)` operator：plan-owned TIR约`0.00585 ms`，
  PyTorch四Conv约`0.04514 ms`，约`7.72x`；
- ResNet2B完整IBP graph replay：baseline约`0.1816 ms`，CIBC约`0.06727 ms`，约`2.70x`；
- 6/6 Conv由TIR执行；所有中间interval最大绝对差约`2.4414e-4`，最终logit bound最大差
  `1.8311e-4`。

这些数字只用于确认路线物理上成立。正式结果必须使用fresh process、counterbalanced schedule、输入copy
计入、逐层allclose/sign检查、operator与whole-model分层、root replay和tamper；完成前
`performance_claimed=false`。

## 下一步

生成CIBC-IBP formal artifact：6个真实Conv signature×3 schedule的operator sweep，以及6 fresh
baseline-graph/CIBC-graph成对计时。正式门禁暂定operator geomean≥2x、whole-model geomean≥1.5x、
worst pair≥1.2x、最终及逐层`atol=rtol=3e-4`、sign exact、peak allocated/reserved≤1.05x。
