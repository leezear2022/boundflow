---
status: implemented-b4-b3-cibc-exact-call-pending-five-fresh
updated: 2026-08-24T04:05:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4b3-cibc-exact-call-implementation
stage: s01
---

# FSG4/B4-B3 CIBC Exact-call Implementation Changelog

## Scope

把B4-B2 v2融合结果接入10-evaluation/9-update production optimizer。P-anchor在每次evaluation
激活dense-alpha manual TIR；S-anchor保持B3并显式记录unsupported。本轮没有whole-core/query
performance claim。

## Ownership Correction

首次接线证伪了“86-coordinate compressed α可直接替换live optimizer α”的假设：局部frozen-adjoint
replay虽通过，10-step mutation会漂移。production optimizer实际更新完整`[6,16,8,8]`native α，且
incoming lower bias也是有autograd所有权的上游状态。最终实现因此：

- 新建dense-alpha TIR，forward直接消费完整native α；
- backward在同一kernel返回完整native α gradient与incoming-A gradient；
- incoming-bias identity gradient显式回传，不再被错误截断；
- broadcast bias seed以零拷贝scalar view传入TIR，无materialization kernel；
- exact `10 forward + 9 backward`，fallback/eager/materialization均为0；
- P-anchor TIR float32值与native路径存在`~1e-7` reduction-order差异，会被Adam的符号归一化放大；
  因此当前通过无kernel的exact-value/candidate-gradient bridge保留native float32轨迹，同时让一阶梯度
  完整走TIR。receipt明确冻结`native_value_bridge_count=10`，不隐瞒该边界。

## Current Evidence

- dense TIR public-PyTorch forward/VJP测试通过，sign exact；
- exact-call smoke完成10/9，terminal lower与全部α/β state allclose/sign exact；
- smoke maximum absolute difference=`3.5762786865234375e-07`；
- evaluation-0 local output-A/bias max diff=`2.2351741790771484e-08`/
  `1.1920928955078125e-07`；
- receipt：provider=10、forward/backward=`10/9`、S unsupported=1、fallback/eager/materialization=
  `0/0/0`；
- targeted dense/CIBC tests=`7 passed`；mypy 4 files clean。

一次wall-clock smoke显示first B3/candidate=`849/143 ms`，但first-call cuDNN warmup与顺序偏差未
排除，明确不是performance evidence。

## Next

从clean source运行5个fresh B3/B4-B3 semantic pairs并做root replay；关闭后进入累计core timing。
累计计时必须预热、交错BC/CB并披露exact-value bridge成本。在移除native value bridge、让CROWN在
provider-owned region跳过native lower branch之前，不得主张该integration产生求解器加速。
