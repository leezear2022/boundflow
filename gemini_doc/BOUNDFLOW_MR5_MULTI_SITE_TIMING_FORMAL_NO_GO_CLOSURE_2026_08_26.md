---
status: validated-no-go
updated: 2026-08-26T22:20:00+08:00
type: closure
topic: boundflow
slug: mr5-multi-site-timing-formal-no-go
stage: s01
---

# MR5 Multi-Site Production Timing 正式 NO-GO Closure

## 1. Verdict

MR5三独立site warm-cache production bridge以
`VALIDATED-NO-GO-MR5-MULTI-CONV-PRODUCTION-BRIDGE-PHYSICS`关闭。

Correctness没有失败：所有pair allclose/sign exact，三site仍为每worker `30/27` forward/backward，
cache均为`0 miss + 10 hit`。失败的是性能传播：完整outer exact-call host geomean只有
`0.8344066482697061x`，即candidate约慢`19.84%`；因此same-solver complete-query timing继续关闭。

本结论只杀死“3个独立TIR module + 每site独立validation/DLPack/launch/materialization”的当前runtime
形态，不否定MR5三site correctness，也不否定独立CIBC-IBP整图`2.4563x`结果。

## 2. Frozen provenance

- worker source=`24a208140b73ed943d983ea73f2a20f842a19015`；
- formal gate commit=`9c08bf6a2636768770c3af9c97c7852fba750820`；
- artifact=`artifacts/measurement-recovery/mr5-multi-conv-timing-v1/`；
- summary hash=`b9e3b766408083f20e6c533ad35caf05bcbce7437e90b3c68747ee7b03fb07e9`；
- manifest SHA256=`617f130c040d9308441da1ae85aeb6dbaf5ecaaf66b26122817eeeef244ad793`；
- raw SHA256=`558fde6b823ce53a909a779223e7ffe5b6ad480b9dcd47c6f8dcc6ab4c87b612`；
- tamper report SHA256=`1d086e4173a085a001468b1a51e59f53f2384e46adb7de60122027d0d956fd24`；
- correctness identity先经amendment重签并replay，summary数值未变；timing protocol绑定新manifest。

## 3. Six-pair result

| Gate | Frozen threshold | Result | Pass |
|---|---:|---:|---|
| host geomean | `>=1.05x` | `0.83440665x` | no |
| bootstrap 95% lower | `>=1.00x` | `0.82849773x` | no |
| worst pair | `>=0.98x` | `0.82464351x` | no |
| absolute peak allocated | `<=1.05x` | `0.99457036x` | yes |
| absolute peak reserved | `<=1.05x` | `1.00000000x` | yes |
| host/event direction | `6/6` | `6/6` | yes |
| correctness/module/pair count | exact | exact | yes |

六个host speedup为`0.839349/0.830704/0.834649/0.828195/0.824644/0.849131x`，全部低于1；
provider约`100.76—109.15 ms`，bridge约`122.19—128.96 ms`。host与CUDA event逐pair方向一致，
所以不是host计时噪声或时钟域错配。

## 4. Correctness与证据完整性

- 12 fresh process，顺序=`PM/MP/PM/MP/PM/MP`；
- 每pair semantic elements=`9540`，global max diff=`4.708766937255859e-06`，sign exact；
- 三module receipt在6个candidate process完全稳定；
- compile与3次dummy forward/backward warm排除在headline外；
- outer内每site cache=`0 miss/10 hit`，fallback/eager/native shadow=`0/0/0`；
- replay从raw重算全部pair metric与summary；
- 20/20 fully re-signed timing、semantic、module、cache、workspace、stream与顺序攻击被拒绝；
- `performance_claimed=false`、`same_solver_complete_query_timing_open=false`由summary强制。
- closing full regression=`1796 passed, 3 skipped, 6 warnings`，耗时`685.61s`；MR6新增CPU测试另行
  `3 passed`，不混入MR5冻结计数。

## 5. 可行动根因：热路径同步guard，而不是compile/cache

当前代码静态审计发现每个site evaluation至少有12个会迫使device→host同步的value guard：

1. `validate_mr5_generalized_conv_tensors`对7个输入逐个`isfinite(...).all().item()`；
2. lower≤upper和α∈[0,1]各一次`.item()`；
3. ReLU→Conv handoff使用`torch.equal`做content equality；
4. TIR输出A/bias各一次`bool(isfinite(...).all())`。

三site×10 evaluation即至少`12×30=360`个同步guard。compile已排除、cache全hit、显存未增加，
而host/event同向，故下一步先量化这些guard占多少损失；尚不能仅凭代码计数宣称它们解释全部约
`21 ms`差额。另有permute/contiguous、zero allocation、DLPack封装和57次独立kernel launch，必须在
后续账本中保持为独立项。

## 6. Claim boundary与下一门禁

- 保留：MR5 generalized TIR、多shape/stride correctness、三site production ownership、atomic rollback；
- 关闭：当前三独立site runtime的性能传播、same-solver complete-query、queue、B0/B3 parity；
- 不得：降低`1.05/1.00/0.98x`门槛、挑pair、把CIBC独立IBP图结果代入本路径；
- 下一唯一动作：MR6 hot-path guard attribution。它先做unsafe diagnostic ceiling，不形成性能claim；
  只有证明确为dominant后，才实现“静态admission + kernel内device status + outer单次commit check”的
  fail-closed replacement。
