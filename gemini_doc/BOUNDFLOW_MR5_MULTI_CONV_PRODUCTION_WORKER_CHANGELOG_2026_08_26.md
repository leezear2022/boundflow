---
status: implemented-preflight-passed
updated: 2026-08-26T19:35:00+08:00
type: changelog
topic: boundflow
slug: mr5-multi-conv-production-worker
stage: s01
---

# MR5 Multi-Conv Production Worker 修改记录

## 1. 实现

- 新增C2→C1→C0三site fail-closed bridge；每site独立signature/module receipt/cache；
- 在真实αβ-CROWN outer optimized exact call内替换三条lower ReLU+Conv路径；
- provider继续拥有α、Adam、clip、split/history、termination与非三site图；
- 复用既有outer atomic owner snapshot，在evaluation 5、C1完成后支持注入失败与完整rollback；
- worker同时支持纯provider与bridge，保存逐site、逐evaluation、optimizer/final/module状态；
- 无timer，`performance_claimed=false`。

## 2. 非正式fresh预检

一组独立provider/bridge：

- verdict=`verified/verified`，visited domains=`6/6`；
- candidate每site forward/backward=`10/9`，累计=`30/27`；
- cache每site=`1 miss + 9 hit`；
- β每site=`10 × [6,0]`，总numel=`0`；
- handoff content每site=`10/10`，pending/fallback/eager/native shadow=`0/0/0/0`；
- provider/candidate递归状态值各`425,952`个，最大绝对差约`4.41074e-6`；
- C0/C1/C2 module receipt与signature hash各自不同。

独立failure worker：evaluation 5、C1后异常；owner tensor=`12`，content hash与pointer hash前后exact，
version delta=`[1,6]`，staged/commit/rollback=`0/0/1`。

这些数字只用于确认formal值得启动，不形成正式correctness或性能claim。

## 3. 静态与合成验证

- generalized + bridge focused=`15 passed`；
- mypy五文件clean；
- pylint=`10.00/10`；
- Black与diff check通过。

## 4. 下一步

以本提交作为clean source，生成冻结`PB/BP/PB/BP/PB`五pair raw、独立rollback raw、formal replay与
fully re-signed tamper。正式artifact通过前timing仍关闭。
