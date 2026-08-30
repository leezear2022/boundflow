---
status: validated-s4-0-mutable-state-admission
date: 2026-08-30
type: external-audit-closure
topic: boundflow
slug: asplos27-s4-0-external-audit-closure
stage: s04
assurance-level: E2-DIRECT-LEGACY
performance-claimed: false
---

# S4-0 external audit closure

## 结论

外审判定`approve-with-minor-correction`，0 blocker、0 major、2 minor、2 info。执行方接受F1/F2并已完成
权威文档修正，因此S4-0以`VALIDATED-S4-0-MUTABLE-STATE-ADMISSION`关闭。该状态只证明真实provider
mutable-state admission、receipt与local lease correctness，不是性能结论。

## Findings响应

### F1：接受并关闭

原“10/10全重签攻击拒绝”的措辞过宽。现统一改为：10/10攻击都重签内外hash，但仍与冻结派生语义不一致，
因此被semantic replay拒绝；外审第11类coherent full resign同步伪造source/raw/protocol/summary/manifest，E0
self-check接受。source物理真实性来自外审对三个外部仓库commit/model/property的独立核验，以及外审控制的
5-fresh真实provider执行，不来自自签manifest。

### F2：接受并关闭

5个admission hash不同不能全归因于run ordinal。exact-call identity hash直接绑定ordinal；另外四个
plan/snapshot/oracle/plan-binding hash绑定每进程provider snapshot的全量metadata/history，因正常捕获变化而不同。
外审fresh与formal共10个进程的admitted α/β slots及content hash逐位一致，故不影响admission结论。

### F3/F4：记录

- 本轮没有challenge字段，保证等级限定为`E2-DIRECT-LEGACY`；S4-4不得复用该例外；
- 外审环境没有dol PATH；执行方使用仓库固定DocOps CLI补跑lint并记录validation。

## 证据与下一门禁

- 外审报告：`gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_EXTERNAL_AUDIT_REPORT_2026_08_30.md`；
- 外审独立脚本：`artifacts/asplos27-s4-admission/audit-20260830/`；
- 正式artifact保持不变：`artifacts/asplos27-s4-admission/resnet2b-prop0-v1`；
- 开放：S4-1A persistent compressed α/active-β buffer implementation/correctness；
- 关闭：S4-1A timing/performance、TIR evaluator、same-solver、complete-query、10x、ASPLOS-ready。
