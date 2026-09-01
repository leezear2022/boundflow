---
status: implemented-awaiting-clean-source-formal-run
updated: 2026-08-26T19:55:00+08:00
type: changelog
topic: boundflow
slug: mr5-multi-conv-formal-gates
stage: s01
---

# MR5 Multi-Conv Formal Gates 修改记录

- 冻结worker source=`3e1a70933910c009019c59de4f44d233a75f7950`；
- 冻结5 pair/10 fresh顺序=`PB/BP/PB/BP/PB`；
- 机械复核三site逐evaluation顺序、150/135 launch、β/consumer/pending/cache closure；
- 机械复核三个site-specific signature、TIR/device source与workspace receipt跨fresh稳定；
- 逐site region、inner/outer/final/module与optimizer trajectory分别按`2e-4/2e-5`比较且sign exact；
- 冻结evaluation 5/C1 failure后的12-owner content/pointer/version rollback；
- replay绑定11个code path digest与artifact内每个文件digest；
- tamper probe覆盖21类fully re-signed semantic/lifecycle/compiler/rollback/provenance攻击；
- focused=`19 passed`，mypy十文件clean，pylint/Black/diff通过。

本提交只建立formal gate。clean-source commit后才运行11个fresh worker并生成artifact；当前仍不claim
MR5 correctness或性能。
