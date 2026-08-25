---
status: implemented-awaiting-clean-source-formal-run
updated: 2026-08-26T21:15:00+08:00
type: changelog
topic: boundflow
slug: mr5-multi-site-timing-formal-gates
stage: s01
---

# MR5 Multi-Site Timing Formal Gates 修改记录

- 冻结worker source=`24a208140b73ed943d983ea73f2a20f842a19015`；
- 冻结6 pair/12 fresh `PM/MP`、完整outer host headline、event诊断和absolute peak memory；
- 复用MR3已批准的geomean/bootstrap/memory计算，但独立校验MR5三module prewarm与30/27 receipt；
- correctness artifact必须先replay，worker/protocol/code revision/外部repo/model/property全部hash绑定；
- 20类fully re-signed timing/semantic/module/cache/workspace/order攻击；
- focused=`12 passed`，mypy四文件分别clean，pylint/Black/diff通过。

本提交只建立clean-source formal gate。尚无6-pair正式结果，不形成性能claim。

## Formal-run前证据链 amendment

timing worker为bridge追加prewarmed cache入口后，既有correctness artifact按预期因code revision变化拒绝
replay。未放宽validator；而是将该追加接口及负向测试纳入correctness manifest，重签后重新replay通过，
summary hash保持`293c5c8b…a718`，新manifest file SHA256为`15ba6b30…f443`。正式timing只能绑定
这个可重放的新identity。
