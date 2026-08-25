---
status: implemented-awaiting-clean-source-formal-run
updated: 2026-08-26T23:05:00+08:00
type: changelog
topic: boundflow
slug: mr6-guard-attribution-formal-gates
stage: s01
---

# MR6 Guard Attribution Formal Gates 修改记录

- 冻结worker source=`fb3c245fc8de1be08471d91b97b026ded9ce204b`；
- 实现3 triplet/9 fresh Latin顺序`PFD/FDP/DPF`与raw-first、no-resume artifact；
- 三方继续使用相同solver、module、outer host/event和完整semantic state；
- full/diagnostic guard=`360/60`，路由门禁=`full/diagnostic >=1.10x`、
  `provider/diagnostic >=0.98x`、worst `>=0.95x`；
- diagnostic始终`production_admitted=false/performance_claimed=false`；
- replay绑定MR5 timing NO-GO identity、worker/generator code revision、外部repo/model/property；
- 12类fully re-signed timing、semantic、guard、module、bridge、order、source攻击已实现；
- CPU unit gates覆盖三种guard receipt、全重签count攻击和空raw拒绝。

本提交只建立formal gate。正式9 fresh尚未运行，不形成guard dominance或性能claim。
