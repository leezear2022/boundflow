---
status: implemented-b4-c0-cumulative-core-artifact-pending-formal
updated: 2026-08-24T05:45:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4c0-cumulative-core-artifact
stage: s01
---

# FSG4/B4-C0 Cumulative Core Artifact Changelog

新增6 fresh `BC/CB/BC/CB/BC/CB` artifact/replay：root从180个raw timing groups重算每worker
median、speedup、geomean、10k bootstrap lower、worst worker及peak memory ratios，同时校验全部
semantic/receipt/identity/hash链。

冻结门禁：no-regression geomean≥`1.00x`、bootstrap lower>1、worst≥`0.98x`、memory≤`1.05x`；
research gate=`1.05x`。若native-value bridge候选不通过no-regression，状态必须为
`VALIDATED-NO-GO-B4-C0-NATIVE-VALUE-BRIDGE`，只开放provider-owned lower path rewrite。

下一步：提交clean source并运行6 fresh formal。
