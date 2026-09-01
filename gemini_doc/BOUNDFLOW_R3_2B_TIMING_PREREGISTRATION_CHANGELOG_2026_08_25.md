---
status: preregistered-r3-2b
updated: 2026-08-25T05:48:00+08:00
type: changelog
topic: boundflow
slug: r3-2b-timing-preregistration
stage: s01
---

# R3-2B Timing 预注册修改记录

- 冻结同一P-anchor、同一10/9轨迹的native/candidate完整wrapper边界；
- 冻结host wall + CUDA边界同步、3 warmup/30 samples、5 fresh pair与median/geomean统计；
- 冻结setup SHA与hot O(1) receipt分层，禁止把formal capture同步放入timed path；
- 冻结untimed terminal correctness/counter/memory复核；
- 冻结`1.20x` geomean、`0.98x` worst和memory `<=1.0x` kill gate；
- 未运行timing，尚无performance claim。

