---
status: implemented-preflight-passed
updated: 2026-08-26T20:55:00+08:00
type: changelog
topic: boundflow
slug: mr5-multi-site-timing-worker
stage: s01
---

# MR5 Multi-Site Timing Worker 修改记录

- 在正式region外编译并逐site执行一次dummy forward/backward，outer内cache=`0 miss/10 hit`；
- 复用MR3已批准的完整outer host/CUDA-event/memory bracket，不把kernel-only时间升级为headline；
- outer内只保留三site route与O(1) receipt，无formal trajectory、hash、CPU copy或owner snapshot；
- candidate累计`30/27`，provider/candidate均保持真实solver verdict与final state；
- bridge prewarmed receipt增加专项accept/reject测试。

一组非正式fresh预检：provider/candidate host=`106.815/120.839 ms`，ratio=`0.883947x`；CUDA-event
ratio=`0.883922x`，方向一致；absolute peak allocated candidate未增加，reserved exact。该单pair暗示当前
三独立site runtime可能NO-GO，但不是正式结论，不得据此停止冻结6-pair实验或改门槛。
