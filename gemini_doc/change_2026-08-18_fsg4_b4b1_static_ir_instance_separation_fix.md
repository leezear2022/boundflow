---
status: fixed-b4b1-static-ir-instance-separation
updated: 2026-08-18T15:18:00+08:00
type: change
topic: boundflow
stage: s01
---

# FSG4/B4-B1 静态 IR / Instance 分离修正

## 问题

five-fresh预检查发现S-anchor IR hash稳定，但P-anchor的5次hash各不相同。原因是静态IR错误携带
单次`base_capture_hash`；P-anchor raw存在合法fresh数值变化，因此把instance digest带入Plan会破坏
静态语义身份。

## 修正

- 从`DifferentiableLowerRegionIRV1`移除单次`base_capture_hash`；
- `DifferentiableLowerRegionInstanceV1`继续绑定并校验base/reference capture及全部tensor digest；
- 新增5 fresh双锚点静态IR hash一致性测试。

## 验证与边界

S/P两个静态IR hash分别固定为`f5085dde...a08`与`f781e56c...f67`，各5次fresh均唯一；targeted=
`17 passed`。这只是Plan/Instance合同修正，不升级B4-B1关闭状态，也不开放B4-B2/TIR。
