---
status: fixed-before-closure
updated: 2026-08-25T08:20:00+08:00
type: changelog
topic: boundflow
slug: r3-d0-correlation-tamper-hardening
stage: s01
---

# R3-D0 Correlation Tamper 加固

## 失败

首轮fully re-signed tamper中，攻击者修改一个CUDA event的`correlation_id`，同步重签worker
`event_hash`和manifest后，原ledger的数值聚合不变，replay错误接受该artifact。其余case尚未形成通过结论，
tamper整体失败。

## 修复

- 派生ledger新增`event_count`与由全部canonical event row重算的`event_payload_hash`；
- replay先逐行canonical parse，再重建该hash并与冻结ledger整对象比较；
- correlation id、marker ordinal、attribution method、phase、family、duration任一变化，即使外层全部重签，
  都必须导致ledger不一致并fail closed。

## 边界与后续

没有改变profile、Amdahl公式或10x门槛。因code revision变化，上一份formal artifact不作为最终证据；修复提交
后必须重新运行5 fresh formal与全部12类tamper。
