---
status: implemented-pending-v2-formal
updated: 2026-08-25T05:05:00+08:00
type: changelog
topic: boundflow
slug: r3-0-compressed-alpha-fix
stage: s01
---

# BoundFlow R3-0 Compressed Alpha Fixture 修正记录

## Finding

R3-0 v1 的验证器、closure/liveness 和篡改门禁成立，但 formal fixture 的 `alpha` binding/saved ledger
错误使用了 dense native shape，而 R3 设计和 production P-anchor 的真实叶子是
`[2,1,6,86]` compressed α。v1 因此只能证明通用合同机制，不能单独授权 R3-1 production-shaped
custom backward。

## Fix

- 将 frozen P-anchor instance 的 α binding 从 dense shape 改为 production compressed
  `[2,1,6,86]`；
- saved α bytes 从 `98,304` 修正为 `2,064`；
- saved logical/unique storage 重新冻结为 `207,888 / 109,584 B`；
- 新增 shape 与两项 derived byte 的直接测试；
- 不修改 closure、scratch、claim 或 tamper 门槛。

## Boundary

- v1 artifact 保留为历史机制证据，但 `r3_1_open=true` 被 v2 取代；
- 当前状态=`IMPLEMENTED-R3-0-COMPRESSED-ALPHA-FIX-PENDING-V2-FORMAL`；
- v2 clean-source replay/tamper 通过前，R3-1 重新关闭；
- 本修正没有 custom backward、correctness、memory 或 performance claim。

## Next

提交修正源码，从 clean commit 生成 `r3-0-contract-v2`，重跑 replay 与 12 类全重签 tamper，然后
再决定是否恢复 R3-1 admission。
