---
status: validated-r3-0-compressed-alpha-v2
updated: 2026-08-25T05:35:00+08:00
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

该修正已由 clean source commit `8941e66` 生成 `r3-0-contract-v2`。replay stdout 与 summary
逐字节一致，12/12 全重签 tamper 均被 semantic replay 拒绝；alpha binding exact=
`[2,1,6,86]`，saved logical/unique=`207888/109584 B`。R3-1 admission 因此恢复，但只开放
`25/Conv_8` 单 evaluation mandatory custom-backward correctness，不计时。

正式证据见 `BOUNDFLOW_R3_0_COMPRESSED_ALPHA_V2_FORMAL_CLOSURE_2026_08_25.md`。
