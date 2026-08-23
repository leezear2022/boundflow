---
status: externally-approved
updated: 2026-08-23T08:50:00Z
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-2-external-audit-closure
stage: s01
---

# FSG4/B4-B2 B2-2 外部审计关闭记录

## Summary

- 独立外审 verdict=`APPROVE`，0 blocker、0 major、0 minor、2 info；
- 最终状态=
  `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS`；
- 只开放 B2-3 P-anchor Conv dense forward/backward correctness；
- timing、B2-4/B2-5、B4-B3 继续关闭。

## Independent Evidence

- 审计方以无 autograd 的独立 float64 闭合公式重算 5 raw、20 metrics、31,590 元素；
- 4 项输出对 TIR 最大差不超过`6.99e-07`；
- 现场 GPU runner 逐位复现 max diff=`8.642673492431641e-07`及三项 receipt hash；
- 27 个 alpha index 严格递增唯一，6 个 beta location/sign 合法；unowned native gradient 为 0；
- scheduled TIR 现场确认只有`adjoint_matmul`、`output_bias_delta`两个 alloc buffer，禁止 workspace=0；
- DLPack=`21/21`、launch=`1/1`、fallback/eager=`0/0`；
- targeted/related/full=`34/88/1448 passed`，3 skipped；静态与 DocOps 全过。

## Information Findings

1. dense B2-1 runtime 已有 dtype/device/nonfinite 校验，但缺专项测试；随 B2-3 补齐；
2. B2-2 forbidden workspace receipt 使用 scheduled-script 子串计数；B2-2 的结构性 TIR 经审计成立，
   B2-3 起增加基于 alloc buffer shape/structure 的辅助门禁。

## Claim Boundary

B2-2 只证明 S-anchor compressed alpha/beta source、compressed gradient projection、first-class
receipts 与无 native dense alpha/beta/scaled-A/relu-A workspace 的 correctness。它不证明 P-anchor、
timing、speedup、memory ratio、same-solver integration 或 ASPLOS-ready。

## Next Action

唯一下一工程动作是 B2-3：P-anchor `performance-conv-8-candidate` dense Conv transpose-contraction
forward/backward correctness；必须返回 incoming-A/native-alpha gradient，compressed beta=`[6,0]`
且 beta gradient absent。B2-3 不计时，也不得提前进入 sparse schedule search。

## Links

- audit: `gemini_doc/external_audit_b4b2_b2_2_sparse_linear_tir_2026_08_23.md`
- handoff: `gemini_doc/BOUNDFLOW_FSG4_B4B2_B2_2_EXTERNAL_AUDIT_HANDOFF_2026_08_23.md`
- implementation: `8bd1db2`
- handoff commit: `7a3f5f4`

