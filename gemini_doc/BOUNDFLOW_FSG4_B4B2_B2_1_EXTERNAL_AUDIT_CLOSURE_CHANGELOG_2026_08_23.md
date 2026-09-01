---
status: externally-approved
updated: 2026-08-23T05:45:00Z
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-1-external-audit-closure
stage: s01
---

# FSG4/B4-B2 B2-1 外部审计关闭记录

## Summary

- 独立外审 verdict=`APPROVE`，0 blocker、0 major；
- 最终状态=
  `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS`；
- 按预注册 DAG 只开放 B2-2 S-anchor sparse-source fused forward/backward；
- timing、P-anchor、B2-4/B2-5、B4-B3 继续关闭。

## Independent Evidence

- 审计方未复用仓库 reference，以独立 float64 表达式重算 5 raw、20 metrics、36,750 元素；
- 对 float64 ground truth 最大差=`6.988e-07`，全部 allclose/sign exact/finite；
- 现场 GPU runner 逐位复现执行方 max diff=`8.642673492431641e-07` 与三项 receipt hash；
- forward/backward launch=`1/1`，fallback/eager=`0/0`，cache=`miss,hit,hit,hit,hit`；
- targeted=`23 passed`、B4-B related=`77 passed`、full=`1437 passed, 3 skipped`；
- Black、Mypy、Pylint 10.00、TVM rebuild 与 DocOps lint 全过；
- B2-0 的真实 fallback counter 与 rebuild 措辞两个 minor 均由外审确认关闭。

## Finding Disposition

- minor：内部 changelog/handoff 记为 B4-B related=`76 passed`，现场为`77 passed`；原因是
  `eb74e45`同时新增 identity fallback counter 测试。当前权威文档已更正，结论方向不变；
- info：5 raw 的 S capture 数值相同、B2-5 需冻结 raw stdout；保持为 B2-5 门禁；
- info：dense 侧缺显式 dtype/device/nonfinite 专项负例；纳入 B2-2 测试，不削弱已存在的 runtime
  fail-closed 校验；
- info：TIR `max(u-l, eps)` 与 reference `clamp_min(eps)` 等价；无需修改。

## Claim Boundary

B2-1 只证明 S-anchor dense Linear/Gemm forward/output-bias 与 native alpha/beta 一阶 VJP 的
CUDA/TIR correctness、first-class receipts、zero-copy/current-stream/cache/fail-closed 机制。它不证明
sparse-source fusion、workspace elimination、timing、speedup、memory、P-anchor、same-solver integration
或 ASPLOS-ready。

## Next Action

唯一下一工程动作是 B2-2：将 27 项 compressed alpha 与每 domain 单项 beta location/sign 直接纳入
S-anchor TIR，返回 compressed alpha/beta gradient 并投影回 native receipt；禁止 global dense
alpha/beta/scaled-A workspace。B2-2 失败即
`VALIDATED-NO-GO-B4-B2-SEMANTICS`，不得转向 P 性能规避。

> **2026-08-23 后续状态**：上述B2-2实现动作已完成内部correctness门禁，当前下一步
> 只为B2-2外审；B2-3/P-anchor仍未开放。

## Links

- audit: `gemini_doc/external_audit_b4b2_b2_1_dense_linear_tir_2026_08_23.md`
- handoff: `gemini_doc/BOUNDFLOW_FSG4_B4B2_B2_1_EXTERNAL_AUDIT_HANDOFF_2026_08_23.md`
- implementation: `eb74e45`
- handoff commit: `2da99da`
