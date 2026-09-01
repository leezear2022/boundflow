---
status: preregistered-not-implemented
updated: 2026-08-23T02:47:06Z
type: changelog
topic: boundflow
slug: fsg4-b4b2-typed-cuda-tir-preregistration
stage: s01
---

# FSG4/B4-B2 Typed CUDA/TIR Preregistration Changelog

## Summary

- B4-B1 Round 2外审已`closed/approved`；本轮只冻结B4-B2计划，没有实现或计时TIR。
- 将“dense语义ABI”与“sparse-source fused ABI”分开，避免把dense materialization移出timed region
  后虚构融合收益。

## Changes

- 冻结S active-beta Linear与P empty-beta Conv双锚点shape、attrs、gradient ownership；
- 冻结first-class Template/Instance/Schedule/Module/Launch receipts与cache key；
- 冻结custom autograd、DLPack lifetime、current stream、alias、no-fallback合同；
- 冻结B2-0—B2-5顺序、最多12个schedule hash的bounded search；
- 冻结5-fresh correctness、6-worker AB/BA timing、memory与integrity门禁。

## Validation

- 独立核对B4-B1 v3 IR/instance/raw shapes与gradient inventory；
- 核对现有PR-12 TIR仅支持plain/non-grad/αβ-disabled，不能作为B4-B2实现；
- 现场确认Torch 2.12.1+cu132、TVM 0.23.dev0/CUDA、TVM-FFI、RTX 4060 sm_89；
- `git diff --check`与DocOps lint在提交前执行。

## Decisions

- dense ABI只证明mechanism，不进入timing；sparse-source ABI才是物理candidate；
- 主物理指标为wrapper-inclusive forward+backward，不用kernel-only替代；
- P geomean≥1.05x、95% CI lower>1.00x、worst≥0.98x且memory≤1.05x才允许外审；
- 任一S语义失败或bounded P schedule耗尽直接NO-GO，不追加候选续命。

## Follow-Ups

- 下一唯一工程动作=B2-0 lowering/receipt/identity-TIR ABI probe；
- B2-0通过前不得实现region TIR；B4-B2外审批准前不得预注册B4-B3。

## Links

- plan：`gemini_doc/BOUNDFLOW_FSG4_B4B2_TYPED_CUDA_TIR_PREREGISTRATION_PLAN_2026_08_23.md`；
- roadmap：`gemini_doc/BOUNDFLOW_FSG4_B4_CUMULATIVE_CUDA_TIR_FUSION_PLAN_2026_08_16.md`；
- B4-B1 closure：`gemini_doc/change_2026-08-23_fsg4_b4b1_round2_external_closure.md`。
