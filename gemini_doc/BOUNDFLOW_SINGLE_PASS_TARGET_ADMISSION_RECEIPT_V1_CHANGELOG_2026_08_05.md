---
status: preregistered
updated: 2026-08-05T11:27:57Z
type: changelog
topic: boundflow
slug: single-pass-target-admission-receipt-v1
stage: s01
---

# BoundFlow Single-Pass Target Admission Receipt v1 Changelog

## Summary

- NRIR47 single-pass target admission receipt 已预注册；当前只有计划与证据边界，没有实现、artifact
  或新性能结果。

## Changes

- 基线冻结为 `main@ca0bcf3`、NRIR45 production 与 NRIR46 Phase 0 attribution；
- 唯一变量冻结为每 child exact target selection `2→1`，而非共享 60 个动态 target ledger；
- 冻结 typed receipt 的 graph/input/split/bounds/policy/objective/influence/ordered-target 绑定；
- 冻结 production fast admission 与 explicit full replay 的职责分离；
- 冻结 Phase A compiler/queue 与 Phase B whole-query fresh-process timing 门禁。

## Validation

- preregistration only；没有运行 candidate benchmark，也没有修改 production code；
- 路线 ceiling 来自 NRIR46：target reselection 估计 median=`1.038153 s`，约占 NRIR45 whole trace
  3.3%，不能据此预先声称 speedup。

## Decisions

- Template/Instance 已因 strict static gate 失败关闭，不在 NRIR47 中暗中恢复；
- target ledger 逐 child 动态拥有，receipt 只证明一次 exact selection 的输入/输出绑定；
- Phase A compiler ratio `<=0.85`、两条 queue ratio `<=0.97`；Phase B trace/measured ratio 均
  `<=0.98`，且所有改善必须大于 pooled MAD；
- full replay 的 selector 重算不计入 production timing，但必须进入 artifact 计数与语义重建。

## Follow-Ups

- 先实现 receipt IR、具名 compiler admission/full replay API 和负向测试，再运行 Phase A；
- Phase A 任一门禁失败即关闭，不启动 Phase B。

## Links

- plan: `gemini_doc/BOUNDFLOW_SINGLE_PASS_TARGET_ADMISSION_RECEIPT_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
