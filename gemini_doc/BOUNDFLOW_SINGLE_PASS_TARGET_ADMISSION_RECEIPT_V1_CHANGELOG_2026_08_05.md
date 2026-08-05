---
status: validated-no-go
updated: 2026-08-05T12:32:05Z
type: changelog
topic: boundflow
slug: single-pass-target-admission-receipt-v1
stage: s01
---

# BoundFlow Single-Pass Target Admission Receipt v1 Changelog

## Summary

- NRIR47 typed single-pass target admission receipt、prepared binding、显式 production route、full replay
  与 formal artifact 已完成；correctness/ownership 通过，但 compiler 和两条 queue timing 门禁失败，
  因此以 `VALIDATED-NO-GO` 关闭且不启动 Phase B。

## Changes

- 基线冻结为 `main@ca0bcf3`、NRIR45 production 与 NRIR46 Phase 0 attribution；
- 唯一变量冻结为每 child exact target selection `2→1`，而非共享 60 个动态 target ledger；
- 冻结 typed receipt 的 graph/input/split/bounds/policy/objective/influence/ordered-target 绑定；
- 冻结 production fast admission 与 explicit full replay 的职责分离；
- 冻结 Phase A compiler/queue 与 Phase B whole-query fresh-process timing 门禁。
- 新增 typed receipt/Task/Schedule IR、additive single-pass compiler/prepared Program 与 production
  candidate route；旧核心 compiler 文件恢复原样以保持 NRIR33/34 frozen revision；
- 新增 6 条 contract test 与 three-repeat formal generate/replay/tamper runner；
- artifact：`artifacts/single-pass-target-admission/`
  `vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1/`。

## Validation

- candidate 每条 queue compile selector/reselection=`30/0`、runtime selector=`30`、receipt/full
  replay=`31/31`；correctness/ownership exact，186 份 typed receipt replay；
- compiler ratio=`0.936003 > 0.85`；clauses 2/3 queue ratio=
  `1.011205/1.019338 > 0.97`，Phase A timing gate 失败；
- formal hash=`a7561e5187a6e396905d261e739280e39f2c3480e83ba2af0fbe6e3b1ec042ce`；
  replay 与同步外层重哈希 tamper 拒绝通过；
- targeted `55 passed`，全量 `992 passed, 37 skipped`，Black、mypy、Pylint `10.00/10` 通过。

## Decisions

- Template/Instance 已因 strict static gate 失败关闭，不在 NRIR47 中暗中恢复；
- target ledger 逐 child 动态拥有，receipt 只证明一次 exact selection 的输入/输出绑定；
- Phase A compiler ratio `<=0.85`、两条 queue ratio `<=0.97`；Phase B trace/measured ratio 均
  `<=0.98`，且所有改善必须大于 pooled MAD；
- full replay 的 selector 重算不计入 production timing，但必须进入 artifact 计数与语义重建。
- 计数口径区分 compile selector、compile reselection、runtime semantic selector 与 replay selector；
  NRIR47 只消除 compile reselection，不删除既有 runtime semantic selection；每条 queue 另把一次 root
  source admission 显式计入，因此 receipt/full replay=`31/31`。

## Follow-Ups

- Phase B 已 gated off；candidate 不默认启用，也不形成 speedup claim；
- 下一单变量先做剩余 top-2 production execution math/queue phase attribution，识别约 20 秒路径中的
  dominant execution cost，再决定 stronger bound、kernel/backend 或 queue fusion 路线。

## Links

- plan: `gemini_doc/BOUNDFLOW_SINGLE_PASS_TARGET_ADMISSION_RECEIPT_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
