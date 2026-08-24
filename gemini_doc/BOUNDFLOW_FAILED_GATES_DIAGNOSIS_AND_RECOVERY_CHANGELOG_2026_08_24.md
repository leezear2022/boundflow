---
status: documented-pending-external-advice
updated: 2026-08-24T10:39:09+08:00
type: changelog
topic: boundflow
slug: failed-gates-diagnosis-and-recovery
stage: s01
---

# BoundFlow 失败门禁诊断与恢复计划变更记录

## Summary

- 新增全量失败门禁诊断，明确区分正式 NO-GO、VALIDATED-REDUCED、未运行和已通过；
- 复盘 selected-CROWN、B2/B3、B4-A、B4-B2 v1/v2、B4-C0/C1/C2 和 CIBC-IBP；
- 承认并记录原路线中的三类结构问题：过早缩到单区域、v1 未做真实融合/调优、CROWN 所有权与
  autograd lifetime 边界错误；
- 用 TVM/CUDA/PyTorch/αβ-CROWN 一手资料重写 R0—R5 恢复路线；
- 新增可直接交给外部大模型的审计 prompt。

## Changes

- `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`
  - 加入全部关键数字、门槛、分类和 root cause；
  - 量化 B3 回到 B0 parity 仍需 `1.09890x`，达到 final query gate 仍需 `1.26373x`；
  - 把 CIBC 下一动作限定为 candidate-only optimized-graph attribution；
  - 把 α-CROWN 恢复限定为 structured owner + custom backward，不复活 dense-retention C2；
  - 明确 steady-state/cold、operator/graph/query、local/cumulative claim 边界。
- `BOUNDFLOW_FAILED_GATES_EXTERNAL_ADVISOR_PROMPT_2026_08_24.md`
  - 要求外部模型独立核 raw、质疑分类、推导 Amdahl 可行性并给出带 kill condition 的排序；
  - 禁止只给无法映射到代码和门禁的通用 GPU 建议。
- `gemini_doc/README.md`、`asplos_execution_memo_v1_0.md`、`current_status_after_pr13.md`、
  `asplos_claims_map.md`
  - 增加 CIBC 外审正式关闭和本文路线入口；
  - 将“下一步”统一为 R0 + R1 预注册，不提前实现 TIR/CROWN 改动。
- DocOps exchange `cibc-ibp-horizontal-20260824`
  - Round 1 外审已批准，executor 已执行 close；
  - closure、audit 与 full report 纳入版本控制。

## Validation

- 已逐项读取 9 份 formal `summary.json` 与外审报告，headline 数字一致；
- 计划、变更记录、prompt 分别为 446/70/118 行，内部证据路径存在性检查通过；
- `git diff --check`：PASS；
- `dol exchange status cibc-ibp-horizontal-20260824`：`closed/approved`；
- `dol exchange validate cibc-ibp-horizontal-20260824`：PASS；
- `dol lint --soft`：PASS；
- 本轮只改文档/DocOps 审计状态，不重跑性能，不产生新性能 claim。

## Decisions

- 立即动作不是“继续 B4-C2”，而是先关闭审计卫生并预注册 CIBC-G1 candidate-only attribution；
- CIBC-IBP `2.45631x` 保持 reduced claim，不外推 auto_LiRPA/α-CROWN/BaB；
- B4-C2 只关闭当前 dense-retention 集成，不否定 structured custom-backward 重新设计；
- B5/B6/B7/complete solve 标为未运行，不标为失败；
- `docs/CIBC_for_DAC.pdf` 是用户提供的本机未跟踪资料，本次不提交。

## Follow-Ups

1. R0 修复 3 条新 mypy arg-type 和新增 pylint scope，并补 tolerance/steady-state 披露；
2. 预注册 CIBC-G1 NVTX/CUPTI/CUDA Graph attribution protocol；
3. 外部模型评审本文，先处理 blocker/major，再开任何性能实现；
4. 根据 R1 measured critical-path share 选择 R2-A/B/C/D；
5. 若恢复 α-CROWN，先单独评审 custom-backward live-set 契约。

## Links

- plan: `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`
- advisor prompt: `BOUNDFLOW_FAILED_GATES_EXTERNAL_ADVISOR_PROMPT_2026_08_24.md`
- external audit: `external_audit_cibc_ibp_horizontal_2026_08_24.md`
- roadmap: `boundflow_asplos_master_plan_2026_07_12.md`
