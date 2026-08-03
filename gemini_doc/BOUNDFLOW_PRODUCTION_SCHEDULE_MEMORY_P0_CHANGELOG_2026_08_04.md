---
status: completed
updated: 2026-08-03T18:45:01Z
type: changelog
topic: boundflow
slug: production-schedule-memory-p0
stage: s01
---

# Production Schedule IR + Memory P0 变更记录

## Summary

- 新增 Production Schedule IR + Memory P0 可重放审计，最终判定 `NO_GO`。
- 判定不否认 Schedule IR 的已有实现；它证明当前真实 ResNet 主计算仍是外部黑盒，现有
  residual template 也没有 materialization/storage 选择或预算触发的结构性计划切换。

## Changes

- `boundflow/planner/production_schedule_coverage.py`：源 artifact digest 验证、8 个 residual
  结构复算、51 个 ResNet IR hash 重编译、预算 decision signature 与统一门禁。
- `scripts/run_production_schedule_coverage_audit.py`：deterministic generate/replay CLI。
- `tests/test_production_schedule_coverage.py`：正向判定与“payload + manifest digest 同步篡改”
  的 semantic replay 拒绝测试。
- `artifacts/schedule-p0/production-schedule-memory-p0-20260804/`：manifest + coverage payload。
- 更新 ASPLOS memo、master plan、current status、claims map、README 与 change log。

## Validation

- 专属测试：`2 passed`。
- artifact generation：`NO_GO`；semantic replay：同一 `NO_GO` 与 failed-gate 列表。
- 51/51 VNN-COMP ResNet activation rows 的五层 IR hash 与冻结记录一致。
- 8/8 residual cases 在峰值减 1 byte 的预算下均以 `memory_budget_exceeded` fail closed。
- 全量回归：`462 passed, 37 skipped`；37 个 skip 为既有 CUDA/环境边界。
- Black check、Mypy（2 files）、Pylint（10.00/10）与 `git diff --check` 均通过。
- DocOps `va` 与 soft lint 在交付前记录。

## Decisions

- 不启动 `feat/production-schedule-memory-v1`，不旋转新的 IR-5 timing split。
- 下一分支改为 `feat/native-real-network-bound-ir-v1`：先让一个冻结真实 residual network
  的主计算从 external-call 变为 native multi-region Bound IR。
- 只有真实路径同时出现显式 Schedule ownership、两个以上合法 memory plan、预算决策切换，
  并有 OOM rescue 或可重现 Pareto，才允许重开 Schedule-memory 性能路线。

## Follow-Ups

- 冻结一个真实 ResNet initial-CROWN query 与原始 ONNX/VNNLIB digest。
- 补齐 ONNX/Primal→native Bound IR 的 residual/conv/relu/flatten/linear lowering，不用
  `EXTERNAL_VERIFIER_CALL` 包装主图。
- 用现有 external-semantics result 作相同浮点语义 oracle；先 correctness，再设计 storage/
  batching 候选，最后才进行 GPU 性能门禁。

## Links

- plan: `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_PLAN_2026_08_04.md`
- artifact: `artifacts/schedule-p0/production-schedule-memory-p0-20260804/`
- prior closure: `gemini_doc/real_verifier_ir_integration_closure_2026_08_03.md`
