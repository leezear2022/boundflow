---
status: completed
updated: 2026-08-03T18:45:01Z
type: plan
topic: boundflow
slug: production-schedule-memory-p0
stage: s01
---

# Production Schedule IR + Memory P0 门禁计划

## Goal

- 回答 RVIR 关闭后能否直接把 **Production Schedule IR + Memory Feasibility** 作为下一条
  ASPLOS 主线。
- 用机器可重放门禁区分“已有 Schedule IR 对象”与“真实 verifier 主计算已由 Schedule IR
  控制并产生预算相关决策”。

## Scope

- 冻结输入：IR-5 residual-final-v3 与 RVIR CPU correctness v2；先验证各自 manifest digest。
- 当前编译器结构复算：2 个 residual workload × 4 个 backend，共 8 个 Plan/Task/Schedule
  路径；比较 64 MiB 与 512 MiB 下的 PlanInstance 决策。
- 真实路径复算：逐条重编译 VNN-COMP ResNet 的 51 个 activation call，并核对五层 IR hash。
- 本门禁不重新计时、不执行 CUDA、不产生新的 correctness、latency、memory 或 speedup claim。

## Tasks

- [x] 验证两个源 artifact 的 14 个 payload digest。
- [x] 审计 residual 路径的 Bound-op coverage、arena、batch loop、launch 与 free。
- [x] 区分 PlanInstance identity 变化与 region/backend/batch/storage/state 决策变化。
- [x] 审计 storage candidate、MaterializeAction 和预算边界 fail-closed。
- [x] 复算 51 个真实 ResNet external-call 的 Bound/Plan/Task/Schedule hash 与 action profile。
- [x] 生成 deterministic artifact、semantic replay 与同步篡改单测。

## Gate Result

结论：`NO_GO`，不得直接进入 Schedule-memory headline 实现。

| 门禁 | 结果 | 证据 |
|---|---:|---|
| residual 完整 Bound graph 由 Schedule 覆盖 | PASS | 8/8 case；每个 10 个 Bound op 全覆盖 |
| arena lifecycle 显式 | PASS | 均有 check/allocate/batch/launch/emit/free |
| 显式 materialization transition 被 production case 覆盖 | FAIL | 8/8 均为 0 `materialize` action |
| storage 有两个以上合法候选 | FAIL | 8/8 template 均只有 1 个 storage candidate |
| 真实 ResNet 为 native multi-region Bound IR | FAIL | 51/51 均为单个 `EXTERNAL_VERIFIER_CALL` |
| 64/512 MiB 改变实际计划决策 | FAIL | Plan hash 不同，但 8/8 decision signature 相同 |
| 冻结 multi-budget switch / 双 workload Pareto | FAIL | residual-final-v3 两项门禁均失败 |
| baseline OOM rescue | NOT DEMONSTRATED | 当前没有对应冻结 artifact |

不同预算会进入 PlanInstance identity，因此 hash 不同；但当前 region、representation、backend、
batch、storage 与 state decisions 完全相同，不能据此声称 memory-aware scheduling 已生效。

## Validation

- `pytest -q tests/test_production_schedule_coverage.py` → `2 passed`
- `pytest -q tests` → `462 passed, 37 skipped`
- Black check、Mypy（2 files）、Pylint（10.00/10）与 `git diff --check` 均通过。
- 生成与 replay 命令均输出同一个 `NO_GO` 和 8 个 failed gate ID。
- artifact：`artifacts/schedule-p0/production-schedule-memory-p0-20260804/`
- `coverage.json` SHA256：`27bd58877db4ff8d62f6767ad3719de19de95f9b5b86dd9242ea3bb836808fca`。

## Rollback

- 本工作不改变 runtime/backend 行为。若审计 schema 不再需要，可删除新增 audit module、runner、
  tests 与 artifact；IR-5/RVIR 冻结工件未被修改。

## Links

- changelog: `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_CHANGELOG_2026_08_04.md`
- artifact: `artifacts/schedule-p0/production-schedule-memory-p0-20260804/`
- next branch: `feat/native-real-network-bound-ir-v1`
