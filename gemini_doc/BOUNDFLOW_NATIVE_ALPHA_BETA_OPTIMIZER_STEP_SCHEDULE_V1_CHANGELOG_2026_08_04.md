---
status: completed
updated: 2026-08-04T08:10:00+08:00
type: changelog
topic: boundflow
slug: BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZER_STEP_SCHEDULE_V1
stage: s01
---

# Native Alpha/Beta Optimizer-Step Schedule v1 Changelog

## Summary

- NRIR-11 在 NRIR-10 merge 后启动；目标是把 fixed-step optimizer control 从 runtime-owned opaque
  loop 提升为 typed Plan/Task/Schedule program。

## Changes

- 创建计划并冻结 correctness-only、no-full-BaB/no-performance 边界。
- 新增 `boundflow/ir/optimizer.py`：typed optimizer Plan/Task/Schedule IR、stable hash、固定步数
  lower 与 cross-layer verifier。
- 新增 schedule-driven runtime：显式 evaluate/reduce/backward/Adam/project/select-best，绑定 NRIR-10
  source compiler stack、scope、policy、warm-start 与初始 state。
- execution trace 记录 action 输入/输出 hash、evaluation bound/metric/state、alpha/beta gradient、
  projection 和 per-domain best iteration；完整 hash chain 与 Task/Schedule 一一对应。
- toy 测试证明与 legacy optimizer 逐张量一致，selected state 再经 NRIR-10 frozen native
  Bound/Plan/Task/Schedule 执行一致；order/linkage/hash/scope/warm-start tamper fail closed。
- 新增 fixed ResNet generate/replay artifact、manifest 与同步重哈希后的 Plan/Task/Schedule/trace/
  claim tamper tests。

## Validation

- fixed ResNet 1-step optimizer：8 Task/Action、2 evaluated iterations、1 backward/Adam/project，best
  iteration=`1`；alpha/beta gradient L1=`169.23175295069814/12.862210273742676`。
- Schedule 对 legacy optimizer、selected-state native compiler 的 lower/upper max diff 均为 `0.0`；
  selected alpha/beta state hash 与 legacy 相同。
- artifact generate/replay：
  `31261b63d80a7b11dc14484ddab2fe37bbafcc86866aaeaaa53d6af70ea40a19`。
- 聚焦：`35 passed`；全量：`612 passed, 37 skipped, 7 warnings in 151.79s`。
- Black check、Mypy 5 files、Pylint `10.00/10`、`git diff --check` 全过。

## Follow-Ups

- 下一分支把 optimizer Plan/Task/Schedule 接回 NRIR-9 queue 的每个 node evaluation；parent state
  只能作为 monotonic-refinement initialization，selected state 必须逐节点经 native compiler执行。
- dynamic early stop、complete BaB termination/property verdict、CUDA 与性能证据继续不声明。
