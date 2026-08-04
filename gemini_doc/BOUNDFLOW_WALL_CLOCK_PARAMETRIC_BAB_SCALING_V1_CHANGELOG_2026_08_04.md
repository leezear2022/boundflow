---
status: completed
updated: 2026-08-04T13:02:00Z
type: changelog
topic: boundflow
slug: wall-clock-parametric-bab-scaling-v1
stage: s01
---

# Wall Clock Parametric BaB Scaling v1 Changelog

## Summary

- NRIR-29 已启动。目标不是继续缩短 compiler microseconds，而是把 NRIR-28 的 full-query CPU
  收益投入更多 typed BaB nodes/depth，并在固定 60 秒 query deadline 下测量 closure/coverage。

## Changes

- 新建 standalone plan/changelog，预注册 `7/depth2`、`31/depth4`、`127/depth6` 三预算、三
  workload、三 fresh repeats、交替次序与 logical-domain nesting 门禁。
- 新增 `boundflow/ir/search_scaling.py`：search budget、三 workload、三 repeat、fresh-process
  task 与 Latin-rotated schedule 均为 first-class IR，Plan/Task/Schedule 逐层 hash 绑定。
- 新增 `scripts/run_wall_clock_parametric_bab_scaling_artifact.py`：27-worker runner 导出逐 clause
  logical domains、leaf verdict、compiler template/cache/instance、raw timing 与 source/IR/log digest；
  replay 重建 experiment IR 并重算 nesting、repeat 与 closure 门禁。
- 新增两组测试，覆盖 exact schedule、预算/Task rebinding fail-closed、domain lower 篡改和严格
  verified-gain 判定；NRIR-27/28 frozen implementation 未修改。

## Validation

- 非 claim pilot 的 `31/depth4`：MNISTFC unresolved 从 `[3,7,8]` 降为 `[8]`，ResNet 全 9 clauses
  各完成 31 nodes，OVAL21 保持 `[8]`；execution 约 `2.31/13.87/0.77 s`。
- 非 claim pilot 的 `127/depth6`：MNISTFC/OVAL21 仍只剩 clause 8，分别评估 33/19 nodes；ResNet
  全 9 clauses 各完成 127 nodes，execution约 `57.44 s`，仍全部 unknown。三个 workload 均
  `completed=9,pending=[]`。
- pilot 只用于确认正式预算可执行；没有 fresh-process repeats、artifact 或 replay，不能写为论文
  结果。
- 正式 27/27 worker 均完成 `9/9` clauses、无 pending；同预算三次 semantic signature 完全一致，
  三 workload 的 `7⊂31⊂127` split-state domain nesting 成立，公共 lower 最大漂移 `0.0`。
- MNISTFC 三次 verified 都从 n7d2 的 `[0,1,2,4,5,6]` 提升为 n31d4/n127d6 的
  `[0,1,2,3,4,5,6,7]`；ResNet 三档均 `[]`，OVAL21 三档均 `[0..7]`。
- median execution（n7d2/n31d4/n127d6）：MNISTFC `2.011/2.247/2.515 s`，ResNet
  `4.801/15.121/58.566 s`，OVAL21 `1.919/2.060/2.287 s`；ResNet n127d6 p90
  `58.939 s`，三次仍全部在 query deadline 内完成。
- artifact fresh replay 与 NRIR-28 historical replay 通过；evidence hash=
  `e01d35c0afa8501f3d02ffaaa4eeaf609c444ed497c1a2d2efff4e97b3520214`；focused
  `23 passed`、全量 `823 passed, 37 skipped`，Black、Mypy、Pylint `10.00/10` 与 diff gate 通过。

## Decisions

- 正式矩阵保留 127/depth6：它接近但未超过 60 秒 execution boundary，能同时暴露 throughput 与
  deadline 风险；如正式 repeat 超时，原样保留。
- primary claim 是 verified-clause/search-coverage monotonicity，不计算跨不同预算的 speedup。
- 按预注册门禁以 `VALIDATED-REDUCED` 关闭：严格增益只来自 MNISTFC，但所有 workload 都无
  completed/verified 回退且 domain nesting 成立。
- n31d4 已达到 MNIST/OVAL 的 n127d6 closure；ResNet 即使 1143 total evaluated nodes 仍 0/9。
  因而下一步不再单轴扩树，转向 typed hard-clause escalation/stronger-bound integration。

## Follow-Ups

- 新分支冻结 `Typed Hard-Clause Escalation v1`：cheap parametric pass 后只对 unresolved clauses
  编译更强 native intermediate-refinement/branch Plan，并在同一 query deadline 内验证新增 closure、
  budget accounting 与 fallback soundness。

## Links

- plan: `gemini_doc/BOUNDFLOW_WALL_CLOCK_PARAMETRIC_BAB_SCALING_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
