---
status: fsg0-validated
updated: 2026-08-06T09:17:57Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1
stage: s01
---

# BoundFlow Full-Stack GPU Baseline and Attribution v1 Changelog

## Summary

- 纠正NRIR49A结论传播层级：保留selected-CROWN-only NO-GO，不外推BoundFlow全栈GPU上限；
- 以official αβ-CROWN same-solver、full-stack hierarchical attribution和B0—B7累计/leave-one-out
  消融替代“寻找下一个单点winner”路线；
- 本轮只关闭FSG0合同/schema切片，没有新增speedup或production优化；当前下一步为FSG1。

## Changes

- 用DocOps创建独立plan/changelog，作为新的唯一当前路线入口；
- 冻结operator、graph/IR、Plan/Schedule、backend compile/JIT、runtime scheduling、memory allocator、
  solver/adapter九层与solver phase/resource/cache四轴schema；
- 冻结host wall、GPU union、GPU sum、critical path和exclusive critical-path分离口径；
- 冻结B0 original→B1 typed transport→B2 replacement→B3 IR/graph→B4 fusion→B5 JIT→B6 runtime→
  B7 arena/reuse累计链；
- 明确当前RVIR只是original callable exactly-once transport，PR13C也不是official host solver；
- 将combined environment或对称RPC设为same-solver headline强制前提；
- 预注册FSG0—FSG5、correctness/measurement/system gate、13文件artifact与raw replay。
- 新增typed full-stack attribution合同：九层owner、十个solver phase、host/CUDA/runtime/memory资源、
  cache状态、A0—A4 replacement成熟度、依赖边和exclusive critical-path segment；
- 新增physical feature activation ledger，区分IR/Plan/Schedule对象存在与实际驱动dispatch；
- 新增GPU interval union、critical-path closure、`<=3%` residual门禁、joint Amdahl和累计/
  leave-one-out interaction聚合；
- 新增contract-only generate/replay runner，绑定raw、summary、code revision和文件digest；同步更新摘要
  与manifest digest仍会被raw语义重算拒绝；
- 新增19项定向测试；production executor、TIR、runtime默认值均未修改。

## Validation

- 文档作用域只读审计发现12个当前指令风险点，修订清单已纳入本轮；
- 代码盘点确认当前G1仅hook `_run_selected_crown`，native queue主体仍为eager PyTorch，shared
  Task/Schedule在执行后lower，full-stack执行尚未激活；
- RVIR盘点确认replacement executor不存在，必须先完成RVIR-v3 executable payload/mutation contract；
- targeted=`19 passed in 1.11s`；
- 激活 `env.sh` 后全量=`1078 passed, 3 skipped in 402.60s`；首次未加载activation hook的尝试在
  collection阶段因`ModuleNotFoundError: tvm`停止，未产生代码失败，随后按仓库环境合同重跑通过；
- Black check通过；targeted mypy（`--follow-imports=skip`）clean；Pylint=`10.00/10`；
  `git diff --check`通过；
- FSG0状态=`VALIDATED`，仍为`performance_claimed=false`。

## Decisions

- `1.0764x`只标记selected-CROWN deletion-only ceiling，不是BoundFlow full-stack ceiling；
- 单region share仅用于工程优先级或关闭该专属实现，不再作为整条系统路线kill gate；
- 最终`1.20x queue/1.15x complete-query`只施加到累计B7 vs B0；
- diagnostic native-vs-official不同算法数据永不升级为same-solver speedup；
- 历史artifact的`gpu-winner-reselection`字段不改，文档标记为已被本路线取代。

## Follow-Ups

1. 接official control observer，生成五fresh B0 full-stack baseline；
2. 设计并实现RVIR-v3 executable payload与BoundFlow replacement correctness；
3. correctness关闭后才运行B0/B1/B2 paired timing；
4. 逐层实现B3—B7并做累计与leave-one-out消融。

## Links

- plan: [Full-stack GPU plan](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
- roadmap: [GPU compiler acceleration research](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
