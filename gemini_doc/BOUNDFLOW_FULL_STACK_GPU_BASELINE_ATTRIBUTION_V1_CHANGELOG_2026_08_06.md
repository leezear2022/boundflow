---
status: fsg1-runner-ready-formal-pending
updated: 2026-08-06T13:54:29Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1
stage: s01
---

# BoundFlow Full-Stack GPU Baseline and Attribution v1 Changelog

## FSG1 Runner Preparation

- 新增official αβ-CROWN B0 control/profile typed worker合同与full-stack重建器；
- 新增独立Python 3.11/Torch 2.11 CUDA worker、交替AB/BA fresh-process编排和raw-first artifact/replay；
- compute-bound observer记录嵌套host/CUDA event、solver phase、stream、allocator counters并可逆恢复；
- fresh isolated VNNLIB副本避免`.compiled`缓存污染pair中的第二个worker；
- 真实`mnistfc:2` smoke result exact，ratio=`1.014834<=1.05`，捕获1个initial-CROWN call；
- 定向`10 passed`、全量`1089 passed, 3 skipped`，三个新文件mypy clean、Pylint 10.00/10；
  正式五轮结果尚未运行，
  `performance_claimed=false`。

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
- 新增typed full-stack attribution合同：九个功能层加一个residual哨兵、九个功能phase加
  `setup/unclassified`两个哨兵、host/CUDA/runtime/memory/IPC资源、cache状态、A0—A4 replacement
  成熟度、依赖边和exclusive critical-path segment；
- 新增physical feature activation ledger，区分IR/Plan/Schedule对象存在与实际驱动dispatch；
- 新增GPU interval union、critical-path closure、`<=3%` residual门禁、joint Amdahl和累计/
  leave-one-out interaction聚合；
- 新增contract-only generate/replay runner，绑定raw、summary、code revision和文件digest；同步更新摘要
  与manifest digest仍会被raw语义重算拒绝；
- 新增20项定向测试；production executor、TIR、runtime默认值均未修改。

## Validation

- 文档作用域只读审计发现12个当前指令风险点，修订清单已纳入本轮；
- 代码盘点确认当前G1仅hook `_run_selected_crown`，native queue主体仍为eager PyTorch，shared
  Task/Schedule在执行后lower，full-stack执行尚未激活；
- RVIR盘点确认replacement executor不存在，必须先完成RVIR-v3 executable payload/mutation contract；
- targeted=`20 passed in 1.07s`；
- 激活 `env.sh` 后全量=`1079 passed, 3 skipped in 372.54s`；首次未加载activation hook的尝试在
  collection阶段因`ModuleNotFoundError: tvm`停止，未产生代码失败，随后按仓库环境合同重跑通过；
- Black check通过；合同、runner与测试三个文件mypy（`--follow-imports=skip`）clean；Pylint=`10.00/10`；
  `git diff --check`通过；
- FSG0状态=`VALIDATED`，仍为`performance_claimed=false`。

## External Audit Response

外部独立审计结论为`APPROVE-WITH-MINOR`（0 blocker / 0 major / 3 minor）。三项均已关闭：

1. PLAN四轴枚举已逐值对齐代码规范，明确功能值与`setup/unclassified`等哨兵值；
2. 测试中的聚合对象用显式`cast`收窄，mypy从2个源文件扩大到合同、runner、测试3个文件；
3. replay实时校验`git_head`，新增同步更新manifest hash后的伪造HEAD拒绝测试。

审计原文归档于`external_audit_fsg0_full_stack_gpu_baseline_2026_08_06.md`；原文不回写，以上为
executor后续响应。

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
