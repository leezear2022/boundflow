---
status: active
updated: 2026-08-14T10:05:19Z
type: changelog
topic: boundflow
slug: fsg4-b3-ir-graph-plan-schedule-reuse
stage: s01
---

# FSG4 B3 IR Graph Plan Schedule Reuse Changelog

## Summary

- FSG3正式基线关闭后启动FSG4/B3；
- 仅预注册IR/graph/Plan/Schedule复用，不实现或计时candidate；
- B3拆为PreparedCoreTemplate、terminal-only optimizer Schedule和device-resident AtomicCommitPlan。

## Changes

- 首次fresh B2 diagnostic在counter gate fail closed；同source debug聚合确认唯一偏差是D2H=`6/12`。
  源码审计定位原seam只覆盖6个β `_replacement`，漏掉6个α `_project_alpha` GPU→CPU sparse-layout
  copy；已补计数点，不降低`12`门槛；
- B3-0显式counter diagnostic已实现但尚未真实运行：命名seam event journal、B2固定结构门禁、raw worker/
  semantic/environment/provider/fallback绑定、code revision/manifest与独立replay均已落地；
- 诊断不使用`sys.setprofile`，也不修改B2生产函数；instrumentation在context退出后完整恢复；
- 从FSG3 v5 profile冻结B2五区域成本与B0/B2比例；
- 将源码中的module move、10-step trace clone、重复forward、12-path GPU→CPU digest/copy与重复
  validate映射到B3子阶段；
- 冻结B0/B2/B3 36-process协议、physical activation counters、correctness/performance分类与rollback；
- 记录通用cProfile与provider callback guard冲突的失败诊断，改用显式counter。

## Validation

- B3-0 targeted=`17 passed`，mypy clean，Pylint=`10.00/10`；
- 全量回归=`1243 passed, 3 skipped`；
- FSG3 source artifact static replay与33项测试已在前一阶段通过；
- 尚无fresh B2 counter artifact、B3 candidate、correctness artifact或performance claim。

## Decisions

- 不从selected-CROWN单区或B2较慢外推全栈上限；
- 不在B3混入TIR、JIT、streams或arena；
- formal逐step trace保留给审计，production改为terminal-only必须由独立parity证明；
- artifact digest移出headline timing，但transaction fail-closed不能移除。

## Follow-Ups

1. 提交B3-0实现后，从不可变source运行fresh B2 GPU call/copy/hash counter；
2. 只有counter与预注册一致才实现B3-A PreparedCoreTemplate；
3. B3-A关闭后再进入B3-B，不并行混合变量。

## Links

- plan：
  `gemini_doc/BOUNDFLOW_FSG4_B3_IR_GRAPH_PLAN_SCHEDULE_REUSE_PLAN_2026_08_14.md`；
- roadmap：
  `gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`。
