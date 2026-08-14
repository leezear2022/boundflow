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

- B3-A已以`VALIDATED-B3-A-COUNTERS`关闭：5157条event、三项预注册物理变化、冻结语义与6/6 tamper
  通过；无timing/speedup claim；
- B3-0已以`VALIDATED-B2-COUNTERS`关闭：4625条event、固定counter全中、六个冻结B2 control语义锚定与
  6/6 outer-resigned tamper通过；
- FSG3正式基线关闭后启动FSG4/B3；
- 仅预注册IR/graph/Plan/Schedule复用，不实现或计时candidate；
- B3拆为PreparedCoreTemplate、terminal-only optimizer Schedule和device-resident AtomicCommitPlan。

## Changes

- B3-B第一版实现候选新增first-class 10/9 terminal Schedule IR、terminal-only production result和
  optimizer→backward forward-trace handoff；formal逐step trace保留；
- B3-B counter预注册仅允许full snapshots `10→0`、forward builds `5→4`，其他B3-A计数不变；
- B3-A第一版实现候选新增typed static template/dynamic instance和exact cache；只有显式cache/hash pair
  才切换prepared executor，B2默认行为不变；
- optimizer新增exact `CorePlanInstanceV1` receipt入口，拒绝跨state receipt并跳过第二次scope构造；
- diagnostic runner新增B3-A模式及预注册counter：只允许module move `1→0`、scope `2→1`、template hit
  `0→1`，其余B2固定结构不变；
- 首次fresh B2 diagnostic在counter gate fail closed；同source debug聚合确认唯一偏差是D2H=`6/12`。
  源码审计定位原seam只覆盖6个β `_replacement`，漏掉6个α `_project_alpha` GPU→CPU sparse-layout
  copy；已补计数点，不降低`12`门槛；
- 修正后provisional artifact的全部counter/replay通过，但审计发现semantic hash缺少独立真值锚点；正式版
  增加FSG3 v5六个冻结B2 control语义绑定与六类outer-resigned tamper probe，不追认provisional目录；
- `e04bdd3`正式rerun因Python 3.11/3.12重算历史profile geomean的`2.6e-9`表示差异fail closed；锚定
  改为验证FSG3 manifest/完整file digest/36-run顺序和raw B2 semantics，不在worker环境重算历史性能
  summary；
- B3-0实现阶段先完成命名seam event journal、B2固定结构门禁、raw worker/semantic/environment/provider/
  fallback绑定、code revision/manifest与独立replay；该中间“尚未运行”状态现已被正式关闭结果取代；
- 诊断不使用`sys.setprofile`，也不修改B2生产函数；instrumentation在context退出后完整恢复；
- 从FSG3 v5 profile冻结B2五区域成本与B0/B2比例；
- 将源码中的module move、10-step trace clone、重复forward、12-path GPU→CPU digest/copy与重复
  validate映射到B3子阶段；
- 冻结B0/B2/B3 36-process协议、physical activation counters、correctness/performance分类与rollback；
- 记录通用cProfile与provider callback guard冲突的失败诊断，改用显式counter。

## Validation

- B3-B实现候选CPU冻结case与负向门禁targeted=`42 passed`，mypy touched clean，Pylint=`10.00/10`；
  fresh GPU尚待执行；
- B3-A formal artifact replay、六个冻结B2 control语义与6/6 tamper通过；targeted=`34 passed`，full=
  `1257 passed, 3 skipped`；
- B3-A实现候选targeted=`31 passed`，mypy touched clean，Pylint=`10.00/10`；fresh GPU尚待执行；
- 正式artifact replay通过，targeted=`25 passed`，full=`1248 passed, 3 skipped`，tamper=`6/6`；
- B3-0 targeted=`17 passed`，mypy clean，Pylint=`10.00/10`；
- 全量回归=`1243 passed, 3 skipped`；
- FSG3 source artifact static replay与33项测试已在前一阶段通过；
- B2已有正式counter artifact；B3-A目前只有实现候选，尚无fresh B3-A correctness artifact或performance
  claim。

## Decisions

- 不从selected-CROWN单区或B2较慢外推全栈上限；
- 不在B3混入TIR、JIT、streams或arena；
- formal逐step trace保留给审计，production改为terminal-only必须由独立parity证明；
- artifact digest移出headline timing，但transaction fail-closed不能移除。

## Follow-Ups

1. 实现B3-B terminal-only optimizer Schedule与terminal forward-trace handoff；
2. 保持formal逐step trace用于审计，production path才允许消除10份snapshot；
3. B3-B关闭后再进入B3-C，不并行混合变量。

## Links

- plan：
  `gemini_doc/BOUNDFLOW_FSG4_B3_IR_GRAPH_PLAN_SCHEDULE_REUSE_PLAN_2026_08_14.md`；
- roadmap：
  `gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`。
