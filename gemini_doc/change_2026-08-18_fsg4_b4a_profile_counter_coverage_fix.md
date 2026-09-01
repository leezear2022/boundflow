---
status: implemented-pending-clean-source-formal-run-v2
updated: 2026-08-18T18:15:00+08:00
type: change
topic: boundflow
slug: fsg4-b4a-profile-counter-coverage-fix
stage: s01
---

# FSG4/B4-A Profile Counter 覆盖修复

## 失败事实

source=`292a035`启动的正式v1 artifact在前三个worker完成后，于worker 3
`block-00-pos-03-B4-A-profile` fail closed。preflight已通过，失败发生在typed activation写raw前：旧门禁
首先报告`forward_trace_build_count expected=4 observed=3`。增加逐字段诊断后，独立复跑还显示
optimizer bound/trace/evaluation/update分别被记为`0/0/0/0`。

该失败artifact保留在`artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v1`；它只有3个complete worker和
worker 3的失败metadata，不得resume到新source，不得进入任何ratio或performance分类。

本机失败证据SHA256：protocol=`8194708c...474cc8`、failed worker=`2e78ce23...2a5661`，三个complete
worker依次为`e70375d7...55564`、`392761b0...6dcb`、`dafbd356...9032a`。该目录按仓库策略不提交；这些
digest只用于追踪失败诊断，不构成可移植正式artifact。

## 根因

B4-A terminal optimizer实现位于`fsg4_b4a_terminal_lower_adjoint_handoff.py`，模块持有自己的
`_forward_ibp_trace_mlp`、两类CROWN evaluation与terminal optimizer函数引用。B3显式instrumentation只
patch了B3 terminal/optimizer模块：KFSB的3次forward仍被记录，但B4-A自己的1次forward和10/9 optimizer
结构没有进入recorder。因此`3/0/0/0`是观测覆盖缺口，不是物理执行结构。

## 修复

- `_instrument_b2`同时patch B4-A的forward build；
- 9次update CROWN与第10次terminal lower-adjoint CROWN共同计为10次bound evaluation；
- B4-A terminal wrapper从typed result记录trace/evaluation/update=`1/10/9`；
- outer activation保存完整profile counter payload与hash，formal replay要求B3/B4-A都满足原冻结的B3-C
  physical counter合同；没有兼容投影或计数豁免；
- outer-resigned tamper新增B4-A profile counter同步重签攻击，总数增至12类。

## 独立live诊断

修复后独立`B4-A-profile` worker退出码0，直接从worker JSON读取：

- forward build=`4`；
- optimizer bound evaluation=`10`；
- optimizer trace/evaluation/update=`1/10/9`；
- handoff/rerun=`1/0`；
- provider/fallback=`0/0`。

## 边界与下一步

验证：固定related 9文件=`62 passed`；全量=`1352 passed, 3 skipped`；Black、Mypy、Pylint
`10.00/10`、`git diff --check`均通过。

这只是measurement instrumentation修复，不是性能结果。v1明确作废为不完整诊断证据；下一唯一动作是
完成回归、提交clean source，然后在新目录从position 0生成v2。正式replay、12/12 tamper与外审前继续
保持`performance_claimed=false`，B4-B/TIR关闭。
