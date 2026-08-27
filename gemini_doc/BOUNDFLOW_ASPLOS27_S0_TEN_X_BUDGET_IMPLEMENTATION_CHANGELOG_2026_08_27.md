# BoundFlow ASPLOS’27 S0 10×预算第一批修改记录

date: 2026-08-27
status: internally-validated-s0-not-admitted
external-audit: deferred-by-user
performance-claimed: false

## 1. 修改摘要

本轮把v6计划从用户审阅稿切到`execution-active-s0`，实现typed 10×预算、claim-scope隔离、事务语义覆盖
门禁、历史direct observation ledger及可语义重放artifact。没有修改production solver、TVM/TIR、RVIR或
第三方代码，没有新性能测量。

## 2. 文件

- `boundflow/runtime/asplos27_tenx_budget.py`
- `scripts/run_asplos27_s0_tenx_budget_artifact.py`
- `tests/test_asplos27_tenx_budget.py`
- `tests/test_asplos27_s0_tenx_artifact.py`
- `artifacts/asplos27-s0-tenx-budget/fsg1-diagnostic-and-history-v1/`
- 本计划、rapid-review两页草稿、CIBC changes note与本修改记录
- v6主计划、`gemini_doc/README.md`及DocOps状态

## 3. 独立于旧summary的派生结论

- FSG1 10个profile的transaction topology context全部闭合；
- ResNet2B 5个run的inter-call host mechanism unresolved=`30.62%—31.51%`，MNISTFC为0；
- existing operator target `12.795107698×`时projection最大`2.318856753×`；
- operator无限加速时系统上限最大`2.610777967×`；
- 10× feasibility=`0/10`；S1性能门禁保持关闭；
- 四条历史direct ratio只列账、不聚合。

## 4. Claim边界

这些数是冻结FSG1 fixed-16-iteration prefix的S0诊断，不是complete query、TTV或新BoundFlow candidate结果。
`performance_claimed=false`保持。结论只证伪“operator-only可达10×”，不证伪后续coarse solver/runtime编译。

## 5. 验证

已完成：

- 新增预算与artifact专项：`12 passed`；
- full-stack/FSG1/S0关联链：`42 passed`；
- 全量：`1844 passed, 3 skipped, 6 warnings`，skip均为既有环境/重复编译边界；
- Black：4个新增Python文件通过；
- mypy：2个新增source文件clean；
- pylint：2个新增source文件`10.00/10`；
- artifact generate/replay：退出码0，summary hash=
  `386d2aeb686c5c609af9c80526490c184d3d634d584069f9f3c992f886056eb9`；
- fully re-signed semantic tamper专项通过；
- `git diff --check`与DocOps lint在交接前执行。

外部审计按用户指令暂缓；下一批先做explicit transaction marker，不做S1/TIR越序实现。
