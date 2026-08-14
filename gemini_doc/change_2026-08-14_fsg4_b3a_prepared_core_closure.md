# FSG4/B3-A Prepared Core 正式关闭记录

日期：2026-08-14
状态：`VALIDATED-B3-A-COUNTERS`

## 结论

source `c7851c8bae1bc943aa9e3d458e5105deafc553f1`的fresh GPU真实same-solver call通过冻结语义、显式
物理counter、provider/fallback、replay和tamper门禁。B3-A PreparedCoreTemplate/CorePlanInstance已
实际激活；没有延迟或speedup主张。

## 正式证据

- artifact：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3a-v1/`；
- manifest hash：`205978cb69238598dfcb860922e3202677d5b1775f0bd6062218f0369e982c95`；
- report hash：`89a3584dddb47d2a835bca689bdb0ba6b936d26fa5aff20a968c2323dc6cd05b`；
- event journal：5157条，SHA256=
  `3117890847ce6d503da033ff38e4cdebfc8202cc47678f0978c78ca338858848`；
- worker SHA256：`5dc804a380ae1304d784db08ebda2604e5326f5e5f7499ff99c335ae5818640a`；
- tamper report hash：`92a1900a8cdba5f42833dbd02efd2aa510d6027d58d43a1152d9a20f280d9997`。

## 物理计数

- B3-A目标变化：template compile/hit=`1/1`、module move in core=`0`、scope construction=`1`；
- 保持项：optimizer evaluation/update=`10/9`、full snapshots=`10`、forward traces=`5`、KFSB
  candidate/child=`3/3`、candidate D2H=`12`、commit/backup/copy=`12/12/12`；
- provider core/compute/update和fallback均为0；
- 观察项：tensor hash=`4913`、GPU tensor hash=`45`、typed validate=`111`、stable hash=`20`。

## 正确性与验证

- 与FSG3 v5六个冻结B2 control逐项语义一致；
- artifact replay通过；
- 六类outer-resigned counter/journal/semantic/provider/code攻击6/6拒绝；
- targeted：`34 passed in 4.60s`；
- full：`1257 passed, 3 skipped, 6 warnings in 478.68s`；
- Black clean；mypy touched runtime source clean；Pylint `10.00/10`；DocOps lint待最终提交前执行。

## 边界与下一步

本轮只有一个fresh B3-A真实call，不是正式性能样本，也没有满足完整B3计时前至少5个fresh B2/B3
correctness pairs的门禁。`diagnostic_timing_claimed=false`、`performance_claimed=false`。下一步只实现
B3-B terminal-only optimizer Schedule和terminal forward-trace handoff；B3-C、TIR/JIT/runtime/arena
继续关闭。
