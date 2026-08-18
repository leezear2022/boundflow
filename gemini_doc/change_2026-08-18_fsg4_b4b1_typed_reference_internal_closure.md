---
status: validated-b4b1-typed-pytorch-reference-pending-external-audit
updated: 2026-08-18T16:18:00+08:00
type: change
topic: boundflow
stage: s01
---

# FSG4/B4-B1 typed pure-PyTorch reference 内部关闭

## 判定

`VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`。

这关闭typed IR、纯PyTorch forward/local VJP与冻结双锚点gradient parity；不开放B4-B2/TIR，
不产生性能、显存、whole-core/query或ASPLOS-ready claim。

## 正式证据

- v2 artifact：`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v2`；
- artifact source=`d9164b880d8b9c7a19a02d421bc3204262d8fbc4`；
- manifest=`14923b0398f95b4adc0f95980d990a8f03260c173f0fd040f3ad5767e63f3167`；
- summary=`becd8ae57536bc678392748bee5568d8b18922526df02da1238720b44045d744`；
- 5 fresh/10 captures、60 metrics/196,380 elements；
- max diff=`6.109476089477539e-07`、allclose/sign exact；
- S native β gradient=5/5、P incoming-A gradient=5/5；
- S/P static IR hashes=`f5085dde...a08`/`f781e56c...f67`。

## 完整性

- integrity report=`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v2-integrity-report.json`；
- report source=`255d5fb2211faf5983bb9006ce3d2ef75c4f1c0b`；
- report hash=`6a3192f6a6ab2e14ab012bfedd3cc4251de416739ec6850290c98cd3aa399313`；
- incoming-bias与output-adjoint两类all-run改写均同步重签capture、source summary/manifest和
  derived protocol，`2/2`由数值reference拒绝；
- report绑定probe、完整reference code revision和source artifact manifest。

## v1纠正

首次full=`1402 passed, 3 skipped, 1 failed`，唯一失败为v1 exact record replay受前序测试改变
PyTorch线程数影响。v2冻结threads=1、deterministic algorithms、float32 precision与MKLDNN，并
恢复调用者状态；threads=1/4/8入口records一致。v1保留为superseded/fail-closed历史证据。

## 验证

- B3/B4 related：`131 passed`；
- full：`1405 passed, 3 skipped, 6 warnings`；
- Black `--fast --check`：PASS；
- scoped Mypy：PASS；
- scoped Pylint：`10.00/10`；
- `git diff --check`与`dol lint --soft`：PASS。

## 下一步

只提交独立外审。外审批准后才允许另行预注册B4-B2 CUDA/TIR；不得直接实现或计时。
