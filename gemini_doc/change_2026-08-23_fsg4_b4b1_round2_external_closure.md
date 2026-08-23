# FSG4/B4-B1 Round 2 外审关闭记录

最终状态：`EXTERNALLY-APPROVED-VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE`  
exchange：`fsg4-b4b1-typed-reference-20260818`，Round 2，`closed/approved`

## 结论

Round 1 的两个 major finding 已由独立外审关闭，Round 2 AC1—AC6 全部 PASS，findings=0。
Executor 已执行 `dol exchange close`。该状态只关闭两个冻结 production anchors 的 typed
pure-PyTorch correctness/gradient reference；不支持 B4-B2/TIR、性能、显存、whole-core/query
speedup 或 ASPLOS-ready。

## F1/F2 独立关闭证据

- F1：审计方对 S/P 各构造 10 类 receipt inventory/target negative cases，`20/20 rejected`；
- F2：deterministic debug mode 0/1/2 × 正常/异常退出，`6/6` 精确恢复 threads、warn/debug
  mode、float32 precision 与 MKLDNN；
- v1/v2 均由新 protocol fail closed，v3 provenance 与 root replay 通过；
- 正式 all-run negative integrity cases `2/2 rejected`；
- 独立 raw 重算为 5 runs、10 captures、60 metrics、196,380 elements、max abs diff=
  `6.109476089477539e-07`、allclose/sign exact；S native α/β 与 P native α/incoming-A 各 5/5，
  P empty beta gradient 5/5 缺席。

## 回归与环境边界

- executor RTX 4060：targeted=`32 passed`、related=`140 passed`、full=
  `1414 passed, 3 skipped, 6 warnings`；
- auditor CUDA-unavailable：targeted=`32 passed`、related=`128 passed, 12 skipped`、full=
  `1366 passed, 51 skipped, 7 warnings`；
- 51 skip = 48 CUDA 条件项 + 3 非 CUDA 项；与 executor 的 `1414+3` 集合边界一致；
- Black、Mypy、Pylint 10.00、diff、exchange validate 与 DocOps lint 均通过。

## 下一步

只开放“另行预注册 B4-B2 typed CUDA/TIR reference candidate”。必须先冻结 production shape、
IR→TIR lowering、forward/VJP parity、stream/alias/memory ownership、five-fresh、完整性与 kill gate；
不得直接实现或计时 TIR，也不得把 B4-B1 correctness 自动升级为性能 claim。
