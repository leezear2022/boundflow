# RVIR-v4 V4-2E Atomic Copy-Out 修改记录

日期：2026-08-13

## 目标

将V4-2D terminal native α/β先投影到私有production candidate，验证12个mutable path、final lower、
read-only/history/layout与真实post-state后，再一次性提交到provider-owned live targets；任何失败必须保持
全部live target不变。

## 当前实现

- dense α按冻结sparse coordinates写回lower polarity/start-spec，upper plane原样copy-through；
- dense β按冻结SparseBeta location压缩回6个value path；
- 12个mutable path完整性、schema、finite、`2e-4`数值与sign逐项验证；
- candidate是新immutable snapshot，read-only/history/policy从pre继承且逐hash不变；
- commit前验证全部live target仍与pre一致；写入异常或post-check失败会从12份backup回滚；
- receipt绑定pre/candidate/expected post、12 path和final lower，callback/fallback=`0/0`。

## Capture-Ready 结果

- 12/12 path全部stage，α最大post diff=`1.4662742614746094e-05`、β最大post diff=
  `3.6135315895080566e-07`、final lower diff=`2.6226043701171875e-06`，全部sign exact；
- pre/candidate/expected-post snapshot hash=`2a775b66...a256`/`103fa9eb...c08`/
  `1fdf3843...cfc`，copy-out hash=`d0c9f1e5...ec64`；candidate因独立snapshot id及容差内浮点差异不要求
  与provider post hash相等，但每个mutable path逐项`allclose+sign exact`；
- 正向commit恰12 path；NaN terminal在stage前拒绝、stale live target在写前拒绝、第五次copy故障注入后
  已写paths全部回滚；focused copy-out=`4 passed`，扩展V4-2D/E=`9 passed`，mypy clean，Pylint=
  `10.00/10`；full=`1173 passed, 3 skipped`。

本切片仍需formal artifact与完全重签tamper；当前状态为
`IMPLEMENTED-ATOMIC-COPY-OUT / FORMAL-ARTIFACT-PENDING`，不关闭V4-2E/V4-2/B2，也不声明性能结果。

## Formal Runner 准备

- 新增`run_rvir_v4_atomic_copy_out_artifact.py`，从冻结V4-2D artifact重新执行V4-2C初始化、
  V4-2D native optimizer、12-path private staging与真实commit，而不是复制已有summary；
- manifest绑定source capture/source manifest、模型、topology与三份执行代码的SHA256；replay会重新导入
  ONNX、重跑10次evaluation/9次update、逐项重建copy-out和commit receipt；
- 正式summary只预备`optimizer_replacement_admitted=true`，同时强制
  `b2_same_solver_timing_admitted=false`和`performance_claimed=false`；该字段只有在artifact和同步重签
  tamper均通过后才构成正式证据；
- capture-ready runner测试连同copy-out负向合同共`5 passed`，mypy clean、Pylint=`10.00/10`。
- formal gate进一步显式冻结`1 core / 6 domains / 6 topology rows / 10 evaluations / 9 updates /
  12 receipts / 7 changed receipts`，避免只由人工从path receipt反推正式结构。

下一步先提交runner以冻结code provenance，再生成artifact和篡改报告；在这些证据落盘前，上述pending
状态保持不变。

## 后续关闭

上述pending状态已由
`gemini_doc/change_2026-08-13_rvir_v4_atomic_copy_out_formal_closure.md`取代：正式artifact/replay、
6类完全重签tamper、全量回归与V4-2逐项acceptance审计均已通过。当前V4-2E=
`VALIDATED-ATOMIC-COPY-OUT`、V4-2=`VALIDATED-OPTIMIZER-REPLACEMENT`；B2仍等待V4-3 whole-core
live integration，不得把本节capture-ready中间状态继续当作当前指令。
