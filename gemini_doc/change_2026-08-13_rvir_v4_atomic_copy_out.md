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
