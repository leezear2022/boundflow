# RVIR-v4 V4-2D Native Optimizer Parity 修改记录

日期：2026-08-13

## 目标

在不读取production期望step、不调用provider的条件下，从V4-2C native pre-state独立执行固定ResNet2B
的10次lower evaluation和9次Adam update，再将结果与V4-2B冻结GPU真值逐step比较。该切片只验证
mutation语义，不执行solver copy-out，不开启B2计时。

## 实现

- 新增typed native optimizer step/trace/parity合同；
- 固定两组Adam参数与α/β学习率`0.01/0.05`、每次update后`0.98`指数衰减；
- 每轮使用V4-2C external intermediate bounds、dense α/β/split执行native CROWN lower；
- loss严格使用production `reduction_sum`对应的`-lower.sum()`；update后α投影到`[0,1]`、β投影到
  `[0,+∞)`；
- executor签名不接收production trace；reference trace只在独立comparator中逐step映射为dense状态；
- scope、policy、10/9 cardinality、LR schedule、tensor inventory、shape/dtype/finite/sign/allclose均
  fail closed。

## 当前边界

capture-ready实测10/10 step全部allclose且sign exact：跨production GPU真值的lower/α/β全局最大绝对
误差分别为`4.0531158447265625e-06`、`1.4662742614746094e-05`、
`3.986060619354248e-07`，均低于`atol=rtol=2e-4`。native trace hash=
`4e173c22...bc76`，parity hash=`a6b5df97...3959`。

- focused RVIR-v4=`36 passed`；full=`1167 passed, 3 skipped`；
- mypy两文件clean；Pylint两文件=`10.00/10`；
- lower同步重签漂移与scope漂移均fail closed。

formal artifact、完整同步重签名tamper及post-state atomic copy-out仍待后续门禁；因此V4-2D状态仅为
`IMPLEMENTED-STEP-PARITY / FORMAL-ARTIFACT-PENDING`，`optimizer_replacement_admitted=false`、
`b2_same_solver_timing_admitted=false`、`performance_claimed=false`保持不变。

## Formal Runner 准备（同日）

新增`run_rvir_v4_native_optimizer_artifact.py`：从V4-2C正式artifact重新校验source manifest/capture、
ONNX digest和pre-state mapping，再独立执行native loop，输出逐step native trace、parity、summary、
topology、replay stdout、源码revision与文件inventory。replay会重新执行10/9 loop并逐项比较，不信任
序列化摘要。capture-ready focused=`4 passed`，mypy四文件clean，Pylint runner/test=`10.00/10`。

runner需先进入clean commit再生成正式artifact；本段不改变V4-2D pending状态。
