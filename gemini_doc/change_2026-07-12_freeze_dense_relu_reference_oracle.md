# 变更记录：冻结 dense ReLU backward reference oracle

## 修改

- 将当前 eager ReLU backward 数学抽成 `_backprop_relu_step_dense_reference`。
- reference 显式返回 `A_u`、`A_l`、`b_u`、`b_l`，不依赖 Git 历史作为隐式 oracle。
- 当前 `_backprop_relu_step` 仍 materialize upper/lower coefficient，再调用 reference 并返回
  `DenseLinearOperator`；因此本批不改变执行语义或 materialization 行为。

## 覆盖

- stable-positive、stable-negative 与 unstable ReLU 同时存在；
- batch > 1、spec > 1；
- flat MLP 与 NCHW CNN；
- α relaxation；
- αβ 使用的 `relu_pre_add_coeff_u/l`；
- 局部 `A/b` 等价和 α gradient 的独立 sign-selection 公式。

## 下一步

以该函数作为 SignSplit operator 的 local-step oracle；再增加完整 backward、solver 与多轮 α
optimization 的 dense/structured 双路径对照。

## 验证结果

- dense oracle、materialization、CROWN、αβ 与 DAG 定向回归：24 passed；
- 全量：170 passed、1 个预期 skip；
- `git diff --check` 通过；未对 `crown_ibp.py` 做全文件格式化。
