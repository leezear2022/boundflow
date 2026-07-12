# 变更记录：ReLU backward 保持 structured coefficient

## 修改

- 默认 ReLU backward 返回 `SignSplitLinearOperator`，不再把主 coefficient 永久转换为
  `DenseLinearOperator`。
- ReLU intercept/bias reduction 使用 `relu_bias_sign_reduce` 局部 materialization，标记为
  `ephemeral`，归约到 `[B,S]` 后结束逻辑生命周期。
- αβ 的 pre-add coefficient 作为 structured SignSplit 与 batch/feature dense addend 的
  `AddLinearOperator`，不物化主 coefficient。
- `BOUNDFLOW_RELU_BACKWARD_MODE=dense` 保留进程级 reference fallback；测试使用 task-local
  context 在 dense/structured 间切换，未修改现有 public solver API。
- 增加 deterministic operator-tree dump，稳定编号、记录 shape/metadata，不包含 tensor 值或
  对象地址。

## 数学边界

SignSplit 只表达 `A⁺⊙s⁺ + A⁻⊙s⁻`。后续 matmul/conv/add/reshape/slice 只能包裹该节点，
没有把 sign 判断分配进权重或卷积内部。

## 验证结果

- local `A/b` dense 等价；flat/NCHW α gradient 等价；
- plain CROWN 与 3-step α-CROWN dense/structured 等价；
- CROWN、α、αβ、BaB、CNN、DAG 既有回归全部通过；
- 全量：177 passed、1 个预期 skip；`git diff --check` 通过；
- materialization 与 operator dump 定向测试 14 passed，materialization 模块 Pylint 10/10。

## 未完成

- 仍需显式的 αβ/BaB dense-vs-structured fixed replay oracle；
- 仍需 dense/structured profile 对照，判断 persistent bytes、peak memory 与 runtime guardrail；
- real BaB domain batch 尚未进入 profile。
