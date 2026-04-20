# 变更记录：2026-03-28 Phase 7A PR-10——ReLU barrier structured

## 动机

PR-9 已经把 general DAG backward 的 `add/concat` 热路径从显式 dense 回退成 operator-preserving，但 `ReLU` backward 仍会：

- 先把 `A_u/A_l` materialize 成 dense
- 再做 sign-selective relaxation
- 最后重新包回普通 `DenseLinearOperator`

这会让 plain CROWN、alpha-CROWN、alpha-beta-CROWN、BaB 共享的 backward 主路径在 ReLU 处提前失去 operator 结构。

PR-10 的目标是把这一步改成 **sign-split structured**：

- 允许在 ReLU 内部做一次精确的正/负系数分解
- 但 ReLU 之后不再回退成普通 `DenseLinearOperator`
- 后续线性/卷积/DAG backward 继续沿 `LinearOperator` 主路径传播

## 主要改动

### 1. `linear_operator.py` 扩展 sign-split 与结构化 operator

更新：`boundflow/runtime/linear_operator.py`

- `LinearOperator` 协议新增 `split_pos_neg()`：
  - 返回与原 operator 精确等价的正/负系数分解 `(A_pos, A_neg)`
  - contract 不要求 lazy；默认实现允许精确 materialize
- `DenseLinearOperator` 提供直接的逐元素 `clamp_min/clamp_max` 分解
- 其余 operator（`RightMatmulLinearOperator`、`ReshapeInputLinearOperator`、`Conv2dLinearOperator`、`AddLinearOperator`、`SliceInputLinearOperator`）统一补 `split_pos_neg()`，默认通过 dense 精确分解回退
- 新增 `ScaledInputLinearOperator`：
  - 表示对逻辑输入轴做逐元素非负缩放
  - ReLU 的 `alpha_u/alpha_l` 通过它乘到 operator 输入轴
  - 保留 autograd 图，不对 scale 做 `detach()`
- 新增 `RepeatedRowLinearOperator`：
  - 表示同一输入系数向量在 spec 维上广播到所有 rows
  - 用于结构化承接 `relu_pre_add_coeff_{u,l}`

### 2. `crown_ibp.py` 重写 ReLU backward

更新：`boundflow/runtime/crown_ibp.py`

- `_backprop_relu_step(...)` 改为：
  - 先对 `state.A_u/A_l` 调用 `split_pos_neg()`
  - 再通过 `ScaledInputLinearOperator` 组合出新的 `A_u'/A_l'`
  - `beta_u/beta_l` 对 `b_u/b_l` 的贡献仍按原公式精确更新
  - `relu_pre_add_coeff_{u,l}` 改由 `RepeatedRowLinearOperator` 注入
- ReLU backward 不再返回普通 `DenseLinearOperator`
- `relu_pre_add_coeff_u` 对 rank>2 pre-activation 的 `NotImplementedError` 继续保留，不扩范围

### 3. 新增 PR-10 专项测试

新增：`tests/test_phase7a_pr10_relu_barrier_structured.py`

覆盖：

- `split_pos_neg()` 与 dense reference 等价
- `ScaledInputLinearOperator` 与 dense reference 等价
- `RepeatedRowLinearOperator` 与 dense reference 等价
- `_backprop_relu_step(...)` 在 operator 路径下不回退成普通 `DenseLinearOperator`
- 结构化 ReLU 路径与旧 dense 参考数值一致
- `relu_alpha` 梯度仍能穿过结构化 operator
- `relu_pre_add_coeff_l` 接回结构化路径后，beta 梯度仍非零且有限

### 4. 文档对齐

更新：

- `boundflow/runtime/crown_ibp.py`
- `boundflow/runtime/alpha_crown.py`
- `boundflow/runtime/alpha_beta_crown.py`
- `boundflow/runtime/bab.py`

修正了先前关于 chain-only / MLP-only / conv BaB 能力的过期描述，使 runtime 文档与当前 Phase 7A 代码现状一致。

## 影响面

- 不改 public API；对外函数签名保持不变
- `LinearOperator` 内部协议新增 `split_pos_neg()`
- `row_abs_*` 在新 operator 上允许精确 dense fallback，但不引入三角不等式式的额外松弛
- first-layer infeasible detector 中的独立 dense 点继续保留，不纳入本 PR

## 验证

已执行：

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7a_pr10_relu_barrier_structured.py
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr9_operator_preserving_dag_backward.py \
  tests/test_phase7a_pr5_alpha_crown_cnn.py \
  tests/test_phase7a_pr6_alpha_beta_crown_cnn.py \
  tests/test_phase7a_pr7_bab_chain_cnn.py \
  tests/test_phase6d_alpha_crown_mlp.py \
  tests/test_phase6g_alpha_beta_multispec_batch.py
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr9_dag_linear_operator.py \
  tests/test_phase7a_pr4_conv_lazy_norms.py
```

结果：

- `tests/test_phase7a_pr10_relu_barrier_structured.py`: `6 passed in 0.81s`
- Phase 7A/Phase 6 shared-path 回归：`27 passed in 1.67s`
- operator / conv lazy-norm 回归：`12 passed in 0.76s`
