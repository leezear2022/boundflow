# 变更记录：Phase 7A PR-9——operator-preserving DAG backward

## 动机

PR-8 已经把 solver 栈从 chain 扩到了最小 residual/general DAG 子集 `{add, concat}`，但 backward 热路径里仍保留两个明确的 dense barrier：

- DAG 汇合点的 adjoint merge 通过 `to_dense() + sum`
- `concat` backward 通过 dense slice 回退到 `DenseLinearOperator`

这会让 general DAG 的语义支持先于表达能力扩展：图虽然能跑，但一遇到多分支/汇合，`LinearOperator` 就会在热路径上提前 materialize，直接抵消 PR-2/PR-3/PR-4 已经搭好的 operator 化基础。

因此 PR-9 的目标收敛为：

- 不扩新图语义
- 不扩新 merge/op
- 只把 PR-8 DAG backward 的 merge/slice 热路径改成 operator-preserving
- 顺手收掉 3 个小问题：
  - `_normalize_concat_axis` 重复定义
  - `concat` 的 shape/axis 校验分散
  - `_split_bias_once(num_children=0)` 没有显式 guard

---

## 主要改动

### 1) `linear_operator.py`：新增 DAG backward 需要的组合算子

- 更新：`boundflow/runtime/linear_operator.py`
  - `LinearOperator` 协议新增：
    - `add(...)`
    - `slice_input(...)`
  - 新增 `AddLinearOperator`
    - 表示两个 shape/input_shape 完全一致的 operator 之和
    - `center_term(...)` / `contract_input(...)` / `to_dense()` 都走组合语义
    - 对 rank-3 `NCHW` 输入的 `row_abs_*` 通过 feature-map rows 归约，不依赖 `Conv2dLinearOperator.to_dense()`
  - 新增 `SliceInputLinearOperator`
    - 表示对逻辑输入轴做 batch-preserving 的 contiguous slice/view
    - 支持 flat 输入与 rank-3 `(C,H,W)` 输入
    - 对 rank-3 输入的 `row_abs_*` 同样走 feature-map rows 归约，不依赖 `Conv2dLinearOperator.to_dense()`
  - `DenseLinearOperator` / `RightMatmulLinearOperator` / `ReshapeInputLinearOperator` / `Conv2dLinearOperator`
    - 统一补上 `add(...)` / `slice_input(...)`
  - `_materialize_feature_map_rows(...)`
    - 新增对 `AddLinearOperator` 和 `SliceInputLinearOperator` 的支持

### 2) 新增 `dag_utils.py`，统一 concat 的 axis/shape 规则

- 新增：`boundflow/runtime/dag_utils.py`
  - `normalize_concat_axis(...)`
  - `validate_concat_tensor_shapes(...)`
  - `validate_concat_value_shapes(...)`
- 更新：
  - `boundflow/runtime/crown_ibp.py`
  - `boundflow/runtime/task_executor.py`
- 这样 `concat` 的 axis 归一化和 shape 校验不再在 runtime 内重复维护两套逻辑。

### 3) `crown_ibp.py`：DAG backward 热路径去掉 dense barrier

- 更新：`boundflow/runtime/crown_ibp.py`
  - `_accumulate_backward_state(...)`
    - 从 `to_dense() + sum` 改成 `A_u.add(...)` / `A_l.add(...)`
  - 删除 `_slice_concat_operator(...)` 的 dense 路径
  - `concat` backward
    - 改成先对 `base.A_*` 对齐 `out_shape`
    - 再走 `slice_input(...)`
  - `_split_bias_once(...)`
    - 对 `num_children < 0` 显式报错
    - `num_children == 0` 明确返回空列表
  - `get_crown_ibp_mlp_stats(...)`
    - 改成复用新的 concat axis helper，而不是各处手写允许值

### 4) 新增 PR-9 专项测试

- 新增：`tests/test_phase7a_pr9_dag_linear_operator.py`
  - `AddLinearOperator` 与 dense reference 等价
  - `SliceInputLinearOperator` 在 flat/channel-concat 场景与 dense reference 等价
  - 新 operator 的 rank-3 row norms 不调用 `Conv2dLinearOperator.to_dense()`
  - `add(...).matmul_right(...)` 继续保持 `RightMatmulLinearOperator` 组合路径
- 新增：`tests/test_phase7a_pr9_operator_preserving_dag_backward.py`
  - `run_crown_ibp_mlp(...)` 的 add backward 确实调用 operator `add(...)`
  - `run_crown_ibp_mlp(...)` 的 concat backward 确实调用 operator `slice_input(...)`
  - plain CROWN 在 DAG toy 上保持 sound
  - alpha / alpha-beta 路径继续可运行
  - `_split_bias_once(num_children=0)` 行为被显式锁定

---

## 影响面

- 不改 public API，不改 `run_crown_ibp_mlp(...)` / `run_alpha_crown_mlp(...)` / `run_alpha_beta_crown_mlp(...)` / `solve_bab_mlp(...)` 的签名。
- 不扩 general DAG 的公开语义范围；仍然只承诺 PR-8 的 `{linear, conv2d, relu, flatten, add, concat}` 子集。
- ReLU backward 的 dense barrier 继续保留；PR-9 只去掉 DAG merge / concat backward 的 dense barrier。
- `alpha_crown.py` / `alpha_beta_crown.py` / `bab.py` 不需要重写主算法，它们通过共享的 CROWN backward 路径自动继承 PR-9。

---

## 验证

已执行：

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr9_dag_linear_operator.py \
  tests/test_phase7a_pr9_operator_preserving_dag_backward.py \
  tests/test_phase7a_pr8_general_dag_runtime.py \
  tests/test_phase7a_pr8_general_dag_frontends.py \
  tests/test_phase7a_pr4_conv_lazy_norms.py \
  tests/test_phase7a_pr3_crown_ibp_cnn.py \
  tests/test_phase7a_pr5_alpha_crown_cnn.py \
  tests/test_phase7a_pr6_alpha_beta_crown_cnn.py \
  tests/test_phase7a_pr7_bab_chain_cnn.py \
  tests/test_phase7a_pr7_bab_batch_examples.py
```

结果：

- `47 passed, 4 warnings in 2.52s`

另执行：

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase6b_crown_ibp_mlp.py \
  tests/test_phase6d_alpha_crown_mlp.py \
  tests/test_phase6e_bab_mlp.py \
  tests/test_phase6f_alpha_beta_crown_pr1.py \
  tests/test_phase6g_alpha_beta_multispec_batch.py
```

结果：

- `21 passed in 1.66s`

---

## 已知边界

- `RightMatmulLinearOperator.row_abs_*()` 仍可能通过 `to_dense()` 求精确范数；PR-9 不试图一次性把所有 operator 的 row norm 都完全结构化。
- `concat` backward 仍只支持 batch-preserving 的 feature/channel concat，不扩一般 axis/general reshape DAG。
- ReLU backward 仍是 PR-2 起就明确保留的 dense barrier。

---

## 下一步

- 如果后续要继续沿“扩表达”方向推进，下一步更自然的是评估：
  - 是否要把 ReLU barrier 也进一步结构化
  - 是否要给 `RightMatmulLinearOperator` / 更一般 operator 组合补更强的 lazy row-norm 实现
- 但这应作为后续单独 PR 讨论，不和 PR-9 混在一起。
