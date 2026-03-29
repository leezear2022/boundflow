# 2026-03-28：shared CROWN layout-only 支持（reshape + structured permute）

## 背景

在 PR-10 完成 ReLU barrier 结构化之后，`pytest tests/` 仍有 2 个前端回归失败：

- `tests/test_phase7a_pr8_general_dag_frontends.py::test_phase7a_pr8_torch_and_onnx_frontends_match_on_residual_add_ibp_and_crown`
- `tests/test_phase7a_pr8_general_dag_frontends.py::test_phase7a_pr8_torch_and_onnx_frontends_match_on_concat_ibp_and_crown`

根因不是 ONNX `flatten`，而是 Torch frontend 在这两个用例上产出了 terminal `reshape`，而 shared CROWN 的 `_forward_ibp_trace_mlp(...)` / backward 路径都还不支持 `reshape`。

在收敛这两个失败后，顺手把相邻的 layout-only `permute/transpose` 也接进 shared CROWN，避免下一批前端图再次在 layout op 处卡住。

## 本次改动

### 1. shared CROWN 补齐 `reshape`

更新：`boundflow/runtime/crown_ibp.py`

- `_forward_ibp_trace_mlp(...)` 新增 `reshape` 分支：
  - `shape is None` 时按 identity 处理
  - `shape` 存在时直接对 interval lower/upper 调 `reshape`
  - 语义对齐现有 `PythonTaskExecutor.run_ibp`
- backward 新增 `_backprop_reshape_step(...)`
  - 直接调用 `state.A_u.reshape_input(pre_shape)` / `state.A_l.reshape_input(pre_shape)`
  - 不引入新的 dense materialize
- `run_crown_ibp_mlp(...)` 文档字符串和 `get_crown_ibp_mlp_stats(...)` 同步把 `reshape` 纳入支持集

### 2. 新增结构化 `permute/transpose` 支持

更新：`boundflow/runtime/linear_operator.py`

- 新增内部 `ReindexInputLinearOperator`
  - 使用 gather 语义保存 layout 重排：`base_input_flat = logical_input_flat[:, gather_index]`
  - 预计算 `scatter_index`，用于 `to_dense()` 列重排
  - `center_term(...)` / `contract_input(...)` 通过 gather 映射回 base 输入顺序
  - `row_abs_sum` / `row_l2_norm` / `row_abs_max` 直接复用 base，因为双射列重排不改变行范数
  - `split_pos_neg()` 保持结构化，不走 dense fallback
- `_materialize_feature_map_rows(...)` 补 `ReindexInputLinearOperator` 分支，保证高维 row-norm / add / slice 组合路径仍可工作

更新：`boundflow/runtime/crown_ibp.py`

- 新增 batch-preserving `permute` helper：
  - 统一校验 `dims`
  - 强制 `dims[0] == 0`
  - 生成 backward 用的 gather index
- `_forward_ibp_trace_mlp(...)` 新增 `permute` / `transpose` 分支
- backward 新增 `_backprop_permute_step(...)`
  - 将当前 adjoint 的 input 轴从 permute 后顺序结构化映射回 permute 前顺序
  - 返回 `ReindexInputLinearOperator`，不回退成普通 `DenseLinearOperator`
- `get_crown_ibp_mlp_stats(...)` 与 runtime 文档同步扩到 `permute` / `transpose`

### 3. Torch frontend 对齐 layout-only 图

更新：`boundflow/frontends/pytorch/frontend.py`

- 将 `aten._unsafe_view.default` 规范化为 `reshape`
- 将 `aten.clone.default` 规范化为 no-op `reshape`，并补充注释说明这是 bound propagation 里的 identity 语义
- 为 `reshape` 提取 shape attrs，覆盖：
  - `aten.reshape.default`
  - `aten.view.default`
  - `aten._unsafe_view.default`

这样 Torch export 在 `permute -> reshape` 场景下生成的 `clone + unsafe_view` 能进入现有 primitive op-set，而不用在 executor/runtime 里额外引入 clone 语义。

### 4. 测试补齐

更新：`tests/test_phase7a_pr8_general_dag_frontends.py`

- 新增 `PermuteReshapeNet`
- 新增 Torch / ONNX frontends 在 `permute -> reshape -> linear` 上的 IBP / CROWN 一致性回归

新增：`tests/test_phase7a_pr10_layout_only_shared_crown.py`

覆盖：

- `ReindexInputLinearOperator` 与 dense reference 精确等价
- `split_pos_neg()` 穿过 reindex 后仍与 dense reference 一致
- `_backprop_permute_step(...)` 返回结构化 operator 而不是普通 `DenseLinearOperator`
- `permute -> reshape -> linear` 的 shared CROWN 路径保持 sound

## 影响面

- 不改 public API
- shared CROWN 的支持集扩到：
  - `reshape`
  - batch-preserving `permute/transpose`
- `permute/transpose` 仍只支持 layout-only 子集：`dims[0] == 0`
- 不扩更一般的批维重排语义
- 不新增 layout-op dense barrier

## 验证

已执行：

```bash
conda run -n boundflow python -m py_compile \
  boundflow/frontends/pytorch/frontend.py \
  boundflow/runtime/linear_operator.py \
  boundflow/runtime/crown_ibp.py \
  tests/test_phase7a_pr10_layout_only_shared_crown.py \
  tests/test_phase7a_pr8_general_dag_frontends.py

conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr10_layout_only_shared_crown.py \
  tests/test_phase7a_pr8_general_dag_frontends.py

conda run -n boundflow python -m pytest -q tests/
```

结果：

- targeted tests：`7 passed in 1.49s`
- full suite：`173 passed, 1 skipped in 26.58s`
