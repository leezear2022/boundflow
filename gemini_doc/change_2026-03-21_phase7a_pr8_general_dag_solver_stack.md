# 变更记录：Phase 7A PR-8——solver 栈从 chain 扩到 residual/general DAG（含 Torch/ONNX 前端）

## 动机

到 PR-7 为止，BoundFlow 的 solver 栈已经覆盖：

- chain MLP
- chain CNN
- conv `alpha/alpha-beta/BaB`
- `B>1` 的 mixed-example node-batch

但一个明显的边界还没打通：所有核心 solver 仍默认“图是链式的”。这会直接卡住 ResNet/basic-block 风格的最小 general DAG：

- residual add
- projection skip
- channel/feature concat

PR-8 的目标就是把当前 `{linear, conv2d, relu, flatten}` 的 chain-only solver 栈，扩到 **单 task、静态 shape 的 general DAG 子集**，并在同一 PR 内一起打通：

- plain CROWN-IBP
- alpha-CROWN
- alpha-beta-CROWN
- BaB
- Torch / ONNX 前端导入

本 PR 明确只做：

- `add`
- `concat`
- residual/general DAG

本 PR 不做：

- `mul/gate`
- dynamic shape
- 一般 `permute/reshape` DAG 语义
- skip/branch/general DAG 之外的新 merge
- deeper DAG infeasible 证书

---

## 主要改动

### 1) `crown_ibp.py`：从 chain backward 扩到 reverse-topo DAG backward

- 更新：`boundflow/runtime/crown_ibp.py`
  - `_forward_ibp_trace_mlp(...)` 新增：
    - `add`
    - `concat`
  - `add` 只支持 **exact same-shape** 输入，不允许 broadcast add
  - `concat` 只支持：
    - rank-2 `[B,F]` 的 feature axis
    - rank-4 `NCHW` 的 channel axis
  - backward 主路径从“倒序链式回扫”改成：
    - reverse topo 遍历
    - 按 value 名称聚合 adjoint
  - 在 DAG 汇合点统一使用 **exact dense barrier**：
    - 多路 adjoint 汇合时，将各路 `LinearOperator` materialize 成 dense 后求和
    - `concat` backward 也走 exact dense slice
  - 线性/卷积链路仍沿用 PR2-PR4 的 operator 路径；只在 DAG 汇合点落 dense barrier

### 2) plain CROWN-IBP stats/runtime 不再把 non-chain 视为天然 unsupported

- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(...)` 去掉 chain-only 结构检查
  - `get_crown_ibp_mlp_stats(...)` 改成接受 general DAG 子集：
    - `{linear, conv2d, relu, flatten, add, concat}`
  - 继续保留边界：
    - `flatten(start_dim=1,end_dim=-1)` only
    - `add` no-broadcast
    - `concat` 仅 feature/channel axis

### 3) `task_executor.py`：IBP reference executor 新增 `concat`，并显式拒绝 broadcast add

- 更新：`boundflow/runtime/task_executor.py`
  - `PythonTaskExecutor.run_ibp(...)`
  - `PythonTaskExecutor.run_ibp_task(...)`
  - 新增 `concat` 的 interval 执行
  - `add` 改成显式 shape-check，不再默许 broadcast

这样前端 residual/concat 图在 `plan_interval_ibp_v0(...) + PythonTaskExecutor` 路径上也能稳定落地。

### 4) Torch frontend：新增 `torch.cat` / `aten.cat.default`

- 更新：`boundflow/frontends/pytorch/frontend.py`
  - `_map_torch_target_to_primitive(...)` 新增：
    - `aten.cat.default -> concat`
    - `aten.concat.default -> concat`
  - `_extract_attrs_for_call_function(...)` 新增 `concat.axis`

这样 `torch.export` 导出的 residual/concat 图可以直接进 Primal IR，再进入现有 planner/runtime。

### 5) ONNX frontend：新增 `Concat`

- 更新：`boundflow/frontends/onnx/frontend.py`
  - 新增 `Concat` 导入
  - 保留已有 `Add`
  - residual add / concat 图现在都能从 ONNX 进入同一套 runtime

### 6) alpha/alpha-beta 在“全 stable ReLU”时不再误炸

- 更新：
  - `boundflow/runtime/alpha_crown.py`
  - `boundflow/runtime/alpha_beta_crown.py`
  - 当当前图上的 ReLU 全部已被 IBP 打成 stable，导致 loss 不再依赖可训练 `alpha/beta` 参数时：
    - 不再在 `backward()` 处报 `does not require grad`
    - 直接把 step-0 结果当作最优结果返回

这个修正不是 DAG 专属语义，但在 PR-8 的 residual toy 上第一次稳定暴露出来，所以一并收口。

---

## 支持范围与默认值

PR-8 完成后，solver 栈支持的 general DAG 子集固定为：

- 单 task
- 静态 shape
- op 子集：
  - `linear`
  - `conv2d`
  - `relu`
  - `flatten`
  - `add`
  - `concat`

额外规则：

- `add`：严格同 shape，不支持 broadcast
- `concat`：
  - 只支持 `NCHW` channel axis
  - 只支持 flatten 后 `[B,F]` 的 feature axis
- `flatten`：继续只支持 classifier-style flatten
- infeasible detector：继续只收集 direct-input first-layer affine producer
  - deeper residual/concat split 允许 oracle/BaB 正常运行
  - 但 detector 继续返回 `unknown`

---

## 测试

### 新增

- `tests/test_phase7a_pr8_general_dag_runtime.py`
  - residual add block：plain CROWN / alpha / alpha-beta / BaB 都可运行且 sound
  - projection skip：`1x1 conv` skip 可运行
  - concat merge：feature concat 与 channel concat 都可运行
  - deeper residual split 不误触 detector 新证书
  - unsupported case 明确报错：
    - broadcast add
    - bad concat axis

- `tests/test_phase7a_pr8_general_dag_frontends.py`
  - Torch residual/add import 后可跑 IBP 与 CROWN
  - Torch concat import 后可跑 IBP 与 CROWN
  - ONNX residual/import 与 Torch import 对齐
  - ONNX concat/import 与 Torch import 对齐

### 更新

- `tests/test_phase6b_crown_ibp_mlp.py`
  - general DAG branch graph 不再被视为 unsupported

- `tests/test_phase7a_pr3_crown_ibp_cnn.py`
  - branch-like CNN DAG stats 不再被视为 unsupported

---

## 如何验证

### PR-8 新增路径

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr8_general_dag_runtime.py \
  tests/test_phase7a_pr8_general_dag_frontends.py \
  tests/test_phase6b_crown_ibp_mlp.py::test_crown_ibp_mlp_supports_general_dag_branch_graph \
  tests/test_phase7a_pr3_crown_ibp_cnn.py::test_crown_ibp_stats_supports_chain_cnn_and_branch_like_cnn_dag
```

实测结果：`9 passed, 4 warnings in 2.32s`

### 定向回归

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr8_general_dag_runtime.py \
  tests/test_phase7a_pr8_general_dag_frontends.py \
  tests/test_phase7a_pr7_bab_chain_cnn.py \
  tests/test_phase7a_pr7_bab_batch_examples.py \
  tests/test_phase7a_pr6_alpha_beta_crown_cnn.py \
  tests/test_phase7a_pr5_alpha_crown_cnn.py \
  tests/test_phase7a_pr4_conv_lazy_norms.py \
  tests/test_phase7a_pr3_crown_ibp_cnn.py \
  tests/test_phase6b_crown_ibp_mlp.py \
  tests/test_phase6d_alpha_crown_mlp.py \
  tests/test_phase6e_bab_mlp.py \
  tests/test_phase6f_alpha_beta_crown_pr1.py \
  tests/test_phase6g_alpha_beta_multispec_batch.py \
  tests/test_phase6g_bab_node_batch.py \
  tests/test_phase6g_bab_node_eval_cache.py \
  tests/test_phase6g_node_batch_grad_isolation.py \
  tests/test_phase6g_branch_pick_reuses_forward_trace.py \
  tests/test_phase6g_node_batch_partial_infeasible_prune.py \
  tests/test_phase6h_bench_e2e_schema.py \
  tests/test_phase4d_onnx_frontend_matches_torch.py
```

实测结果：`65 passed, 8 warnings in 3.68s`

---

## 备注

- PR-8 是 **行为扩展**，不新开 `*_dag` 函数族；现有 API 名字全部保持不变。
- exactness 优先于性能：DAG 汇合点现在允许走 dense barrier；后续若要做 operator-preserving DAG backward，建议单独开 PR。
- 更一般的 skip/branch/general DAG 虽然语义已从 chain 扩到最小 residual/basic-block 子集，但本 PR 仍然只承诺：
  - 单 task
  - 静态 shape
  - `{add, concat}` merge
