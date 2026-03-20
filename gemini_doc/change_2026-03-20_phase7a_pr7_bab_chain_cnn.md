# 变更记录：Phase 7A PR-7——BaB on chain CNN（含 node-batch，与真实样本 batch 共存）

## 动机

PR-6 已经把 `alpha-beta-CROWN` oracle 扩到链式 CNN 子集 `{conv2d,relu,flatten,linear}`，但 `bab.py` 仍然停在 MLP-only：

- conv 图在入口处直接 fail-fast；
- `ReluSplitState.empty(...)` 只能从 rank-2 `linear` 权重推断 `[H]`；
- node-batch 仍假设输入 `B==1`，把 batch 维硬编码成“节点维”；
- `BabResult` 只有聚合结果，无法在 `B>1` 下表达逐样本搜索状态。

本 PR 的目标是把 `solve_bab_mlp(...)` 扩到 **chain CNN**，同时允许：

- conv 图上的 BaB（仅 `oracle="alpha_beta"`）；
- node-batch 混合不同样本的节点；
- 真实样本 batch `B>1` 与 node-batch 共存；
- 每个样本保留独立搜索树、独立预算与独立 verdict。

本 PR **不** 扩 skip/branch/general DAG；这条线留到后续 PR-8。

---

## 主要改动

### 1) `bab.py`：BaB 从 MLP-only 扩到 chain CNN

- 更新：`boundflow/runtime/bab.py`
  - `solve_bab_mlp(...)` 现在支持：
    - MLP：`linear/relu/linear`
    - chain CNN：`conv2d/relu/.../flatten(start_dim=1,end_dim=-1)/linear`
  - 但 conv 图只在 `cfg.oracle == "alpha_beta"` 时开放；`oracle="alpha"` 仍显式报错：
    - `alpha-only BaB does not yet support conv graphs`

### 2) `ReluSplitState` 扩到高维逻辑 shape

- 更新：`boundflow/runtime/bab.py`
  - `ReluSplitState.empty(...)` 新增 `input_spec=`，通过一次 forward trace 从 `relu_pre` 推断逻辑 shape；
  - conv ReLU 的 split 张量现在可存成 `[*S]`（例如 `[C,H,W]`），而不是只支持 `[H]`；
  - `with_split(...)` 继续使用 flat idx，但内部改成对逻辑 shape 做 `view(-1)` 更新后再恢复。

### 3) host 侧采用“每样本独立搜索树”，oracle 侧采用 batched subproblems

- 更新：`boundflow/runtime/bab.py`
  - `_QueueItem` 新增 `example_idx`；
  - root 初始化改成“每个样本一个 root 节点”；
  - 全局 heap 继续存在，但每个样本独立维护：
    - `visited/evaluated/expanded`
    - `best_lower/best_upper`
    - `status`
  - `max_nodes` 语义改成 **每样本独立预算**；
  - node-batch 现在可以从不同样本各取若干节点，stack 成一次 `alpha-beta` oracle 调用。

### 4) `BabResult` 保留旧聚合字段，并新增逐样本结果

- 更新：`boundflow/runtime/bab.py`
  - 新增 `BabPerExampleResult`
  - `BabResult` 新增 `per_example`
  - 旧聚合字段保持不变：
    - `status`
    - `nodes_visited/evaluated/expanded`
    - `best_lower/best_upper`
    - `batch_rounds/max_queue/avg_batch_fill_rate`
  - 聚合口径：
    - 任一样本 `unsafe` => 聚合 `unsafe`
    - 否则全部 `proven` => 聚合 `proven`
    - 否则 `unknown`

### 5) per-node infeasible prune 扩到 mixed-example node-batch

- 更新：`boundflow/runtime/bab.py`
  - `prune_infeasible_first_layer_items(...)` 改成按 `item.example_idx` 切样本，再调用 first-layer detector；
  - `NodeEvalCache` 改成按样本隔离，避免不同样本共享 cache key。

### 6) branch fallback 改成 rank-agnostic

- 更新：`boundflow/runtime/bab.py`
  - `_pick_branch(...)` 不再假设 rank-2 `[B,H]`；
  - 对任意 `relu_pre` 先 flatten 成 `[B,I]`；
  - fallback 与 oracle `branch_choices` 一样都返回 `(relu_input_name, flat_idx)`。

### 7) bench 脚本兼容 PR-7 的新接口

- 更新：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - 适配 `_QueueItem.example_idx`
  - `prune_infeasible_first_layer_items(...)` 改为传 `cache_by_example`
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - instrumentation 的 `_prune_wrapper` 改为兼容 PR-7 新签名
  - 保持旧的计数口径不变，只修复脚本层兼容性

---

## 测试

### 新增

- `tests/test_phase7a_pr7_bab_chain_cnn.py`
  - conv `ReluSplitState.empty/with_split` 使用高维逻辑 shape，branch/fallback 均返回 flat idx
  - chain CNN 上 `solve_bab_mlp(..., oracle="alpha_beta")` 可运行
  - `node_batch_size=1` 与 `node_batch_size>1` 在同一 CNN toy 上 verdict 一致
  - conv first-layer infeasible prune 在 mixed-example batch 中支持 partial prune
  - `oracle="alpha"` 在 conv 图上继续 fail-fast

- `tests/test_phase7a_pr7_bab_batch_examples.py`
  - `B=2` 时返回 `per_example`
  - `max_nodes` 按样本独立预算
  - 真实样本 batch 与 node-batch 共存时，一次 oracle 调用可混合不同样本节点
  - 聚合 `BabResult` 与 `per_example` 字段口径一致

### 更新

- `tests/test_phase7a_pr5_alpha_crown_cnn.py`
  - 将“conv BaB 全局 fail-fast”改为“`oracle="alpha"` fail-fast”

- `tests/test_phase6g_node_batch_partial_infeasible_prune.py`
  - 适配 `_QueueItem.example_idx` 与按样本分 cache 的新接口

---

## 如何验证

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr7_bab_chain_cnn.py \
  tests/test_phase7a_pr7_bab_batch_examples.py \
  tests/test_phase7a_pr5_alpha_crown_cnn.py \
  tests/test_phase6g_node_batch_partial_infeasible_prune.py \
  tests/test_phase6g_bab_node_batch.py \
  tests/test_phase6g_node_batch_grad_isolation.py \
  tests/test_phase6g_bab_node_eval_cache.py \
  tests/test_phase6g_branch_pick_reuses_forward_trace.py \
  tests/test_phase6e_bab_mlp.py \
  tests/test_phase6f_alpha_beta_crown_pr1.py \
  tests/test_phase6g_alpha_beta_multispec_batch.py \
  tests/test_phase6d_alpha_crown_mlp.py \
  tests/test_phase7a_pr6_alpha_beta_crown_cnn.py \
  tests/test_phase7a_pr3_crown_ibp_cnn.py \
  tests/test_phase7a_pr4_conv_lazy_norms.py \
  tests/test_phase6h_bench_e2e_schema.py
```

实测结果：`49 passed in 7.35s`

---

## 备注

- PR-7 只开 **chain CNN BaB**，不扩 skip/branch/general DAG。
- `B>1` 采用混合方案：
  - host/BaB：每个样本独立搜索树
  - oracle：batch 维表示一批待评估节点/domain
- `InputSpec` 仍要求一个 batch 内共享同一个 `PerturbationSet` 类型与同一个 `eps`；本 PR 不支持“同一 batch 内每个样本不同 `p/eps`”。
- 现有 bench/schema 不因为 `BabResult.per_example` 新字段而 bump 版本；旧脚本继续消费聚合字段即可。
- DAG/skip/branch 的 general graph 支持留到 PR-8。
