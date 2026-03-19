# 变更记录：Phase 6G PR-2（BaB）——node-batch（batch pick + batch eval）回归 + node-batch 梯度隔离钉子 + 吞吐 microbench

## 动机

Phase 6F PR-2 已把 “complete 的支点回到 β 编码” 的语义闭环钉死；Phase 6G 的目标是 **不动语义，只做系统化收益**。PR-2 要把 BaB 从 “一次评估 1 个节点” 升级到 “一次评估 K 个节点”，让 bound oracle（αβ-CROWN）吃满 batch，从而提升吞吐。

此外，node-batch 引入了新的 batch 维度（node 维），需要补齐两类 reviewer-proof 的钉子：

- **K=1 与 K>1 最终 verdict 一致**（至少在确定性 toy 上）；
- **node-batch 的梯度不会串味**（确保后续对 warm-start/优化的扩展不会破坏 autograd 语义）。

## 主要改动

### 1) αβ oracle 支持 node-batch 的 split/beta 注入（必要的接口补齐）

- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - `run_alpha_beta_crown_mlp`：
    - infeasible detector 目前仅支持 `B==1` 且 split 为 `[H]`；当 `B>1`（node-batch）时跳过该 best-effort 检测，避免误用导致报错。
  - `_beta_to_relu_pre_add_coeff`：
    - 支持 `relu_split_state[name]` 为 `[H]` 或 `[B,H]`；
    - 支持 `beta[name]` 为 `[H]` 或 `[B,H]`，并在 `[B,H]` 情形下逐 node 生成 `relu_pre_add_coeff_l[name]`，用于把 `-β*s` 注入到 pre-activation 线性系数。

### 2) BaB driver：batch pick + batch eval（node-batch）

- 已有（本 PR 依赖）：`boundflow/runtime/bab.py`
  - `BabConfig.node_batch_size`：控制每轮 pop 的节点数（K）。
  - `solve_bab_mlp`：当 `oracle="alpha_beta"` 且 `node_batch_size>1` 时，将 node 维度作为 batch 维一次性调用 oracle：
    - `center: [1,I] -> [K,I]` 广播
    - `C: [1,S,O] -> [K,S,O]` 广播
    - `split_state: [K,H]` 堆叠
    - `warm_start alpha/beta: [K,H]` 堆叠并使用 `per_batch_params=True`

## 测试与基准

- 新增：`tests/test_phase6g_bab_node_batch.py`
  - 1D toy 上 `node_batch_size=1` 与 `node_batch_size=4` 均应 `proven`（回归 K=1 等价 + K>1 正确性）。
- 新增：`tests/test_phase6g_node_batch_grad_isolation.py`
  - 对 `run_crown_ibp_mlp` 构造 node-batch 参数 `alpha/beta:[B_nodes,H]`，只对 node0 的 loss 反传，断言其它 node 的梯度为 0（防串味）。
- 新增：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - microbench：对同一组 split states，比较 “batched node-eval vs serial node-eval” 的 p50 耗时与 speedup（stdout 输出 JSON）。

## 如何验证

```bash
python -m pytest -q tests/test_phase6g_bab_node_batch.py
python -m pytest -q tests/test_phase6g_node_batch_grad_isolation.py

# 可选：吞吐 microbench（CPU）
python scripts/bench_phase6g_bab_node_batch_throughput.py --device cpu --nodes 32 --specs 16 --steps 0
```

## 备注

- PR-2 的目标是把 **node-batch 的并行度**（BaB 的吞吐收益）落地；缓存（cache key / reuse）留到 Phase 6G PR-3。
- infeasible detector 在 `B>1` 场景暂不启用（否则需要 per-node 证书与更复杂的 batch 语义）；不影响 BaB 正确性，只影响剪枝效率。

