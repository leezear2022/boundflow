# 变更记录：Phase 6G PR-3C（BaB/αβ）——per-node infeasible/reason/witness + node-batch partial prune（first-layer）

## 动机

在 PR-3A（NodeEvalCache）与 PR-3B（branch pick 复用 forward trace）之后，node-batch 的下一处工程缺口是：

- `B>1`（node-batch）下 infeasible detector 被跳过，导致 batch 内无法“部分剪枝”（只影响效率，不影响 soundness）；
- `feasible/reason/witness` 需要成为 **per-node** 的一等公民（否则 cache/reuse 时语义不完整，reviewer 也会追问）。

本 PR 先在 **first-layer split halfspaces** 的范围内（Phase 6F PR-2 的 convex-combo 证书）实现：

- per-node 的 infeasible 检测与缓存；
- node-batch batch 内的 partial prune（把 infeasible 节点从待评估列表中剔除）。

## 主要改动

### 1) 提供 per-node infeasible 检测入口

- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 新增 `check_first_layer_infeasible_split(...) -> AlphaBetaCrownStats`
    - infeasible 时返回 `feasibility="infeasible"` + `reason` + `infeasible_certificate`
    - 否则返回 `feasibility="unknown"`（并保留 reason）

### 2) BaB node-batch partial prune（first-layer）

- 更新：`boundflow/runtime/bab.py`
  - `BabConfig.enable_batch_infeasible_prune: bool = False`（默认关闭，避免不必要开销）
  - 新增 `prune_infeasible_first_layer_items(...)`：
    - 对 batch 内每个节点做 first-layer infeasible 检测；
    - infeasible 节点直接 prune，并写入 `NodeEvalCache`（记录 `stats.feasibility/reason/witness`）；
    - 支持 cache hit（已知 infeasible 的节点不再重复检测）。
  - `solve_bab_mlp` node-batch 路径接入该 prune：batch 内允许“部分剪枝”。

## 测试

- 新增：`tests/test_phase6g_node_batch_partial_infeasible_prune.py`
  - 构造 batch 内混合 infeasible/feasible 的节点：
    - infeasible：Phase 6F 的 3-direction L2 ball 证书用例（非 pairwise）
    - feasible：同网络的全 0 split
  - 断言：
    - prune helper 只剔除 infeasible 节点（partial prune）
    - infeasible 结果被写入 cache（后续命中不丢失 `feasible/reason/witness`）

## 如何验证

```bash
python -m pytest -q tests/test_phase6g_node_batch_partial_infeasible_prune.py
```

## 备注

- 当前 PR-3C 的 infeasible/证书仍限定在 **first-layer**（与 Phase 6F PR-2 一致）；更一般的 infeasible 需要 β 编码的更深层约束与更复杂的 witness 表示，建议后续单独推进。
- 默认关闭 `enable_batch_infeasible_prune`，避免把“证书搜索”的额外开销默认加到每次 batch eval 上；bench/竞赛模式可显式开启。

