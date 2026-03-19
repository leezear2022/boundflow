# 变更记录：Phase 6G PR-3B（BaB/αβ）——消除分支选择二次 forward（复用 forward trace / branch hint）

## 动机

在 Phase 6G PR-3A（NodeEvalCache）之后，BaB 的下一处“确定性系统收益”是：**消掉分支选择的二次 forward**。

此前 `solve_bab_mlp` 在评估完节点（oracle 已跑过 forward IBP）后，为了选 split neuron 又调用 `_pick_branch -> _forward_ibp_trace_mlp`，导致每个节点至少额外一次 forward，吞吐收益被吃掉。

本 PR 的目标保持不变：

- **不动 αβ 语义**；
- 仅通过“复用 node eval 的 forward trace 产物”减少重复计算；
- 在 cache hit 场景下，分支选择也不应触发 forward。

## 主要改动

### 1) CROWN-IBP：提供 backward-only 入口（从 forward trace 复用）

- 更新：`boundflow/runtime/crown_ibp.py`
  - 新增 `run_crown_ibp_mlp_from_forward_trace(...)`：
    - 输入：`interval_env` + `relu_pre`（来自 `_forward_ibp_trace_mlp`）
    - 输出：与 `run_crown_ibp_mlp` 相同的 `IntervalState`
  - 用途：允许上层（αβ oracle / BaB）一次 forward，多次 backward（优化循环/分支选择复用）。

### 2) αβ oracle：forward trace 只算一次，并产出分支提示（branch_choices）

- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - `run_alpha_beta_crown_mlp`：
    - 在优化循环外先跑一次 `_forward_ibp_trace_mlp` 得到 `interval_env/relu_pre`；
    - 每一步调用 `run_crown_ibp_mlp_from_forward_trace(...)` 执行 backward（避免每步重复 forward）。
  - `AlphaBetaCrownStats` 新增 `branch_choices`：
    - 形状：`List[Optional[(relu_input, neuron_idx)]]`，长度为 batch（node-batch 场景下是 `B_nodes`）。
    - 由 `relu_pre` 的 “最宽不稳定区间”规则生成（与 BaB `_pick_branch` 口径一致）。

### 3) BaB：分支选择优先使用 branch hint（避免二次 forward）

- 更新：`boundflow/runtime/bab.py`
  - `NodeEvalCacheValue` 增加 `branch` 字段（缓存分支提示）。
  - `eval_bab_alpha_beta_node(...)` 返回 `branch_hint`，并写入缓存。
  - `solve_bab_mlp`：
    - alpha-beta 路径在分支阶段优先使用 `branch_hint`；
    - 仅在缺失 hint 时回退到 `_pick_branch`（理论上 alpha-beta + 新 oracle 形态应当总能给 hint）。

## 测试

- 新增：`tests/test_phase6g_branch_pick_reuses_forward_trace.py`
  - monkeypatch 统计 forward trace 调用次数：
    - oracle 内部 forward 调用应为 1；
    - 分支选择不应触发额外 forward（`_pick_branch` 不应再调用 `_forward_ibp_trace_mlp`）。

## 如何验证

```bash
python -m pytest -q tests/test_phase6g_branch_pick_reuses_forward_trace.py
python -m pytest -q tests/test_phase6g_bab_node_eval_cache.py
python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_bab_node_batch.py
```

## 备注

- 本 PR 同时减少了 αβ oracle 的优化循环开销：forward trace 不再随 `steps` 重复计算（forward 与 α/β 无关）。
- `branch_choices` 是“工程提示信息”，不影响 soundness/complete 口径，只影响吞吐。

