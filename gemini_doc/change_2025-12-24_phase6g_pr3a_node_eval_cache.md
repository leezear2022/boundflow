# 变更记录：Phase 6G PR-3A（BaB/αβ）——NodeEvalCache（split+config+spec）与命中回归钉子

## 动机

Phase 6G 的主线是 **不动语义，只做系统化收益**。在 PR-2 完成 node-batch（batch pick + batch eval）后，下一步最稳、最不容易返工的提速点是：

- 避免重复评估同一个 split pattern（同一 `split_state` 在同一 run 内被重复 tighten / 重复触达时）；
- 为后续 PR-3B（warm-start cache 口径固化）与 PR-3C（更复杂的 batch infeasible 策略）预留清晰的 cache key 形态。

因此本 PR 先实现 **NodeEvalCache（主缓存）**，并补齐 “cache hit 不会触发 oracle 重算” 的回归钉子。

## 主要改动

### 1) NodeEvalCache：key/value 形态

- 更新：`boundflow/runtime/bab.py`
  - 新增 `NodeEvalCache`（内存缓存，单进程/单 run 语义）
  - cache key（最小可行版）：
    - `module`（进程内 `id(module)`）
    - `input_spec` 指纹（`center` + `perturbation` + `value_name`）
    - `linear_spec_C` 指纹（可选）
    - `oracle_config` 指纹（`steps/lr/alpha_init/beta_init/objective/spec_reduce/...`）
    - `split_state` 指纹（按 `relu_input` 排序，对 split tensor 做 hash）
  - cache value：
    - `bounds`（`IntervalState`）
    - `best_alpha` / `best_beta`（用于子节点 warm-start）
    - `stats`（当前仅用于调试/记录）

### 2) 统一入口：eval_bab_alpha_beta_node（带 cache）

- 更新：`boundflow/runtime/bab.py`
  - 新增 `eval_bab_alpha_beta_node(...)`：优先查 cache，miss 时调用 `run_alpha_beta_crown_mlp` 并写回 cache
  - `solve_bab_mlp`：
    - alpha-beta 路径默认开启 cache（可用 `BabConfig.enable_node_eval_cache` 控制）
    - node-batch 时允许 **partial hit**：batch 内已命中的节点直接复用，未命中的节点组成子 batch 交给 oracle

## 测试

- 新增：`tests/test_phase6g_bab_node_eval_cache.py`
  - DoD：同一 node 连续 eval 两次，第二次命中 cache 且 oracle 调用次数不增加
  - DoD：只改一个 neuron 的 split，必须 cache miss

## 如何验证

```bash
python -m pytest -q tests/test_phase6g_bab_node_eval_cache.py
python -m pytest -q tests/test_phase6e_bab_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_bab_node_batch.py
```

## 备注

- 这是 **进程内缓存**（本 PR 不承诺跨进程/跨运行复用），优先保证语义正确与接口形态稳定。
- cache key 当前不包含 warm-start 的具体数值（将 warm-start 作为 value 携带），目的是“尽量复用、只提升 tightness”；后续若引入 “同一 node 多次 tighten 且 warm-start/optimizer state 显著影响收敛” 的场景，可在 PR-3B 再讨论是否把 warm-start 进入 key。

