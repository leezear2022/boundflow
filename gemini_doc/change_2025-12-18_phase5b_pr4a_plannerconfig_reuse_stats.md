# 变更记录：Phase 5B PR#4A（PlannerConfig 复用配置 + ReuseStats 可观测性）

## 动机

PR#3/PR#3.1 已经把 logical/physical 分层与保守复用打通，但要进入 5B.2（放宽 key、策略消融、bench）之前，需要先把“复用策略/参数”与“命中统计/原因”变成 **PlannerConfig/PlanBundle 可观测**的东西，避免：

- 实验不可复现（策略散落在 pass 参数里）
- 没有 hit/miss 统计导致无法解释“为什么没省内存”
- pytest 被硬阈值绑死（正确性/性能指标混在一起）

因此先落地 PR#4A：**只做配置与统计基础设施**（不做激进放宽 strides 的不安全复用）。

## 本次改动

### 1) Storage reuse 配置与统计类型（planner 层）

- 新增：`boundflow/planner/storage_reuse.py`
  - `StorageReuseOptions`（enabled/include_scopes/key_mode/policy）
  - `ReuseKeyMode`：`STRICT` / `IGNORE_LAYOUT`（注意：不忽略 strides）
  - `ReusePolicy`：`LIFO` / `FIFO`
  - `BufferReuseStats`：`pool_hit/pool_miss/bytes_saved_est/miss_reasons`
  - `estimate_bytes_saved()`：用于 bytes saved 估计

### 2) PlannerConfig 接入 storage_reuse

- 修改：`boundflow/planner/core.py`
  - `PlannerConfig.storage_reuse: StorageReuseOptions`

### 3) BufferReusePass 输出 ReuseStats

- 修改：`boundflow/planner/passes/buffer_reuse_pass.py`
  - `apply_conservative_buffer_reuse(...) -> BufferReuseStats`
  - `BufferReusePass` 将统计写入 `PlanBundle.meta["reuse_stats"]`
  - 兼容：若 `PlannerConfig.enable_storage_reuse=True` 但 `storage_reuse.enabled=False`，会自动开启 reuse

### 4) interval_v2 复用选项透传（保持默认不启用）

- 修改：`boundflow/planner/interval_v2.py`
  - `IntervalV2PartitionConfig` 增加 `reuse_key_mode/reuse_policy`（默认 STRICT/LIFO）
  - `enable_storage_reuse=True` 时把上述选项封装为 `StorageReuseOptions` 传入 reuse pass

### 5) Bench 脚本（指标不进 pytest）

- 新增：`scripts/bench_storage_reuse.py`
  - 打印逻辑/物理 buffer 数、估计 bytes saved、pool hit/miss 与 miss_reasons
  - 支持 `--model mlp|cnn --min-tasks ... --key-mode ... --policy ...`

## 如何验证

```bash
conda run -n boundflow python -m pytest -q
conda run -n boundflow python scripts/bench_storage_reuse.py --model mlp --min-tasks 2
```

