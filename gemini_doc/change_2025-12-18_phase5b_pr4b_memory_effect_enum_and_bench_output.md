# 变更记录：Phase 5B PR#4B（memory_effect Enum + bench 输出 + 更细 miss reasons）

## 动机

在 PR#3.1 已经钉死 “env 只认 physical id / last_use 由 TaskGraph edges 驱动” 的安全边界后，进入 5B.2 需要进一步：

- 避免 `memory_effect` 使用字符串带来的拼写错误与分支爆炸；
- 把复用效果的统计输出做成可落地的 bench 工具（CSV/JSON），方便后续画图与消融；
- 让 `miss_reasons` 更可解释（例如区分 `KEY_MISMATCH` vs “还没有 free buffer”）。

本 PR 不做任何 “忽略 strides” 的不安全放宽；`IGNORE_LAYOUT` 仍保留 strides。

## 本次改动

### 1) memory_effect 升级为 Enum

- 修改：`boundflow/ir/task.py`
  - 新增 `MemoryEffect` enum：`READ/WRITE/READWRITE/ALLOC/FREE`
  - `TaskOp.memory_effect: Optional[MemoryEffect] = None`

### 2) 更细的 miss reasons + 预留 respect_memory_effect 开关

- 修改：`boundflow/planner/storage_reuse.py`
  - `StorageReuseOptions.respect_memory_effect: bool = False`（占位，后续让复用/插拷贝逻辑尊重 effect 冲突）
  - `ReuseMissReason.KEY_MISMATCH`
- 修改：`boundflow/planner/passes/buffer_reuse_pass.py`
  - `miss_reasons` 进一步拆分：`NO_FREE_BUFFER`（pool 为空）、`KEY_MISMATCH`（pool 非空但无同 key）、`LIFETIME_OVERLAP`（存在同 key 但仍活跃未释放）
  - 统计 free pool 碎片度：`max_free_pool_keys/max_free_pool_buffers`

### 3) bench 脚本支持 text/json/csv 输出

- 修改：`scripts/bench_storage_reuse.py`
  - 新增 `--format text|json|csv` 与 `--out <path>`
  - JSON 输出包含 before/after 两个对象；CSV 输出两行（phase=before/after），并扁平化 `miss_reasons`
  - 输出补齐复现信息：`git_commit`、输入 shape、DAG 规模（num_tasks/num_edges）、reuse 配置
  - 额外输出 `why_not_reused_topk`（Top-5 miss reasons）

## 如何验证

```bash
conda run -n boundflow python -m pytest -q
conda run -n boundflow python scripts/bench_storage_reuse.py --model mlp --min-tasks 2 --format json
conda run -n boundflow python scripts/bench_storage_reuse.py --model mlp --min-tasks 2 --format csv
```
