# 变更记录：Phase 5B PR#3.1（Correctness hardening：edge-driven last_use + physical env 断言）

## 动机

PR#3 已引入 logical/physical 分层与保守复用，但为了避免后续 5B.2/5C/5E 扩展时出现“幽灵别名 bug”，需要把两件事进一步钉死：

1. **env 只认 physical buffer id**：当存在 physical plan 时，runtime 读写必须强校验 key 属于 `physical_buffers`。
2. **跨 task 的 last_use 以 TaskGraph 为准**：task 粒度 liveness 的“最后一次使用”必须从边（consumer tasks）推导，而不是只靠扫 op list（否则遇到 partition/normalize 变化时容易漏依赖）。

## 本次改动

### 1) TaskOp 预留 memory effect 字段（占位）

- 修改：`boundflow/ir/task.py`
  - `TaskOp.memory_effect: Optional[str] = None`
  - 仅占位，用于未来 alias/memory-effect 模型（例如 READ/WRITE/READWRITE），当前 pass 不使用。

### 2) Liveness：跨 task uses 来自 TaskGraph edges

- 修改：`boundflow/ir/liveness.py`
  - `compute_liveness_task_level()` 在扫 task ops 的基础上，额外用 `TaskGraph.edges` 更新 `last_use_index`
  - 保证 branch/merge、多 consumer 情况下 last_use 不会被漏掉

### 3) Runtime：physical buffer id 强校验

- 修改：`boundflow/runtime/scheduler.py`
  - 当 `storage_plan.physical_buffers` 非空时，输入/输出 buffer id 必须在其中，否则直接报错
- 修改：`boundflow/runtime/task_executor.py`
  - `run_ibp_task()` 内 `value_name -> logical -> physical` 后，对 physical id 做同样强校验

### 4) Reuse policy hook（为 5B.2 预留）

- 修改：`boundflow/planner/passes/buffer_reuse_pass.py`
  - 增加 `ReusePolicyFn` 与默认 `lifo_reuse_policy`（仍保持 v0 行为）
  - 允许未来用 cost model/policy 替换选择策略（不影响当前正确性）

## 如何验证

```bash
conda run -n boundflow python -m pytest -q
```

