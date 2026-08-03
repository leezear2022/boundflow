# Materialization Plan Schema v1

> Schema：`boundflow.materialization_plan/v1`
> 状态：PR-11 冻结；字段语义改变时必须 bump schema version。

## 1. Evidence record

`MaterializationPlanRecord.to_dict()` 固定包含：

```text
schema_version
context
plan
```

`context` 是决策前输入，至少包含：

- bound method、`requires_grad`、optimization stage；
- alpha/beta/split state；
- batch/spec/domain axes；
- operator summary；
- user/available/safe memory budget；
- expected query reuse；
- target capability。

`plan` 固定包含：

```text
schema_version
policy
action
safe_memory_budget_bytes
recommended_domain_batch_size
reason
candidates
```

每个 candidate 固定包含：

```text
action
capability_legal
memory_feasible
predicted_peak_bytes
predicted_latency_ms
reasons
```

## 2. Action semantics

- `dense`：执行 dense ReLU backward reference path；
- `structured`：执行 feature-gated structured path；
- `reduce_batch`：当前 query 不执行，向 host runtime 返回建议 domain batch，并要求 re-plan。

`reduce_batch` 不是 OOM 成功恢复的同义词；只有 host runtime 实际拆批、重新规划并完成查询后，
才能记录为 feasible execution。

## 3. Capability rules

- target 不支持 structured 时，structured candidate 必须非法；
- requires-grad 且 target 未验证 structured autograd 时，structured 必须非法；
- α/β optimized bound 且 target 未验证对应 capability 时，structured 必须非法；
- structured latency selection 未验证时：若 dense 可行，Global 必须保持 dense；structured 仅作
  memory escape，避免用外推不可靠的 latency model 把慢路径当作加速路径。

## 4. Feasibility rules

```text
safe_budget = safety_margin * min(user_budget, available_memory)
```

先过滤 capability，再比较 predicted peak 与 safe budget；可行集合非空时再比较 latency。若没有
可行 materialization action，则返回 `reduce_batch`。执行后的实测 peak/status 必须单独写入
evaluation JSONL，不能用 predicted feasibility 代替实际结果。

## 5. Contract tests

`tests/test_phase7a_pr11_materialization_planner.py` 冻结 record/context/plan/candidate 顶层 keys，
并验证 JSON serialization、capability filter、budget fallback 和 runtime execution guard。
