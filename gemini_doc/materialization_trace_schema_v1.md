# BoundFlow Materialization Trace Schema v1

> Schema ID：`boundflow.materialization/v1`  
> 用途：PR-10 mechanism characterization；trace-on 数据不得作为主要 latency 数字。

## 1. Query record

每行 JSONL 表示一次 bound query：

```text
schema_version
run_id / query_id
bound_method / solver_phase
spec_batch / domain_batch
materialization
state_bytes
events
```

`state_bytes` 固定预留 `alpha_state_bytes`、`beta_state_bytes`、
`intermediate_bound_bytes`、`weight_bytes`、`operator_state_bytes`。PR-10 可以先只优化 coefficient
materialization，但不能把其它状态混入 coefficient bytes。

## 2. Event fields

每个 event 固定包含：

- identity：`run_id`、`query_id`、`event_id`；
- solver：`bound_method`、`solver_phase`；
- site：`operator_site`、`source_value`、`source_primal_op`、`reason`；
- structure：`operator_type`、`operator_tree_depth`、`operator_node_count`；
- axes：`shape`、`dtype`、`device`、`spec_batch`、`domain_batch`；
- memory：`logical_bytes`、`observed_allocation_delta_bytes`；
- lifetime：`persistent_or_ephemeral`、`logical_lifetime_begin/end`、`consumer_count`、
  `reuse_count_estimate`；
- autograd：`requires_grad`、`autograd_saved`、`alpha_related`、`beta_related`。

未知但无法可靠观测的字段写 `null`，不能用猜测值填充。

## 3. 三种内存口径

以下字段不能合并：

1. `logical_materialized_bytes`：event 的 `shape × dtype` 总和，是逻辑 footprint；
2. `observed_allocation_delta_bytes`：trace-on 时事件前后 allocator 差值，只是近似观察；
3. `peak_cuda_allocated_bytes` / `peak_cuda_reserved_bytes`：整次 traced query 在重置 allocator
   peak counter 后的峰值。

`logical_lifetime_*` 是 runtime 推断的逻辑边界，不代表 allocator 的真实释放时刻。

## 4. 测量纪律

- `trace=off`：latency、throughput、正式 peak-memory benchmark；
- `trace=on`：event、site、operator tree、logical lifetime 与机制解释；
- trace-on 会创建 Python 对象并可能重置/读取 CUDA allocator counter，禁止把该延迟作为 headline；
- profile runner 必须把失败、OOM 和 unsupported 也写入 JSONL。

## 5. ReLU dense/structured 事件语义

Dense reference mode 的 `relu_sign_split` 表示主 coefficient 转成 persistent dense state：

```text
persistent_or_ephemeral = persistent
logical_lifetime_begin = relu_backward_step
logical_lifetime_end = backward_end
```

Structured mode 使用 `SignSplitLinearOperator` 保存主 coefficient；bias reduction 使用独立
reason `relu_bias_sign_reduce`，center/norm/contract 使用 `sign_split_*`，全部标记为
`ephemeral`，不能与 dense reference 的 persistent fallback 合并统计。
