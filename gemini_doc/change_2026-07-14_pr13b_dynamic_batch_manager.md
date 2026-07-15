# 2026-07-14：PR-13B Dynamic BatchManager

## 阶段判定

PR-13B 已关闭 host-side dynamic batching 的 correctness/mechanism foundation：

```text
compatibility bucketing:       PASS
memory-budget first-fit:       PASS
partial/deadline flush:        PASS
deterministic OOM bisection:   PASS（fault injection）
result order/loss audit:       PASS
physical αβ dense batching:    PASS
PR-13B overall:                VALIDATED FOUNDATION
PR-13C same-solver adapter:    NEXT
```

这仍不是性能结论。当前 authoritative 工件是 CPU smoke；OOM 是确定性 fault injection，不是
真实 CUDA capacity 实验。

## 实现

新增 `boundflow/runtime/query_batcher.py`：

- `BatchPolicy(max_batch_size, memory_budget_bytes, max_wait_us, minimum_fill_ratio)`；
- 完整 `QueryCompatibilityKey` 分桶，不允许不同 method、shape、perturbation、execution options、
  dtype/device/numeric policy 或 capability 混批；
- deterministic first-fit memory packing，oversize query 保留 singleton 而不静默丢失；
- fill、timeout、absolute deadline、force 四类 flush；`next_wakeup_us` 暴露 host 下一唤醒时间；
- physical executor OOM 时按稳定中点递归拆批，singleton OOM 原样抛出；
- executor 可乱序返回，但 runtime 按 query ID 恢复原顺序，并拒绝 unknown、duplicate 和 missing
  result；
- 记录 submitted/emitted/completed/pending、batch size/fill、queue-wait p50/p90/p99、execution
  p50/p90/p99、OOM split、flush reason 和 no-loss audit。

新增 `boundflow/runtime/query_executor.py`，把兼容 query 真正合并为一次现有
`run_alpha_beta_crown_mlp(..., per_batch_params=True)` 调用：center、spec、split、α warm start 和
β warm start 分别 pack，并把 bounds/state/branch hint unbatch 回 query ID。αβ query 只接受
`alpha_beta_dense_split` capability，不接 PR-12 plain-CROWN TIR。

Compatibility key 在 PR-13A 基础上补入 `input_value_name`、perturbation signature 和完整
execution-options hash，避免不同 eps、α steps/lr/objective 等语义误合批。

## 测试与工件

新增 `tests/test_phase7a_pr13b_dynamic_batch_manager.py`，覆盖：

- partial bucket timeout；
- cross-capability 分桶和 deadline wakeup；
- memory budget 将 4 queries 稳定拆为 2+2；
- 5-query OOM 按 5→2+3→2+1+2 拆分并恢复顺序；
- duplicate submission 与 missing result 硬失败。

PR-13A fixed replay 又增加 physical αβ batch 对逐节点 reference 的 8/8 对齐。

Authoritative 工件：`artifacts/phase7a-pr13/pr13b-dynamic-batch-v7-20260714/`：

```text
dynamic queries:             8
physical batches:            3（3/3/2）
dynamic replay:              8/8, max abs diff 0
deadline flushes:            1
queue wait p50/p90/p99:      50/75/100 us（deterministic clock）
query loss / invalid result: 0 / 0

fault-injected OOM batches:  8→4+4→2+2+2+2
OOM events/splits:           3/3
fault replay:                8/8, max abs diff 0
```

上述 queue time 来自确定性逻辑时钟，只证明统计和唤醒语义，不是机器性能。

## 下一门禁

PR-13C 只允许做 same-solver adapter：把原 host solver 的 bound-call path 切换为 query runtime，
保持 branch/queue/node order/αβ/split/termination/seed 不变。必须加入 original-vs-runtime 的
per-query/final solver state 对照和 executor non-invocation capability tests。PR-12 Planner 只能在
plain-CROWN 合法 query 上候选；当前 αβ BaB stream 必须继续走 dense batched executor。
