# 变更记录：Phase 5D PR#13D（JSONL schema contract test）

## 动机

PR#13C 已经把 ablation bench 的 JSONL 输出做到了“可解析 + 字段口径可解释 + 去歧义”。为了把它从“约定”升级成真正的 **契约（contract）**，需要在 CI/pytest 中加入 schema contract test：

- 确保输出严格逐行 `json.loads`（每行一个 JSON object）。
- 确保关键字段存在且类型/范围合理，避免后续改字段名/改嵌套导致画图阶段才发现 JSONL 断裂。

## 本次改动

### 1) schema contract test（替代原 smoke 的纯存在性检查）

- 变更：`tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 运行 `scripts/bench_ablation_matrix.py --matrix small --warmup 1 --iters 1 --no-auto-lirpa`
  - 逐行解析 JSONL 并校验：
    - 顶层 `schema_version == "0.1"`
    - `meta.time_utc` 为字符串
    - `runtime.compile_first_run_ms` 与 `runtime.run_ms_{avg,p50,p95}` 为非负数
    - `tvm.compile_cache_stats_delta_compile_first_run` 含 `task_compile_cache_{hit,miss,fail}` 且为非负整数
    - `correctness.python_vs_tvm_max_{abs,rel}_diff_{lb,ub}` 存在且为非负数

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py
```

