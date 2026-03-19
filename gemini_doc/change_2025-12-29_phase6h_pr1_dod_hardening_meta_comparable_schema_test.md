# 变更记录：Phase 6H PR-1 DoD 补强——meta schema 固化 + 可比性标注 + bench schema 钉子

## 动机

6H PR-1 的目标是 “端到端 time-to-verify 的可归因收益”。为避免 bench 自己引入“假收益/不可比”并提升 reviewer-proof 程度，需要把 DoD 里几个关键口径钉死：

- 计时后端与复现元信息（meta）必须足够完整；
- batch vs serial 若 verdict 不一致，要显式标注不可比；
- bench 输出 schema 要用测试长期守住（比“跑得更快”更容易在 CI 里稳定）。

## 本次改动

- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `rows` 增加：
    - `comparable`：`batch_verdict==serial_verdict` 才为 1
    - `note`：不可比时给出说明（避免误用 speedup）
  - `batch_stats/serial_stats` 增加审计别名：
    - `popped_nodes_total`（= `popped_nodes`）
    - `queue_peak`（= `max_queue_size`）
  - `meta` 固化新增字段：
    - `git_sha`
    - `device_name`
    - `spec_reduce`（当前固定为 `mean`）
    - `torch_num_threads`

- 新增：`tests/test_phase6h_bench_e2e_schema.py`
  - 运行一次 bench（最小参数），断言：
    - `rows==8`
    - `meta` 含关键字段
    - 每行都包含固定 counters 字段（即使为 0）

- 更新：`gemini_doc/change_2025-12-29_phase6h_pr1_e2e_bab_time_to_verify_bench.md`
  - 追加 DoD 补强点与 schema 钉子运行方式。

## 如何验证

```bash
python -m pytest -q tests/test_phase6h_bench_e2e_schema.py
```

