# 变更记录：PR#14D（Phase 5E 固化口径）baseline_key + 去重 join + 主表 baseline 列 + speedup 主图

## 动机

PR#14C 已把 auto_LiRPA baseline 写入每行 JSONL，但仍有两类“论文/AE 容易被挑刺”的风险：

1. **缓存误命中**：未来 spec/C 不再固定时，仅靠 `(workload,shape,eps,method,...)` 可能会把不同 spec 的 baseline 误复用。
2. **后处理口径漂移**：baseline 字段在 JSONL 中按矩阵点重复出现，若 postprocess 直接 groupby 平均，会让 baseline 统计隐式依赖矩阵大小（重复计数）。

本次补齐 baseline 的“指纹 + 去重 + 抬进主证据链”：

- 增加 `baseline_key/spec_hash`，避免未来扩 spec 时缓存误命中；
- postprocess 先按 baseline_key 去重，再把 baseline join 回 `table_main.csv`；
- 固化一张 speedup 主图，让 artifact 一跑就能产出可引用的 baseline 对比证据链。

## 本次改动

- 更新：`scripts/bench_ablation_matrix.py`
  - baseline 缓存 key 增加 `spec_hash`，并输出：
    - `baseline.auto_lirpa.baseline_key`：短摘要
    - `baseline.auto_lirpa.spec_hash`：spec 指纹

- 更新：`scripts/postprocess_ablation_jsonl.py`
  - `ablation.csv` 新增：`auto_lirpa_baseline_key/auto_lirpa_spec_hash` 与 abs diff 字段。
  - `tables/table_main.csv`：
    - baseline 相关列从 “逐点重复计数” 改为 “baseline_key 去重后 join”，并优先使用 `cache_hit=false` 的行作为 baseline 来源。
    - 新增：`speedup_hot_vs_auto_lirpa`、`python_vs_auto_lirpa_ok_rate`、`python_vs_auto_lirpa_max_abs_diff_max`。
  - 新增图：`figures/speedup_hot_vs_auto_lirpa_by_workload.png`（matplotlib 可选）。

- 更新：`scripts/run_phase5d_artifact.py`
  - 复制 speedup 图到：`figures/fig_speedup_hot_vs_auto_lirpa_by_workload.png`（best-effort）。

- 更新：`docs/bench_jsonl_schema.md`
  - 增加 `baseline.auto_lirpa.baseline_key/spec_hash` 的字段说明。

- 更新：`gemini_doc/artifact_claims_phase5d.md`
  - 新增最小 baseline claim（对齐 gate + speedup 证据链）。

- 更新测试：
  - `tests/test_phase5d_pr13e_postprocess_jsonl.py`：断言 `ablation.csv` 包含 baseline 列名。
  - `tests/test_phase5d_pr14d_postprocess_baseline_dedup.py`：构造 baseline_key 重复但数值冲突的输入，验证 postprocess 采用去重 baseline（优先 `cache_hit=false`）。

## 如何验证

- `python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_phase5d_pr14d_postprocess_baseline_dedup.py`

