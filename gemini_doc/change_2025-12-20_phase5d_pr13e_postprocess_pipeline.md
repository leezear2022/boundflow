# 变更记录：Phase 5D PR#13E（postprocess 产线：JSONL → CSV/表格/图）

## 动机

Phase 5D 的 bench 已经输出统一 JSONL schema。下一步要把数据“变成可用的论文素材”，需要一个官方后处理脚本把 JSONL 转成：

- 可扁平化分析的 CSV（便于 pandas/Excel/gnuplot/jq 等工具链）
- 最小汇总表（按关键旋钮聚合）
-（可选）示例图（不强依赖 matplotlib）

## 本次改动

### 1) 新增后处理脚本

- 新增：`scripts/postprocess_ablation_jsonl.py`
  - 输入：一个或多个 `*.jsonl`（bench 输出）
  - 输出目录：默认 `out/phase5d/`
  - 产物：
    - `out/phase5d/ablation.csv`：扁平化逐点记录
    - `out/phase5d/tables/ablation_summary.csv`：按 `(workload, partition_policy, reuse_on, memory_plan_mode, fusion_on)` 分组汇总
    - `out/phase5d/figures/cache_miss_vs_compile_first_run.png`：示例散点图（若 matplotlib 可用）
    - `out/phase5d/MANIFEST.txt`：输入/行数/产物路径摘要

### 2) 回归测试（不依赖 TVM 编译）

- 新增：`tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 构造最小 JSONL 样例行
  - 调用 `postprocess_ablation_jsonl.main(...)`
  - 断言输出 `ablation.csv` 与 `tables/ablation_summary.csv` 存在

### 3) 文档补充

- 更新：`docs/bench_jsonl_schema.md`
  - 增加“后处理（JSONL→CSV/表格/图）”小节，指向 `scripts/postprocess_ablation_jsonl.py`

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py

# 示例：对 bench 输出做后处理
conda run -n boundflow python scripts/postprocess_ablation_jsonl.py /tmp/boundflow_ablation.jsonl --out-dir /tmp/bf_out
ls -la /tmp/bf_out
```

