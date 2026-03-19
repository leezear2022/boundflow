# 变更记录：Phase 6H PR-2——sweep 汇总 + 出图出表（JSONL → CSV/fig），闭环“可发表工件”

## 动机

Phase 6H PR-1 已把端到端 time-to-verify 变成 `{meta,rows}` 的 JSON/JSONL 工件，并具备开关矩阵 + 固化计数器 + schema 钉子。

PR-2 的目标是在 **不动 runtime 语义** 的前提下，把这些工件变成“跑 sweep → 自动汇总 → 自动出图/出表”的最后一公里，直接回答 reviewer 的典型追问：

- time-to-verify 降了多少？
- 收益来自 cache / branch-hint / infeasible prune 哪一项？
- batch fill rate 是否足够高？

## 本次改动

### 1) sweep：批量跑并追加 JSONL

- 新增：`scripts/sweep_phase6h_e2e.py`
  - 通过 subprocess 调用 `scripts/bench_phase6h_bab_e2e_time_to_verify.py`，把每次 stdout JSON 作为 **一行 JSONL** 追加写入 `--out-jsonl`。
  - 支持参数列表（逗号分隔）：`devices/dtypes/workloads/ps/specs-list/eps-list/max-nodes-list/node-batch-sizes/oracles/steps-list/lrs/timers`。
  - 支持 `--dry-run` 与 `--max-runs`（便于调试/截断）。

### 2) report：JSONL → CSV + summary.md

- 新增：`scripts/report_phase6h_e2e.py`
  - 读 JSONL，展平成 CSV：**每个 switch 组合 × {batch,serial}** 输出一条记录（便于做表/统计/再出图）。
  - 输出 `summary.md`：只把 `comparable==1` 的行作为主表，其它行单列不可比原因（复用 PR-1 的 `note`）。

### 3) plot：JSONL → 图（主图/归因图/散点）

- 新增：`scripts/plot_phase6h_e2e.py`
  - 读 JSONL，只对 `comparable==1` 的行出图：
    - speedup（8 组合柱状）
    - counters（oracle_calls/forward_trace_calls/pruned_infeasible 折线对照）
    - avg_batch_fill_rate vs speedup（散点，解释吞吐→E2E 的转化）
  - 依赖 `matplotlib`；若环境未安装，测试会 skip（不影响 runtime）。

## 测试

- 新增：`tests/test_phase6h_report_csv_schema.py`
  - 生成最小 JSONL，断言 CSV 列齐全、记录数为 `8*2`，过滤逻辑可用。
- 新增：`tests/test_phase6h_plot_smoke.py`
  - 最小 JSONL 输入能生成 png 图文件（若 `matplotlib` 不可用则 skip）。

## 如何使用

```bash
# 1) sweep 追加 JSONL（示例：只跑一组配置）
python scripts/sweep_phase6h_e2e.py \
  --out-jsonl /tmp/phase6h_e2e.jsonl \
  --devices cpu --dtypes float32 --workloads 1d_relu --ps linf \
  --specs-list 16 --eps-list 1.0 --max-nodes-list 256 --node-batch-sizes 32 \
  --oracles alpha_beta --steps-list 0 --lrs 0.2 --timers perf_counter

# 2) 汇总 CSV + summary.md
python scripts/report_phase6h_e2e.py \
  --in-jsonl /tmp/phase6h_e2e.jsonl \
  --out-csv /tmp/phase6h_e2e.csv \
  --out-summary-md /tmp/phase6h_e2e_summary.md

# 3) 出图（需要 matplotlib）
python scripts/plot_phase6h_e2e.py \
  --in-jsonl /tmp/phase6h_e2e.jsonl \
  --out-dir /tmp/phase6h_figs
```

