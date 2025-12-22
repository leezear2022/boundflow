# Phase 5 收尾 DoD（Definition of Done）

本文用于把“Phase 5 是否完成”收敛成可执行、可验证的检查清单；当所有 DoD 通过后，可以宣布 **Phase 5 Done**，进入 Phase 6（更强域/更大规模实验/更科研的优化）。

---

## A. 产线 DoD（论文/AE 视角）

- 一键 runner 可跑通并产出固定目录结构：  
  `python scripts/run_phase5d_artifact.py --mode full --workload all --run-id phase5_full`
- 输出目录 `artifacts/phase5d/<run_id>/` 至少包含：
  - `results.jsonl`（合并 JSONL）
  - `runs/mlp/results.jsonl`、`runs/mnist_cnn/results.jsonl`（逐 workload 原始 JSONL）
  - `results_flat.csv`
  - `tables/table_main.csv`、`tables/table_ablation.csv`
  - `MANIFEST.txt`（含 sha256）
  - `CLAIMS.md`、`APPENDIX.md`
  - `figures/fig_*.png`（若安装了 matplotlib；缺失不应导致 runner 失败）

## B. 统计口径 DoD（不会“静默漂移”）

- `status!="ok"` 的点不会混入主表/主图统计，但失败原因可追溯：
  - JSONL 中含 `status/error`（traceback_hash 可聚合）
  - `tables/table_main.csv` 中含 `n_ok/n_fail`
- `correctness gate` 结果可解释且可汇总：
  - `python_vs_tvm_gate`（bounds 对齐）
  - `python_vs_auto_lirpa_gate`（第三方 baseline 对齐）
  - `tables/table_main.csv` 中含 `python_vs_auto_lirpa_ok_rate` 与 `python_vs_auto_lirpa_max_abs_diff_max`
- baseline 统计不随矩阵大小变化：
  - baseline 写入每行 JSONL，但 postprocess 按 `baseline_key` 去重后 join 回主表（避免“16 点矩阵把 baseline 重复算 16 次”的隐式口径问题）

## C. Baseline DoD（对照可复现/不浪费算力）

- `baseline.auto_lirpa` 必须包含：
  - `available/reason/version/method/device`
  - `init_ms/run_ms_cold/run_ms_p50/run_ms_p95`
  - `baseline_key/spec_hash/cache_hit`
- `tables/table_main.csv` 必须包含：
  - `speedup_hot_vs_auto_lirpa`
  - `figures/fig_speedup_hot_vs_auto_lirpa_by_workload.png`（若 matplotlib 可用）

## D. 可复现 DoD（审稿人/AE 最爱抓的）

- 每行 JSONL 带版本与环境指纹（用于定位“为什么跑出来不一样”）：
  - `meta.git_commit/meta.python/meta.torch/meta.tvm/meta.env_flags/meta.device`
- stdout/stderr 约定不破坏管道：
  - stdout 仅 payload（JSONL/CSV），日志/提示走 stderr（`env.sh` + 回归测试覆盖）

---

## 最小验收动作（建议按顺序）

1) Full artifact：`python scripts/run_phase5d_artifact.py --mode full --workload all --run-id phase5_full`
2) 打开 `artifacts/phase5d/phase5_full/tables/table_main.csv`：
   - 检查 `speedup_hot_vs_auto_lirpa`、`python_vs_auto_lirpa_ok_rate` 是否非空且合理
3) 若需要图：确认环境安装 `matplotlib`（见 `gemini_doc/artifact_appendix_phase5d.md` 的 FAQ）

