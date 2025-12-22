# Phase 5D Artifact Appendix（AE 一键复现说明）

本文是 Phase 5D 的最小可执行 artifact 说明：从环境到命令到产物目录，一次跑通即可生成论文/AE 需要的可审计输出。

---

## 0) 环境前置

- 推荐 conda 环境：`boundflow`
- 进入仓库根目录后：`source env.sh`
- 关键约定：`env.sh` 默认只写 stderr，不污染 stdout（回归：`tests/test_env_sh_quiet_stdout.py`）。

若 TVM 不可用（`import tvm` 失败），Phase 5D runner 会直接报错并退出；请按仓库指引重建 TVM（见 `scripts/install_dev.sh` / `scripts/rebuild_tvm.sh`）。

---

## 1) 一键运行（workflow script）

### Quick（冒烟/CI）

```bash
python scripts/run_phase5d_artifact.py --mode quick --workload all --run-id quick_test
```

### Quick（无 TVM 降级）

```bash
python scripts/run_phase5d_artifact.py --mode quick --workload all --allow-no-tvm --run-id quick_no_tvm
```

### Full（默认矩阵）

```bash
python scripts/run_phase5d_artifact.py --mode full --workload all --run-id full_run
```

输出目录位于：

- `artifacts/phase5d/<run_id>/`

---

## 2) 产物目录结构（固定命名）

- `results.jsonl`：bench 原始 JSONL（1 行 = 1 配置点；schema 见 `docs/bench_jsonl_schema.md`）
- `results_flat.csv`：逐点扁平化 CSV（画图/透视）
- `tables/table_main.csv`：主表最小版本（核心分组键 + plan/cold/hot + bytes_est + call_tir）
- `tables/table_ablation.csv`：汇总表（当前为最小版本；由 `ablation_summary.csv` 复制命名）
- `figures/fig_*.png`：示例图（matplotlib 可选依赖）
- `CLAIMS.md`：claims→证据映射（来自 `gemini_doc/artifact_claims_phase5d.md`）
- `MANIFEST.txt`：运行命令、环境摘要、输出清单（自动生成）
  - 含关键产物 `sha256`，用于确认拷贝/下载后未漂移。

---

## 3) 验证（tests）

- 无 TVM 环境的最小回归（postprocess/env contract）：  
  `python -m pytest -q tests/test_env_sh_quiet_stdout.py tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_postprocess_enum_normalization.py`
- 有 TVM 环境的 artifact 冒烟（会实际跑 quick）：  
  `python -m pytest -q tests/test_artifact_phase5d_smoke.py`

---

## 4) 如何解释“cold vs hot”

- cold：`runtime.compile_first_run_ms`（首次 run 的 wall time，通常包含编译触发）
- hot：`runtime.run_ms_p50/p95`（warmup 后 steady-state）

对应实现与 schema 解释：

- `scripts/bench_ablation_matrix.py`
- `docs/bench_jsonl_schema.md`

---

## 5) 常见问题（FAQ）

### Q: 为什么没有 `figures/fig_*.png`？

如果当前环境未安装 `matplotlib`，后处理会自动跳过绘图但不失败（只生成 CSV/表/manifest）。

- 安装：`conda install -n boundflow -c conda-forge matplotlib`
- 或在最小环境下显式关图：`python scripts/postprocess_ablation_jsonl.py ... --no-plots`
