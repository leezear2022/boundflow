# 变更记录：bench_ablation_matrix 硬化（失败行/口径指纹/cold-hot）+ runner 降级模式

## 动机

进一步对齐 AE/系统论文的“可复现/可审计/可定位”要求，避免矩阵实验出现：

- 某点失败导致整次运行退出或静默丢点；
- correctness gate 口径不透明（阈值/参考源不明确）；
- cold/hot 计时解释不清；
- 环境指纹不足导致复现排障成本高；
- 无 TVM 环境无法至少跑通 pipeline（schema+postprocess+tables）。

## 本次改动

### Bench（PR#13A 方向的补钉子）

- 更新：`scripts/bench_ablation_matrix.py`
  - **失败也写一行 JSONL**：新增顶层 `status`/`error` 字段；每个矩阵点 try/except，保证不丢点。
  - **correctness gate 显式化**：新增 `correctness.python_vs_tvm_gate` / `python_vs_auto_lirpa_gate`，包含 `ref/ok/tol/max_*`。
  - **cold/hot 更清晰**：新增 `runtime.run_ms_cold`（编译触发后的一次冷运行，不含编译），保留原有 `compile_first_run_ms` 与 `run_ms_p50/p95`。
  - **版本/环境指纹**：`meta.device`（cuda 指纹）与 `meta.env_flags`（关键环境变量快照）。
  - **不再缓存所有 runs 再输出**：改为逐点写 JSONL，便于长跑矩阵与中途诊断。
  - 增加 `--no-tvm`（python-only）与 `--exit-nonzero-on-fail`（可选严格退出）。

### Postprocess（过滤失败点避免静默口径错误）

- 更新：`scripts/postprocess_ablation_jsonl.py`
  - `ablation_summary.csv` / `table_main.csv` 的均值统计只使用 `status="ok"` 的记录，并增加 `n_ok/n_fail`。
  - 图表也只绘制 `status="ok"`，避免失败点被当作 0 画进图里。
  - `ablation.csv` 额外包含 `status/error_type/error_msg` 便于排障。

### Runner（AE 视角改进）

- 更新：`scripts/run_phase5d_artifact.py`
  - 新增 `--allow-no-tvm`：TVM 不可用时自动调用 bench 的 `--no-tvm`，仍产出 JSONL/CSV/表/MANIFEST（但 TVM 相关 claims 不覆盖）。
  - `MANIFEST.txt` 增加关键产物 `sha256`，用于确认拷贝/下载后未漂移。
  - 增加 repo root 注入 `sys.path`，确保以 `python scripts/run_phase5d_artifact.py ...` 形式运行时可正常 import。
  - 默认设置 `MPLBACKEND=Agg`，避免 headless 环境下 Qt/Wayland backend 警告污染输出。

### 文档与测试

- 更新：`docs/bench_jsonl_schema.md`
  - 补充 `status/error`、`meta.device/env_flags`、`runtime.run_ms_cold`、`tables/table_main.csv` 等说明。
- 更新：`tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - TVM 相关 contract 在无 TVM 环境下自动 skip；新增 `--no-tvm` 模式的最小断言，保证“失败不丢点/可降级跑通”。
- 更新：`tests/test_artifact_phase5d_smoke.py`
  - 新增无 TVM 环境下的 runner 冒烟（仅在 TVM 缺失时运行，避免 TVM 环境重复编译成本）。
- 更新：`tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 断言 `tables/table_main.csv` 存在。
- 更新：`gemini_doc/artifact_claims_phase5d.md`、`gemini_doc/artifact_appendix_phase5d.md`
  - 增加 `--allow-no-tvm` 的运行说明与边界声明。

## 如何验证

- 相关测试（无 TVM 环境会自动 skip TVM 路径）：  
  `python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_artifact_phase5d_smoke.py`
