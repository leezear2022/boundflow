# 变更记录：PR#14B（Phase 5E 收口）artifact runner 支持多 workload + workload 分色图

## 动机

PR#14A 已将 `mnist_cnn` 以同一 JSONL schema 纳入 bench，但 Phase 5E/AE 仍缺少“**一键 runner 能同时产出 MLP+CNN 的固定口径表/图/manifest**”这一封箱能力。

本次补齐：

- `run_phase5d_artifact.py --workload all`：一键跑多 workload，产出可审计目录结构；
- postprocess 生成一张 workload 分色图，避免 AE/审稿人只看到混合散点无法区分工作负载。

## 本次改动

- 更新：`scripts/run_phase5d_artifact.py`
  - `--workload` 增加 `all`（`mlp`+`mnist_cnn`）。
  - 输出目录新增：`artifacts/phase5d/<run_id>/runs/<workload>/results.jsonl`（每个 workload 一份原始 JSONL）。
  - 同时生成合并后的 `artifacts/phase5d/<run_id>/results.jsonl`（用于统一 postprocess 与外部工具）。
  - `MANIFEST.txt` 记录多条 `bench_command.<workload>`，并将 per-workload JSONL 纳入 outputs/sha256 列表。
  - 额外复制一张 workload 分色图到 `figures/fig_mem_bytes_vs_runtime_by_workload.png`（matplotlib 可选）。

- 更新：`scripts/postprocess_ablation_jsonl.py`
  - 新增 `figures/mem_bytes_vs_runtime_by_workload.png`（`physical_bytes_est` vs `run_ms_p50` 按 `workload` 分色）。

- 更新：`tests/test_artifact_phase5d_smoke.py`
  - quick smoke 默认改为 `--workload all`，并断言：
    - 合并 JSONL 至少包含 `mlp` 与 `mnist_cnn` 两种 workload；
    - `runs/mlp/results.jsonl` 与 `runs/mnist_cnn/results.jsonl` 存在。

## 如何验证

- 无 TVM 环境：`python -m pytest -q tests/test_artifact_phase5d_smoke.py::test_phase5d_artifact_runner_allow_no_tvm_smoke`
- 有 TVM 环境：`python -m pytest -q tests/test_artifact_phase5d_smoke.py::test_phase5d_artifact_runner_quick_smoke`

