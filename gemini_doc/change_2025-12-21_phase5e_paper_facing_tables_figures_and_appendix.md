# 变更记录：固化 Phase 5E 论文口径产物（tables/figures）+ AE appendix

## 动机

在已有 Phase 5D bench(JSONL schema) 与 postprocess 基础上，把“论文/AE 需要的主表/主图”以**固定命名**与**固定口径**落到脚本与 runner 中，确保后续扩 workload/扩旋钮不破坏复现口径。

## 本次改动

- 更新：`scripts/postprocess_ablation_jsonl.py`
  - 新增 `tables/table_main.csv`：主表最小版本（核心分组键 + plan/cold/hot + bytes_est + call_tir + correctness 缺失计数）。
  - 在 matplotlib 可用时新增示例图：
    - `figures/call_tir_vs_fusion.png`
    - `figures/mem_bytes_vs_runtime.png`
    - `figures/runtime_cold_vs_hot.png`
    - `figures/mem_quadrants.png`（reuse_on × memory_plan_mode 象限视图）
    - `figures/runtime_breakdown.png`（plan/cold/hot 堆叠示例）
- 更新：`scripts/run_phase5d_artifact.py`
  - 将 `table_main.csv` 与新增 figures 复制/重命名为 `artifacts/phase5d/<run_id>/tables/*` 与 `figures/fig_*.png`。
  - 自动打包 AE 文档：`APPENDIX.md`（来自 `gemini_doc/artifact_appendix_phase5d.md`）。
- 更新：`gemini_doc/artifact_claims_phase5d.md`
  - 将新增表/图纳入 claims→证据映射，并补充 memory 象限视图的最小 claim。
- 新增：`gemini_doc/artifact_appendix_phase5d.md`
  - 提供 AE 一键复现说明（环境前置、命令、产物结构、验证方式）。
- 更新：`tests/test_artifact_phase5d_smoke.py`
  - runner 冒烟时额外断言 `tables/table_main.csv` 与 `APPENDIX.md` 存在（TVM 不可用时仍保持 skip）。

## 如何验证

- 无 TVM 环境（最小回归）：`python -m pytest -q tests/test_env_sh_quiet_stdout.py tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_postprocess_enum_normalization.py`
- 有 TVM 环境（artifact 冒烟）：`python -m pytest -q tests/test_artifact_phase5d_smoke.py`
  - 或直接运行：`python scripts/run_phase5d_artifact.py --mode quick --run-id smoke`

