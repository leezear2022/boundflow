# 变更记录：Phase 5D artifact runner + claims 口径 + smoke 测试

## 动机

将 Phase 5D 已完成的 bench(JSONL schema) 与 postprocess 产线“封箱”为 AE 友好的一键入口，并提供可审计的 MANIFEST 与 claims→证据映射，避免后续扩 workload/扩旋钮时反复改口径。

## 本次改动

- 新增：`scripts/run_phase5d_artifact.py`
  - 一键执行：bench → postprocess → 产出固定目录结构（`results.jsonl`、`results_flat.csv`、`tables/table_ablation.csv`、`figures/*`、`MANIFEST.txt`）。
  - 默认设置可复现相关环境变量（`PYTHONHASHSEED/线程数/BOUNDFLOW_QUIET` 等）。
  - 若 TVM 不可 import，会给出明确错误并返回非 0（Phase 5D 需要 TVM 的 compile/cache 统计）。
- 新增：`scripts/run_phase5d_artifact.sh`
  - 作为薄封装，便于 AE/用户直接运行。
- 更新：`scripts/postprocess_ablation_jsonl.py`
  - 在可选 matplotlib 环境下，额外生成 3 张最小示例图：`call_tir_vs_fusion.png`、`mem_bytes_vs_runtime.png`、`runtime_cold_vs_hot.png`（保持 matplotlib 缺失时不失败的策略）。
- 新增：`gemini_doc/artifact_claims_phase5d.md`
  - 固定 Phase 5D 的最小 claims/metrics/threats，并给出命令与产物字段映射。
- 新增：`tests/test_artifact_phase5d_smoke.py`
  - 运行 runner 的 `--mode quick` 冒烟；当 TVM 不可用时自动 skip（与项目中对可选依赖的测试风格一致）。
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 将 “artifact runner 未落地” 的 TODO 状态更新为“已提供最小一键 runner”，并保留后续定稿化口径的 TODO。

## 如何验证

- 无 TVM 环境（最小回归）：`python -m pytest -q tests/test_env_sh_quiet_stdout.py tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_postprocess_enum_normalization.py`
- 有 TVM 环境（artifact 冒烟）：`python -m pytest -q tests/test_artifact_phase5d_smoke.py`
  - 或直接运行：`python scripts/run_phase5d_artifact.py --mode quick --run-id smoke`
