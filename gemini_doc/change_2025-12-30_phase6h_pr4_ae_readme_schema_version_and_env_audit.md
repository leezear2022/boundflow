# 变更记录：Phase 6H PR-4（AE 打包准备）——AE README + claims 映射 + schema_version + 环境审计

## 动机

在 6H PR-1/2/3 的基础上，PR-4 目标是把“可复现流水线”进一步贴近 AE 的交付形态：

- 提供 **Getting Started（≤30min）** 与 **Claim→产物** 映射表；
- bench 输出增加 `schema_version`，避免未来字段扩展导致 report/plot 脚本 silent break；
- runner 输出环境审计信息（threads/env/pip/conda），降低 “fresh machine” 的排错成本。

## 本次改动

- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `meta.schema_version="phase6h_e2e_v1"`。
- 更新：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - `meta.schema_version="phase6g_node_eval_v1"`（为跨阶段工件统一口径做准备）。
- 更新：`tests/test_phase6h_bench_e2e_schema.py`
  - 断言 `meta.schema_version` 存在。
- 更新：`scripts/run_phase6h_artifact.sh`
  - 生成 `env.txt/pip_freeze.txt/conda_list.txt` 作为 best-effort 环境审计。
- 新增：`gemini_doc/ae_readme_phase6h.md`
  - Kick-the-tires（≤30min）+ Claims 映射 + 产物路径 + 限制说明。

## 如何验证

```bash
python -m pytest -q tests/test_phase6h_bench_e2e_schema.py
bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run
```

