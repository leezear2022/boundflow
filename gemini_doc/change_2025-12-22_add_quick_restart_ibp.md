# 修改记录：新增 Quick Restart（IBP 边界快速复跑）（2025-12-22）

## 变更说明
- 新增一份面向“像 auto_LiRPA 一样跑 IBP 边界计算”的 Quick Restart 文档：
  - 覆盖环境启动、自检、bench 一条命令出 JSONL、artifact runner 一键产物、最小 Python API 示例。
- 在 `gemini_doc/README.md` 的“关键交付文档”列表中加入该文档入口，方便交接与索引。

## 修改文件
- 新增：`gemini_doc/quick_restart_ibp.md`
- 更新：`gemini_doc/README.md`

## 影响范围
- 仅文档与索引更新，不涉及运行时逻辑与实验口径变更。

## 验证
- 通过脚本 `--help` 校验命令参数存在（`scripts/bench_ablation_matrix.py`、`scripts/run_phase5d_artifact.py`、`scripts/postprocess_ablation_jsonl.py`）。

