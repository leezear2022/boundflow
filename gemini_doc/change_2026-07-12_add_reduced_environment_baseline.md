# 变更记录：增加 Gate 0 reduced 环境基线

## 背景

原环境安装脚本只运行 `quick` artifact：每个 MLP/CNN workload 仅 warmup 1 次、计时 1 次，
足以验证链路，但无法判断环境迁移后的计时是否基本稳定。

## 修改

- `scripts/run_phase5d_artifact.py` 新增 `--mode reduced`：使用 small matrix、3 次 warmup、
  10 次计时；保留原有 `quick` 与 `full` 语义。
- `scripts/install_dev.sh baseline` 改用 reduced 档，同时覆盖 MLP 与 MNIST CNN。
- artifact manifest 记录完整顶层命令和 `git_dirty`，且不再把运行后删除的 `_postprocess`
  临时文件列为最终输出。
- `gemini_doc/boundflow_build_and_run_workflow.md` 记录运行命令和证据边界。

## 证据边界

reduced 档是 Gate 0 的稳定性回归，不是论文性能实验。它没有替代总体计划要求的至少 5 次
独立重复，也不能作为 ASPLOS headline result。正式实验仍需不同 run ID、完整配置、失败记录
和置信区间。

## 验证

- `python scripts/run_phase5d_artifact.py --help`
- `bash scripts/install_dev.sh baseline`
- `pytest tests/test_artifact_phase5d_smoke.py`：检查 manifest 顶层命令、dirty 字段和所有最终
  output path 均存在。
- 检查 JSONL、CSV、表格与 manifest，并确认所有行的 correctness/status 门禁。
