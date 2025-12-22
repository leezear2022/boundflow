# 变更记录：Phase 5 收尾（DoD 清单）+ postprocess 绘图硬化

## 动机

Phase 5 已具备论文/AE 级证据链，但在实际跑 artifact 时仍可能遇到“没有 figures 但也不报错”的困惑点（matplotlib 缺失或 backend 问题）。同时，需要一个明确的 DoD 清单把 Phase 5 的完成标准固化为可执行检查项。

## 本次改动

- 更新：`scripts/postprocess_ablation_jsonl.py`
  - 默认设置 `MPLBACKEND=Agg`（优先 headless backend）。
  - 当绘图失败/缺 matplotlib 时，向 stderr 打印一条明确 warning（不影响 CSV/表/manifest 生成）。
- 更新：`gemini_doc/artifact_appendix_phase5d.md`
  - runner 示例命令改为 `--workload all`（MLP+CNN 同口径）。
  - 增加 FAQ：如何安装 matplotlib/为何可能没有 figures。
- 新增：`gemini_doc/phase5_dod_checklist.md`
  - 固化 Phase 5 Done 的 DoD 与最小验收动作（artifact full + table_main/fig_speedup 检查）。

## 如何验证

- `python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_phase5d_pr14d_postprocess_baseline_dedup.py`
- 运行 artifact 后若无图，应在 stderr 看到 warning，并且 CSV/表仍生成成功。

