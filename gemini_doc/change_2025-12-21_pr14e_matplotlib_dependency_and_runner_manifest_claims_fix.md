# 变更记录：PR#14E（AE 体验）补齐 matplotlib 依赖 + runner 仅声明实际生成的图

## 动机

用户在 `conda env boundflow` 下运行 `scripts/run_phase5d_artifact.py --mode full` 时可能“没看到 figures”：

- 当前 `environment.yaml` 未包含 `matplotlib`，导致 postprocess 的绘图分支被自动跳过（不失败，但不会产图）。
- 同时 runner 之前会把 `figures/fig_*.png` 无条件写入 `MANIFEST.txt` 的 `paper_facing_outputs`，即使图并未生成，造成“MANIFEST 声明存在但文件缺失”的困惑。

本次把 AE/论文收口体验再硬化一层：默认环境能产图、MANIFEST 不再虚报。

## 本次改动

- 更新：`environment.yaml`
  - 新增 `matplotlib` 依赖（用于 `scripts/postprocess_ablation_jsonl.py` 生成论文示例图）。

- 更新：`scripts/run_phase5d_artifact.py`
  - `_copy_if_exists` 改为返回 bool。
  - `claimed_paths` 仅在实际 copy 成功后才追加，避免 `MANIFEST.txt` 声明不存在的 `figures/fig_*.png`。

- 更新：`tests/test_artifact_phase5d_smoke.py`
  - 解析 `MANIFEST.txt` 的 `paper_facing_outputs`，断言其中列出的每个路径都真实存在（防止 runner 再次“虚报产物”）。

## 如何验证

- 无 TVM 环境：`python -m pytest -q tests/test_artifact_phase5d_smoke.py::test_phase5d_artifact_runner_allow_no_tvm_smoke`
- 有 TVM 环境：`python -m pytest -q tests/test_artifact_phase5d_smoke.py::test_phase5d_artifact_runner_quick_smoke`

