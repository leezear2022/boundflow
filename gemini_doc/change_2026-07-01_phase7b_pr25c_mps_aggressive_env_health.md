# 2026-07-01：Phase 7B PR-25C aggressive MPS env OpenMP cleanup

## 背景

PR-25A 使用 `KMP_DUPLICATE_LIB_OK=TRUE` 绕过了 aggressive env 中的 OpenMP duplicate runtime abort。该 workaround 可用于探索，但不能作为可复现实验环境。

旧环境中 `conda list` 显示：

- `llvm-openmp`
- `_openmp_mutex`
- `libopenblas ... openmp`
- conda-forge `numpy/scipy/matplotlib`
- pip `torch==2.12.1`

该组合导致 clean `import torch` 直接 abort。

## 主要改动

- 更新：`environment-macos-arm64-mps-aggressive.yaml`
  - conda 只保留 `python=3.11` 与 `pip`。
  - PyTorch、NumPy、SciPy、Matplotlib、pytest、ONNX 等均改为 pip wheel。
  - 避免 conda-forge 引入 `llvm-openmp/libopenblas-openmp`。
- 新增：`scripts/report_mps_aggressive_env_health.py`
  - 检查 clean `import torch`。
  - 检查 `PYTORCH_MPS_PREFER_METAL=1` 下 clean import。
  - 检查是否仍需 `KMP_DUPLICATE_LIB_OK`。
  - 扫描 conda env 中 openmp/openblas/mkl 相关包。
  - 输出 schema：`mps_aggressive_env_health.v1`。
- 新增：`tests/test_mps_aggressive_env_health_report.py`

## Evidence

执行：

```bash
conda env remove -n boundflow-mps-aggressive -y
conda env create -f environment-macos-arm64-mps-aggressive.yaml
python scripts/report_mps_aggressive_env_health.py \
  --conda-env boundflow-mps-aggressive \
  > out/phase7b/phase7b_pr25c_mps_aggressive_env_health.json
```

结果：

```json
{
  "clean_import_ok": true,
  "kmp_workaround_needed": false,
  "prefer_metal_clean_import_ok": true
}
```

`detected_packages` 为空，说明 conda env 不再引入 openmp/openblas/mkl 包。

额外 smoke：

```bash
conda run --no-capture-output -n boundflow-mps-aggressive \
  python scripts/report_mps_env_var_sweep.py \
  --cases default,prefer_metal \
  --workloads permute_reshape_linear \
  --scales smoke \
  --policies auto \
  --warmup 1 \
  --iters 2
```

通过，且不再需要 `KMP_DUPLICATE_LIB_OK`。

## 结论

Mac aggressive env 已从 workaround 状态转为 clean import 状态。后续 MPS evidence 不应再使用 `KMP_DUPLICATE_LIB_OK=TRUE`，除非是在诊断旧环境。

## 验证

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_mps_aggressive_env_health_report.py \
  tests/test_mps_prefer_metal_guardrails.py
```

结果：`3 passed`
