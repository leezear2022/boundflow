# Phase 7B PR-23：Mac MPS / Aggressive Lane

**日期**: 2026-07-01

## 背景

当前分支是 `feat/macos-arm64-dev-env`，稳定环境为：

```text
environment-macos-arm64.yaml
torch 2.5.1
auto_LiRPA-compatible: torch < 2.9
```

本机 PyTorch 已支持 MPS：

```text
torch.backends.mps.is_built() == True
torch.backends.mps.is_available() == True
```

PR-23 增加 Mac MPS benchmark support，并新增 aggressive / nightly 环境定义，用于探索更新 PyTorch 的 MPS / Metal 路径。

## 主要改动

### 1. benchmark 支持 `--device mps`

更新：

- `scripts/bench_phase7a_shared_crown_path_attribution.py`
- `scripts/bench_phase7b_crossover_matrix.py`

改动：

- `--device` 支持 `cpu/cuda/mps`。
- MPS timing 使用 `torch.mps.synchronize()`。
- MPS 首版只允许 `--dtype float32`。
- 默认禁止 silent CPU fallback：
  - 若 `PYTORCH_ENABLE_MPS_FALLBACK` 已启用，默认报错。
  - 可显式传 `--allow-mps-fallback` 做 debug，但该结果不能用于 planner promotion。
- benchmark meta 新增：
  - `device_meta.mps_built`
  - `device_meta.mps_available`
  - `device_meta.mps_fallback_env`
  - `device_meta.mps_fallback_enabled`
  - `device_meta.mps_fallback_allowed`

### 2. aggressive / nightly 环境定义

新增：

- `environment-macos-arm64-mps-aggressive.yaml`
- `environment-macos-arm64-mps-nightly.yaml`

`boundflow-mps-aggressive` 目标：

```text
python=3.11
torch==2.12.1
torchvision==0.27.1
```

该环境不安装 / 不强依赖 auto_LiRPA，因为 vendored `auto_LiRPA/setup.py` 当前限制：

```text
torch>=2.0.0,<2.9.0
```

`boundflow-mps-nightly` 只用于 op coverage / Metal feasibility，不用于论文主 benchmark。

### 3. MPS op coverage report

新增：

```text
scripts/report_mps_op_coverage.py
```

输出 schema：

```text
mps_op_coverage.v1
```

报告内容：

- supported / failed workload 数量
- failing exception type / message / traceback hash
- torch version
- macOS platform
- MPS availability
- fallback 是否启用 / 是否允许

## 验证

### 当前稳定环境 MPS smoke

```bash
conda run --no-capture-output -n boundflow python scripts/bench_phase7b_crossover_matrix.py \
  --device mps \
  --dtype float32 \
  --workloads all \
  --scales smoke \
  --policies structured,dense_barrier,auto \
  --warmup 0 \
  --iters 1
```

结果：通过，四个 workload × 三个 policy 均可跑通。

### MPS op coverage

```bash
conda run --no-capture-output -n boundflow python scripts/report_mps_op_coverage.py \
  --workloads all \
  --scales smoke \
  --policy auto \
  --warmup 0 \
  --iters 1
```

结果：

```text
ok = 4
fail = 0
```

### 单测

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase7a_pr11_shared_crown_bench.py \
  tests/test_phase7b_crossover_matrix.py \
  tests/test_mps_op_coverage_report.py
```

结果：

```text
12 passed
```

### 环境文件解析

```bash
conda run -n boundflow python -c "import yaml; ..."
```

结果：`environment-macos-arm64-mps-aggressive.yaml` 与 `environment-macos-arm64-mps-nightly.yaml` 均可解析。

## 未执行项

本轮没有自动创建 `boundflow-mps-aggressive`，原因是这一步会下载新 PyTorch wheel 并改变本机 conda 环境状态。后续可显式执行：

```bash
conda env create -f environment-macos-arm64-mps-aggressive.yaml
```

然后运行正式 aggressive MPS matrix。

## 下一步

PR-24 应创建 aggressive env，并在 `torch==2.12.1` 下跑：

```bash
conda run --no-capture-output -n boundflow-mps-aggressive python scripts/bench_phase7b_crossover_matrix.py \
  --device mps \
  --dtype float32 \
  --workloads all \
  --scales smoke,small,bench \
  --policies structured,dense_barrier,auto \
  --warmup 5 \
  --iters 20
```

只有 high-confidence 且 fallback disabled 的 MPS rules 可以进入 device-specific planner。
