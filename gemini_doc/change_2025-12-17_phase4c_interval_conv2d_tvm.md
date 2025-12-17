# 变更记录：Phase 4C 增补（TVM interval conv2d kernel + CNN 对齐测试 + 运行统计）

## 动机

之前 `TVMTaskExecutor` 仅对 `linear(w:2D)` 走 TVM kernel：

- CNN 的主要计算量来自 `conv2d`，导致 TVM 后端 demo 覆盖面不足
- 运行时也缺少可观测性（无法直观看到哪些 op 走 TVM、哪些走 fallback）

因此需要补齐：

1. TVM 版 `interval_conv2d`（IBP 公式）
2. CNN 端到端对齐测试：`TVMTaskExecutor == PythonTaskExecutor`
3. 简单的运行统计（便于调试与验证“确实用到了 TVM kernel”）

## 本次改动

### 1) TVM interval conv2d kernel

- 文件：`boundflow/backends/tvm/interval_conv2d.py`
- 核心：对 NCHW 的 `conv2d` 做 IBP
  - `W+ = max(W, 0)`, `W- = min(W, 0)`
  - `y_l = conv2d(x_l, W+) + conv2d(x_u, W-) + b`
  - `y_u = conv2d(x_u, W+) + conv2d(x_l, W-) + b`
- v0 限制：`groups==1`（否则走 fallback）

### 2) TVMTaskExecutor 支持 conv2d + last_stats

- 文件：`boundflow/runtime/tvm_executor.py`
- 新增：
  - `conv2d` 优先走 TVM kernel（不满足条件则 fallback 到 `IntervalDomain`）
  - `last_stats`：记录
    - `tvm_ops`：本次 run 中走 TVM kernel 的 op 类型
    - `fallback_ops`：本次 run 中走 Python/torch 的 op 类型
    - `linear_kernel_cache / conv2d_kernel_cache`：kernel 编译缓存命中信息（进程内）

### 3) 新增 CNN 对齐测试

- 文件：`tests/test_phase4c_tvmexecutor_matches_python_cnn.py`
- 用例：MNIST-style CNN
- 断言：
  - `TVMTaskExecutor` 输出与 `PythonTaskExecutor` allclose
  - `last_stats.tvm_ops` 中至少包含一次 `conv2d`（确保该用例确实覆盖 TVM conv2d kernel）

### 4) 基准脚本（可选但便于观察）

- 文件：`scripts/bench_phase4c_tvmexecutor.py`
- 用途：简单对比 `PythonTaskExecutor` vs `TVMTaskExecutor` 的平均耗时，并打印本次 run 的 `last_stats`

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_matches_python_cnn.py
conda run -n boundflow python -m pytest -q
```

（可选）跑基准：

```bash
conda run -n boundflow python scripts/bench_phase4c_tvmexecutor.py --model cnn --target llvm
```

