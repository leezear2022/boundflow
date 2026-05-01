# 变更记录：macOS arm64 开发环境启动支持

## 背景

项目迁移到 Apple Silicon Mac 后，原有开发环境入口不能直接复用：

- `environment.yaml` 面向 CUDA/Linux，包含 `pytorch-cuda=12.1` 和 `nvidia` channel；
- `onnxoptimizer` 在 Conda 的 `osx-arm64` 平台不可用，需要走 pip wheel；
- `scripts/install_dev.sh` 使用了 Linux-only `nproc` 和 GNU `sed -i`；
- TVM-FFI 的 editable install 应从 vendored 根目录执行，而不是进入 `python/` 子目录。

## 主要改动

- 新增 `environment-macos-arm64.yaml`，环境名仍为 `boundflow`：
  - 固定 `python=3.10`；
  - 约束 `pytorch>=2.0,<2.9`、`torchvision>=0.15,<0.24`，对齐 vendored Auto_LiRPA 的上限；
  - 固定 `llvmdev=17.0.6`，避开 vendored TVM 与 LLVM 22 API 的不兼容；
  - 移除 CUDA 依赖；
  - 将 `onnxoptimizer==0.4.2`、`onnxruntime==1.19.2`、`pytest-order` 放入 pip 依赖。
- 更新 `scripts/install_dev.sh`：
  - Darwin/arm64 自动选择 macOS 专用环境文件；
  - 用 `sysctl -n hw.ncpu` 兼容 macOS 核心数检测；
  - 通过追加 `config.cmake` override 设置 TVM：macOS 默认 `USE_CUDA=OFF`、LLVM 指向 Conda 环境内 `llvm-config`；
  - TVM 重新配置前清理旧 `CMakeCache.txt/CMakeFiles`，避免切换 LLVM 版本后沿用缓存；
  - TVM-FFI 从 `boundflow/3rdparty/tvm-ffi` 根目录执行 `python -m pip install -e .`；
  - 同步安装 TVM Python、Auto_LiRPA、BoundFlow editable 包。
- 更新 `scripts/rebuild_tvm.sh`：
  - 保留自动激活 `boundflow`；
  - 当 TVM build 目录不存在时给出明确提示，要求先运行 `bash scripts/install_dev.sh`。

## 影响面

- macOS arm64 目标是先跑通 LLVM CPU 后端；不启用 CUDA，也不默认启用 Metal。
- Linux/CUDA 路径继续使用原 `environment.yaml`，并保留默认 `USE_CUDA=ON`。
- 不修改 vendored third-party 源码，只使用其 build 目录和 editable install。

## 验证

- `conda env create -f environment-macos-arm64.yaml --dry-run`：通过，解到 `llvmdev=17.0.6`。
- `bash -n scripts/install_dev.sh scripts/rebuild_tvm.sh`：通过。
- `git diff --check`：通过。
- `bash scripts/install_dev.sh`：通过，完成 Conda 环境创建/更新、TVM-FFI、TVM、Auto_LiRPA、BoundFlow editable 安装。
- `conda run -n boundflow python tests/test_env.py`：通过，PyTorch / Auto_LiRPA / BoundFlow / TVM 均可 import。
- `conda run -n boundflow python -c "import tvm; print(tvm.runtime.enabled('llvm'), tvm.runtime.enabled('cuda'), tvm.runtime.enabled('metal'))"`：输出 `True False False`。
- `conda run -n boundflow python -m pytest -q tests/test_env.py tests/test_env_sh_quiet_stdout.py tests/test_torch_frontend_import.py`：`5 passed, 1 warning in 10.64s`。
- `conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_matches_python.py tests/test_phase4c_tvmexecutor_matches_python_cnn.py tests/test_phase4d_onnx_frontend_matches_torch.py`：`4 passed in 23.08s`。
- `bash scripts/rebuild_tvm.sh`：通过，`ninja: no work to do.`。

备注：首次完整安装曾在 LLVM 22 下失败，错误集中在 `src/target/llvm/llvm_instance.cc` 的 LLVM 22 API 变更；已通过 pin 到 LLVM 17 修正安装入口。测试中仍有 TVM-FFI 可选 `torch-c-dlpack` JIT warning，当前不影响 LLVM/CPU 路径。
