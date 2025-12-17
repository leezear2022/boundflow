# BoundFlow 安装指南 (Installation Guide)

本文档详细介绍了 BoundFlow 开发环境的安装步骤。由于本项目依赖自定义版本的 TVM 和 auto_LiRPA，且涉及 C++ 核心库的编译，请严格按照以下步骤操作。

## 1. 前置要求

在开始之前，请确保你的系统满足以下要求：
*   **OS**: Linux (推荐 Ubuntu 20.04/22.04)
*   **Conda**: 已安装 Anaconda 或 Miniconda
*   **CUDA**: 推荐 11.8+ (如果需要 GPU 支持)
*   **Git**: 已安装

## 2. 快速安装 (推荐)

我们提供了一键安装脚本 `scripts/install_dev.sh`，它会自动处理子模块更新、环境创建、C++ 编译和 Python 包安装。

```bash
cd boundflow
# 赋予脚本执行权限
chmod +x scripts/install_dev.sh
# 运行安装脚本
./scripts/install_dev.sh
```

脚本执行过程中会自动激活 `boundflow` 环境进行编译。安装完成后，你需要手动激活环境：

```bash
conda activate boundflow
# 验证安装
python tests/test_env.py
```

如果输出 `Environment verification passed!`，则说明安装成功。

## 3. 手动安装步骤 (用于 Debug)

如果自动脚本失败，可以按照以下步骤手动排查和安装。

### 3.1 获取完整代码
BoundFlow 使用 Git 子模块管理依赖，确保存取了递归子模块：
```bash
git submodule update --init --recursive
# 特别注意：确保 tvm 和 tvm-ffi 内部的子模块也被拉取
cd boundflow/3rdparty/tvm && git submodule update --init --recursive
cd ../tvm-ffi && git submodule update --init --recursive
```

### 3.2 创建 Conda 环境
使用 `environment.yaml` 创建环境，包含 LLVM、CMake、Ninja 等编译工具：
```bash
conda env create -f environment.yaml
conda activate boundflow
```

### 3.3 编译安装 TVM-FFI
`tvm-ffi` 需要编译 C++ shared library 并正确链接。**不要直接使用 `pip install .`**，这可能会因为构建隔离导致找不到库。

推荐方式：
```bash
cd boundflow/3rdparty/tvm-ffi
mkdir -p build && cd build
cmake .. -G Ninja
ninja
# 关键步骤：将编译好的 .so 复制到 python 包源码目录
# Linux 下通常是 core.abi3.so 或 libtvm_ffi.so
find . -name "*.so" -exec cp {} ../python/tvm_ffi/ \;

# 安装 Python 包 (Editable 模式)
cd ../python
pip install -e .
```

### 3.4 编译安装 TVM
TVM 同样需要编译。
```bash
cd boundflow/3rdparty/tvm
mkdir -p build && cd build
cp cmake/config.cmake .

# 编辑 config.cmake 开启必要选项
# set(USE_LLVM ON)
# set(USE_CUDA ON)  <-- 如果有 GPU

cmake .. -G Ninja
ninja

# 安装 Python 包
cd ../python
pip install -e .
```

### 3.5 安装 auto_LiRPA
直接以 editable 模式安装：
```bash
cd boundflow/3rdparty/auto_LiRPA
pip install -e .
```

### 3.6 配置环境 Hooks (可选)
为了开发方便，我们提供了自动配置 PYTHONPATH 的钩子：
```bash
cd boundflow
bash scripts/setup_hooks.sh
```
这会在 `conda activate boundflow` 时自动 source `env.sh`，在 deactivate 时复原。

## 4. 常见问题 (Troubleshooting)

### Q1: `ImportError: No module named 'auto_LiRPA'`
*   **原因**: 没有安装 auto_LiRPA 或者 PYTHONPATH 未设置。
*   **解决**: 运行 `pip install -e boundflow/3rdparty/auto_LiRPA`。

### Q2: `AttributeError: module 'tvm_ffi._ffi_api' has no attribute 'MapGetMissingObject'`
*   **原因**: TVM-FFI 的 Python 包被加载了，但底层的 C++ 动态库 (`.so`) 缺失或版本不匹配（通常是 pip 安装使用了空的/旧的二进制）。
*   **解决**: 按照 3.3 节手动编译 TVM-FFI，并确保 `core.abi3.so` 被复制到了 `python/tvm_ffi/` 目录下。

### Q3: `RuntimeError: Cannot find libtvm.so`
*   **原因**: TVM 编译失败，或者 Python 加载路径不对。
*   **解决**: 检查 `boundflow/3rdparty/tvm/build/libtvm.so` 是否存在。如果存在，检查 `TVM_HOME` 环境变量是否指向了 `boundflow/3rdparty/tvm`。

## 5. 开发指南 (Development Workflow)

### 修改 TVM C++ 代码后如何重编译？
如果你修改了 `boundflow/3rdparty/tvm/src` 下的 C++ 代码，需要重新编译动态库。我们提供了一个便捷脚本：

```bash
./scripts/rebuild_tvm.sh
```

该脚本会增量编译 (Incremental Build)，通常非常快。

### 修改 TVM Python 代码？
由于使用了 `pip install -e` (Editable Mode)，修改 `boundflow/3rdparty/tvm/python` 下的 Python 代码 **不需要重编译/重安装**，修改立即生效。

