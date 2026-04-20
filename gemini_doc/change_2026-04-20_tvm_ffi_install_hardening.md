# 变更记录：2026-04-20 TVM-FFI 安装脚本加固

**日期**: 2026-04-20  
**类型**: chore / env / install  
**范围**: `environment.yaml`、`scripts/install_dev.sh`

---

## 动机

当前开发环境里，TVM-FFI 的 Python 侧安装存在两个脆弱点：

- 只编了通用 `.so`，没有显式开启 Cython Python module，容易缺 `core.abi3.so`
- `tvm_ffi.libinfo` 依赖 `importlib.metadata` 解析包名 `apache-tvm-ffi`；如果 editable 安装目录里缺最小 `pyproject.toml`，元数据解析会失败

另外，部分机器上 TVM 顶层编译还会因为系统只提供静态 `gtest` 而在 CMake 配置阶段报错，需要把这个坑在安装脚本里直接绕开。

## 主要改动

- 更新：`environment.yaml`
  - 新增 `cython>=3.0`
  - 用于构建 TVM-FFI 的 Cython core module

- 更新：`scripts/install_dev.sh`
  - 显式记录并使用当前 conda 环境里的 `python`
  - TVM-FFI CMake 配置增加：
    - `-DTVM_FFI_BUILD_PYTHON_MODULE=ON`
    - `-DPython_EXECUTABLE="${CONDA_PYTHON}"`
  - 复制 `lib/*.so` 之外，再显式复制 `core.abi3.so` 到 `python/tvm_ffi/`
  - 若 `boundflow/3rdparty/tvm-ffi/python/pyproject.toml` 缺失，则动态生成最小版本，保证 editable 安装后能解析 `apache-tvm-ffi` 元数据
  - TVM 顶层构建把 `USE_GTEST` 从 `AUTO` 改为 `OFF`，避免仅有系统静态库时的 CMake `IMPORTED_LOCATION` 报错

## 结果

- `scripts/install_dev.sh` 对 TVM-FFI Python 模块的假设更明确，不再依赖隐式副产物恰好落在对的位置。
- 对不同机器环境更稳：
  - 绑定当前 conda Python
  - 缺 `pyproject.toml` 时自动补齐
  - 避开 GTest 配置坑

## 影响面

- 不改 runtime / solver 语义。
- 不升级 vendored submodule 版本。
- 主要影响首次安装和本机重建流程。

## 验证

已执行：

```bash
bash -n scripts/install_dev.sh
```

结果：

- Shell 语法检查通过。
- 本轮未在干净环境里重新完整跑一遍 `scripts/install_dev.sh`；若后续需要，可再做一次全流程安装验证。
