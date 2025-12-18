# 变更记录：禁用 TVM-FFI 可选 torch-c-dlpack JIT（加速 tvm import / pytest）

## 动机

`tvm` import 时会通过 `tvm-ffi` 触发一个“可选的 torch-c-dlpack 扩展 JIT 编译”。该扩展用于更快的 DLPack 转换，但对 BoundFlow 当前阶段（IR/planner/IBP/TVM kernel demo）不是必需依赖。

在某些环境（例如受限的 sandbox、或无编译工具链/缓存目录不可写）下，这个 JIT 会显著拖慢甚至卡住 `import tvm`，导致 `pytest` 超时。

因此我们默认禁用该 JIT，并把 cache/tmp 目录指向 repo 内（可控且便于清理）。如需启用，用户可手动覆盖环境变量。

## 本次改动

- 修改：`env.sh`
  - 默认设置 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`（用户可通过显式设置为 0 覆盖）
  - 默认设置 `TVM_FFI_CACHE_DIR=$BOUNDFLOW_ROOT/.cache/tvm-ffi`
  - 默认设置 `TMPDIR=$BOUNDFLOW_ROOT/.tmp` 并创建目录

## 说明：这是“受限环境默认策略”，不是唯一推荐路径

上面的默认行为主要面向以下场景：CI/WSL/sandbox 等“编译工具链不完整、网络受限、缓存目录不可写或不可持久化”的环境，目标是保证 `import tvm` 与 `pytest` **可稳定完成**。

如果你希望启用更快的 torch↔DLPack 转换（并且避免每次现场编译/JIT），更“正规”的两条路径是：

1) 安装可选依赖 `torch-c-dlpack-ext`（推荐）

这会让 tvm-ffi 不需要在 import 时现场编译扩展：

```bash
pip install torch-c-dlpack-ext
```

2) 设置 `TVM_FFI_CACHE_DIR` 到可写且持久的路径（避免反复编译）

```bash
export TVM_FFI_CACHE_DIR=~/.cache/tvm-ffi
```

## 如何验证

```bash
conda activate boundflow
source env.sh
python -c "import tvm; print('tvm_ok')"
pytest -q
```

## 如何启用 torch-c-dlpack JIT（可选）

如果你没有安装 `torch-c-dlpack-ext`，但确实希望启用该路径（可能触发一次性编译/JIT），可以：

```bash
export TVM_FFI_DISABLE_TORCH_C_DLPACK=0
```
