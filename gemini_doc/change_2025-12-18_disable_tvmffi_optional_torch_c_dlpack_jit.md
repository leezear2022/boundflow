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

## 如何验证

```bash
conda activate boundflow
source env.sh
python -c "import tvm; print('tvm_ok')"
pytest -q
```

## 如何启用 torch-c-dlpack JIT（可选）

```bash
export TVM_FFI_DISABLE_TORCH_C_DLPACK=0
```

