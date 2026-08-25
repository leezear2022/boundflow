# 修改记录：MR3 External Provider TVM-FFI Environment Admission

> 日期：2026-08-26
> 对象：`/home/lee/Codes/alpha-beta-CROWN/.venv`（Python 3.11 / torch 2.11）

## 原因

MR3真实provider进程已能导入BoundFlow纯Python hook，但缺少Python 3.11版`tvm_ffi`。BoundFlow
Conda环境中的扩展为CPython 3.12 ABI，禁止跨版本复用。candidate bridge不得因此回退PyTorch。

## 动作

使用系统pip的`--python`显式指定αβ-CROWN现有venv，对本仓库锁定的
`boundflow/3rdparty/tvm/3rdparty/tvm-ffi`执行editable install。该动作只改变venv site-packages/build
cache，不修改αβ-CROWN或auto_LiRPA Git工作树。

## 验证门禁

- `.venv/bin/python`仍为Python 3.11、torch 2.11 CUDA；
- `tvm_ffi.core`必须加载CPython 3.11扩展；
- TVM Python使用本仓库source，动态库使用`build-boundflow`；
- 能导入并编译现有CIBC dense Conv TIR；
- external repos commit/clean状态保持不变；
- 任一失败则MR3 candidate bridge标记environment blocked，不允许eager fallback。
