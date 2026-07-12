#!/usr/bin/env bash
set -euo pipefail

# TVM 的 HIDE_PRIVATE_SYMBOLS 只能隐藏链接进 libtvm.so 的符号；若 llvm-config
# 返回共享 libLLVM，动态加载器仍会把它与 PyTorch/Triton 的 LLVM 放进同一进程。
case "${1:-}" in
  --libfiles|--libs|--system-libs)
    exec llvm-config --link-static "$@"
    ;;
  *)
    exec llvm-config "$@"
    ;;
esac
