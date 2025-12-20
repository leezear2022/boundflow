#!/bin/bash

# Source this file to setup the environment for BoundFlow
# usage: source env.sh

export BOUNDFLOW_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Helper to modify path
add_to_path() {
    if [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
        export PYTHONPATH="$1:$PYTHONPATH"
    fi
}

# Note: With pip install -e, explicit PYTHONPATH is less critical but still helpful for some tools
# Add boundflow root, TVM python package, and auto_LiRPA to PYTHONPATH
add_to_path "${BOUNDFLOW_ROOT}/boundflow/3rdparty/auto_LiRPA"
add_to_path "${BOUNDFLOW_ROOT}/boundflow/3rdparty/tvm-ffi/python"
add_to_path "${BOUNDFLOW_ROOT}/boundflow/3rdparty/tvm/python"
add_to_path "${BOUNDFLOW_ROOT}"

# Set TVM_HOME for compilation
export TVM_HOME="${BOUNDFLOW_ROOT}/boundflow/3rdparty/tvm"

# TVM-FFI optional torch-c-dlpack JIT may be very slow / undesirable for most workflows.
# Default to disabling it; users can override by setting TVM_FFI_DISABLE_TORCH_C_DLPACK=0.
export TVM_FFI_DISABLE_TORCH_C_DLPACK="${TVM_FFI_DISABLE_TORCH_C_DLPACK:-1}"
# Keep caches/temp files inside the repo by default (useful for sandboxed environments).
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-${BOUNDFLOW_ROOT}/.cache/tvm-ffi}"
export TMPDIR="${TMPDIR:-${BOUNDFLOW_ROOT}/.tmp}"
mkdir -p "${TVM_FFI_CACHE_DIR}" "${TMPDIR}" >/dev/null 2>&1 || true

# Avoid polluting stdout (e.g. JSONL/CSV benchmarks). Use stderr unless explicitly silenced.
if [[ -z "${BOUNDFLOW_QUIET:-}" ]]; then
    echo "BoundFlow environment configured." >&2
    echo "TVM_HOME=$TVM_HOME" >&2
fi
