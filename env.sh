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

add_to_library_path() {
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":$1:"* ]]; then
        export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}

# PyTorch's pip/conda CUDA wheels keep cuDNN under site-packages instead of the
# system CUDA prefix.  TVM links its cuDNN runtime directly, so expose that
# directory before importing libtvm.  The glob is version-independent and only
# admits an existing conda-environment directory.
if [[ -n "${CONDA_PREFIX:-}" ]] && command -v python >/dev/null 2>&1; then
    boundflow_cudnn_root="$(python -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cudnn")')"
    if [[ -f "${boundflow_cudnn_root}/lib/libcudnn.so.9" ]]; then
        export BOUNDFLOW_CUDNN_ROOT="${boundflow_cudnn_root}"
        add_to_library_path "${boundflow_cudnn_root}/lib"
    fi
    unset boundflow_cudnn_root
fi

# Note: With pip install -e, explicit PYTHONPATH is less critical but still helpful for some tools
# Add boundflow root, TVM python package, and auto_LiRPA to PYTHONPATH
add_to_path "${BOUNDFLOW_ROOT}/boundflow/3rdparty/auto_LiRPA"
add_to_path "${BOUNDFLOW_ROOT}/boundflow/3rdparty/tvm/3rdparty/tvm-ffi/python"
add_to_path "${BOUNDFLOW_ROOT}/boundflow/3rdparty/tvm/python"
add_to_path "${BOUNDFLOW_ROOT}"

# Set TVM_HOME for compilation
export TVM_HOME="${BOUNDFLOW_ROOT}/boundflow/3rdparty/tvm"
export TVM_LIBRARY_PATH="${TVM_HOME}/build-boundflow"
# Newer tvm-ffi discovers its standalone library through LD_LIBRARY_PATH,
# while TVM still accepts TVM_LIBRARY_PATH for libtvm.so.
add_to_library_path "${TVM_LIBRARY_PATH}"
add_to_library_path "${TVM_LIBRARY_PATH}/lib"

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
