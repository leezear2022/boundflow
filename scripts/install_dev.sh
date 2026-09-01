#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${BOUNDFLOW_ENV_NAME:-boundflow}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TVM_DIR="${ROOT_DIR}/boundflow/3rdparty/tvm"
TVM_BUILD_DIR="${TVM_DIR}/build-boundflow"
TVM_FFI_DIR="${TVM_DIR}/3rdparty/tvm-ffi"
LIRPA_DIR="${ROOT_DIR}/boundflow/3rdparty/auto_LiRPA"
CONDA_BIN="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"
CUDA_ROOT="${BOUNDFLOW_CUDA_ROOT:-}"
JOBS="${BOUNDFLOW_BUILD_JOBS:-$(nproc)}"

usage() {
  echo "用法: $0 <audit|submodules|conda|pytorch|cuda-smoke|tvm|auto-lirpa|verify|baseline|all>" >&2
  echo "首次搭建建议逐阶段执行；脚本不使用 sudo，也不修改驱动、内核或系统 CUDA。" >&2
}

run_env() {
  local pythonpath="${ROOT_DIR}:${TVM_DIR}/python:${TVM_FFI_DIR}/python:${LIRPA_DIR}"
  local cudnn_library_path=""
  if [[ -n "${BOUNDFLOW_CUDNN_ROOT:-}" ]]; then
    cudnn_library_path="${BOUNDFLOW_CUDNN_ROOT}/lib:"
  fi
  PYTHONPATH="${pythonpath}${PYTHONPATH:+:${PYTHONPATH}}" \
    TVM_LIBRARY_PATH="${TVM_BUILD_DIR}" \
    LD_LIBRARY_PATH="${cudnn_library_path}${TVM_BUILD_DIR}/lib:${TVM_BUILD_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    BOUNDFLOW_QUIET=1 \
    "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" "$@"
}

run_env_capture() {
  BOUNDFLOW_QUIET=1 "${CONDA_BIN}" run -n "${ENV_NAME}" "$@"
}

cuda_root() {
  if [[ -n "${CUDA_ROOT}" ]]; then
    printf '%s\n' "${CUDA_ROOT}"
  elif command -v nvcc >/dev/null 2>&1; then
    dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")"
  else
    local conda_cuda_root
    conda_cuda_root="$(run_env_capture python -c 'import os; print(os.environ["CONDA_PREFIX"])')"
    if [[ ! -f "${conda_cuda_root}/include/cuda_runtime.h" ]]; then
      echo "找不到CUDA开发头文件；请设置BOUNDFLOW_CUDA_ROOT。" >&2
      exit 2
    fi
    printf '%s\n' "${conda_cuda_root}"
  fi
}

cudnn_root() {
  if [[ -n "${BOUNDFLOW_CUDNN_ROOT:-}" ]]; then
    printf '%s\n' "${BOUNDFLOW_CUDNN_ROOT}"
  else
    run_env_capture python -c 'import pathlib, sysconfig; root = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cudnn"; assert (root / "include" / "cudnn.h").is_file() and (root / "lib" / "libcudnn.so.9").is_file(), root; print(root)'
  fi
}

require_conda() {
  if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "找不到 conda: ${CONDA_BIN}；请设置 CONDA_EXE。" >&2
    exit 2
  fi
}

stage_audit() {
  python "${ROOT_DIR}/scripts/env_doctor.py" --json-out "${ROOT_DIR}/artifacts/env/host-doctor.json"
}

stage_submodules() {
  git -C "${ROOT_DIR}" submodule update --init --recursive
  test -f "${TVM_FFI_DIR}/pyproject.toml"
}

stage_conda() {
  require_conda
  if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.yaml" --prune
  else
    "${CONDA_BIN}" env create -f "${ROOT_DIR}/environment.yaml"
  fi
}

stage_pytorch() {
  require_conda
  run_env python -m pip install -r "${ROOT_DIR}/requirements-pytorch-cu132.txt"
  run_env python -c 'import torch, torchvision; assert torch.__version__.split("+")[0] == "2.12.1"; assert torchvision.__version__.split("+")[0] == "0.27.1"; assert torch.version.cuda == "13.2", torch.version.cuda'
}

stage_cuda_smoke() {
  require_conda
  CUDA_ROOT="$(cuda_root)" run_env bash "${ROOT_DIR}/scripts/smoke_cuda.sh"
}

stage_tvm() {
  require_conda
  test -f "${TVM_FFI_DIR}/pyproject.toml" || stage_submodules
  local llvm_config
  local selected_cuda_root
  local selected_cudnn_root
  local clang
  local clangxx
  # 静态链接 LLVM 后再由 HIDE_PRIVATE_SYMBOLS 隐藏，避免与 Triton 的 LLVM 冲突。
  llvm_config="${ROOT_DIR}/scripts/llvm-config-static.sh"
  selected_cuda_root="$(cuda_root)"
  selected_cudnn_root="$(cudnn_root)"
  clang="$(run_env_capture which clang)"
  clangxx="$(run_env_capture which clang++)"
  BOUNDFLOW_CUDNN_ROOT="${selected_cudnn_root}" run_env cmake -S "${TVM_DIR}" -B "${TVM_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${clang}" \
    -DCMAKE_CXX_COMPILER="${clangxx}" \
    -DCMAKE_CUDA_HOST_COMPILER="${clangxx}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DUSE_LLVM="${llvm_config}" \
    -DUSE_CUDA="${selected_cuda_root}" \
    -DUSE_CUBLAS=ON \
    -DUSE_CUDNN="${selected_cudnn_root}" \
    -DCUDA_CUDNN_LIBRARY="${selected_cudnn_root}/lib/libcudnn.so.9" \
    -DHIDE_PRIVATE_SYMBOLS=ON
  BOUNDFLOW_CUDNN_ROOT="${selected_cudnn_root}" run_env cmake --build "${TVM_BUILD_DIR}" --parallel "${JOBS}"
  # Python 与 TVM C++ 构建都来自TVM锁定的同一个内嵌tvm-ffi commit。已有editable
  # 安装可直接复用，避免每次TVM增量构建都在pip隔离环境里重复全量编译FFI。
  if ! BOUNDFLOW_CUDNN_ROOT="${selected_cudnn_root}" run_env python -c 'import tvm_ffi' >/dev/null 2>&1; then
    BOUNDFLOW_CUDNN_ROOT="${selected_cudnn_root}" run_env python -m pip install --no-build-isolation --no-deps -e "${TVM_FFI_DIR}"
  fi
}

stage_auto_lirpa() {
  require_conda
  test -f "${LIRPA_DIR}/setup.py" || stage_submodules
  run_env python -m pip install --no-deps -e "${LIRPA_DIR}"
}

stage_verify() {
  require_conda
  local selected_cudnn_root
  selected_cudnn_root="$(cudnn_root)"
  run_env bash "${ROOT_DIR}/scripts/setup_hooks.sh"
  run_env python "${ROOT_DIR}/scripts/env_doctor.py" --strict --json-out "${ROOT_DIR}/artifacts/env/boundflow-doctor.json"
  BOUNDFLOW_CUDNN_ROOT="${selected_cudnn_root}" run_env python -c 'import tvm; import triton; required=("relax.ext.cublas", "tvm.contrib.cublas.matmul", "relax.ext.cudnn", "tvm.contrib.cudnn.conv2d.forward"); missing=[name for name in required if tvm.get_global_func(name, allow_missing=True) is None]; assert not missing, missing'  # LLVM/符号隔离与S1/S2混合后端门禁
  run_env python "${ROOT_DIR}/scripts/smoke_tvm_cuda.py"
  run_env python -m pytest -q "${ROOT_DIR}/tests"
}

stage_baseline() {
  require_conda
  run_env python "${ROOT_DIR}/scripts/run_phase5d_artifact.py" \
    --mode reduced --workload all \
    --run-id "env-cu132-$(date +%Y%m%d)" \
    --out-root "${ROOT_DIR}/artifacts/environment-baseline"
}

stage_all() {
  stage_audit
  stage_submodules
  stage_conda
  stage_pytorch
  stage_cuda_smoke
  stage_tvm
  stage_auto_lirpa
  stage_verify
  stage_baseline
}

if [[ $# -ne 1 ]]; then usage; exit 2; fi
case "$1" in
  audit) stage_audit ;;
  submodules) stage_submodules ;;
  conda) stage_conda ;;
  pytorch) stage_pytorch ;;
  cuda-smoke) stage_cuda_smoke ;;
  tvm) stage_tvm ;;
  auto-lirpa) stage_auto_lirpa ;;
  verify) stage_verify ;;
  baseline) stage_baseline ;;
  all) stage_all ;;
  *) usage; exit 2 ;;
esac
