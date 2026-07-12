#!/usr/bin/env bash
set -euo pipefail

CUDA_ROOT="${CUDA_ROOT:-/opt/cuda}"
NVCC="${CUDA_ROOT}/bin/nvcc"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

test -x "${NVCC}" || { echo "nvcc 不存在: ${NVCC}" >&2; exit 2; }
printf '%s\n' '__global__ void kernel(int *x) { x[0] = 7; }' \
  'int main() { int *x; cudaMallocManaged(&x, sizeof(int)); kernel<<<1,1>>>(x); cudaDeviceSynchronize(); int ok = (*x == 7); cudaFree(x); return ok ? 0 : 1; }' \
  > "${WORK_DIR}/smoke.cu"
"${NVCC}" -std=c++20 "${WORK_DIR}/smoke.cu" -o "${WORK_DIR}/smoke"
"${WORK_DIR}/smoke"
