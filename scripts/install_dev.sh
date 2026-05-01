#!/bin/bash
set -e  # Exit on error
set -o pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ENV_NAME="boundflow"
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TVM_DIR="${ROOT_DIR}/boundflow/3rdparty/tvm"
TVM_FFI_DIR="${ROOT_DIR}/boundflow/3rdparty/tvm-ffi"
LIRPA_DIR="${ROOT_DIR}/boundflow/3rdparty/auto_LiRPA"

HOST_OS=$(uname -s)
HOST_ARCH=$(uname -m)
ENV_FILE="environment.yaml"
DEFAULT_TVM_USE_CUDA="ON"
DEFAULT_TVM_USE_METAL="OFF"

if [[ "${HOST_OS}" == "Darwin" && "${HOST_ARCH}" == "arm64" ]]; then
    ENV_FILE="environment-macos-arm64.yaml"
    DEFAULT_TVM_USE_CUDA="OFF"
fi

TVM_USE_CUDA="${BOUNDFLOW_TVM_USE_CUDA:-${DEFAULT_TVM_USE_CUDA}}"
TVM_USE_METAL="${BOUNDFLOW_TVM_USE_METAL:-${DEFAULT_TVM_USE_METAL}}"

num_cores() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif [[ "${HOST_OS}" == "Darwin" ]]; then
        sysctl -n hw.ncpu
    else
        getconf _NPROCESSORS_ONLN
    fi
}

CORES=$(num_cores)

echo ">>> BoundFlow Installer"
echo ">>> Root Dir: ${ROOT_DIR}"
echo ">>> Host: ${HOST_OS}/${HOST_ARCH}"
echo ">>> Environment file: ${ENV_FILE}"

# -----------------------------------------------------------------------------
# 1. Submodules
# -----------------------------------------------------------------------------
echo ""
echo ">>> [1/6] Updating Git Submodules (Recursive)..."
git submodule update --init --recursive
# Also update nested submodules specifically to be safe
(cd "${TVM_DIR}" && git submodule update --init --recursive)
(cd "${TVM_FFI_DIR}" && git submodule update --init --recursive)

# -----------------------------------------------------------------------------
# 2. Conda Environment
# -----------------------------------------------------------------------------
echo ""
echo ">>> [2/6] Configuring Conda Environment '${ENV_NAME}'..."
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "    Environment exists. Updating..."
    conda env update -n "${ENV_NAME}" -f "${ROOT_DIR}/${ENV_FILE}"
else
    echo "    Creating new environment..."
    conda env create -f "${ROOT_DIR}/${ENV_FILE}"
fi

# We need to run the rest of the commands inside the conda environment
if [[ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    echo ">>> Switching to conda environment '${ENV_NAME}' for build steps..."
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
fi

if [[ -z "${BOUNDFLOW_TVM_USE_LLVM:-}" ]]; then
    if [[ -x "${CONDA_PREFIX}/bin/llvm-config" ]]; then
        TVM_USE_LLVM="${CONDA_PREFIX}/bin/llvm-config"
    else
        TVM_USE_LLVM="ON"
    fi
else
    TVM_USE_LLVM="${BOUNDFLOW_TVM_USE_LLVM}"
fi

CONDA_PYTHON=$(which python)
echo "    Using Python: ${CONDA_PYTHON}"

# -----------------------------------------------------------------------------
# 3. Build & Install TVM-FFI
# -----------------------------------------------------------------------------
echo ""
echo ">>> [3/6] Building TVM-FFI..."
cd "${TVM_FFI_DIR}"
mkdir -p build
cd build

# Build TVM-FFI with Cython Python module (core.abi3.so)
# -DTVM_FFI_BUILD_PYTHON_MODULE=ON enables the Cython core extension
cmake .. -G Ninja \
    -DTVM_FFI_BUILD_PYTHON_MODULE=ON \
    -DPython_EXECUTABLE="${CONDA_PYTHON}"
ninja

# Copy compiled .so files to the Python package directory
find lib/ -name "*.so" -exec cp {} ../python/tvm_ffi/ \;
# Copy the Cython core module (core.abi3.so)
if [ -f core.abi3.so ]; then
    cp core.abi3.so ../python/tvm_ffi/
    echo "    Copied core.abi3.so to tvm_ffi package"
fi

echo "    Installing TVM-FFI Python package (editable)..."
cd "${TVM_FFI_DIR}"
python -m pip install -e .

# -----------------------------------------------------------------------------
# 4. Build & Install TVM
# -----------------------------------------------------------------------------
echo ""
echo ">>> [4/6] Building TVM..."
cd "${TVM_DIR}"
mkdir -p build
cp cmake/config.cmake build/
cd build
rm -rf CMakeCache.txt CMakeFiles

cat >> config.cmake <<EOF

# BoundFlow installer overrides
set(USE_LLVM "${TVM_USE_LLVM}")
set(USE_CUDA ${TVM_USE_CUDA})
set(USE_METAL ${TVM_USE_METAL})
set(USE_GTEST OFF)
EOF

echo "    Configuring CMake (LLVM=${TVM_USE_LLVM}, CUDA=${TVM_USE_CUDA}, METAL=${TVM_USE_METAL}, GTEST=OFF)..."
cmake .. -G Ninja
ninja -j "${CORES}"

echo "    Installing TVM Python package (editable)..."
cd "${TVM_DIR}"
python -m pip install -e .

# -----------------------------------------------------------------------------
# 5. Install Auto_LiRPA
# -----------------------------------------------------------------------------
echo ""
echo ">>> [5/6] Installing Auto_LiRPA..."
cd "${LIRPA_DIR}"
echo "    Installing Auto_LiRPA Python package (editable)..."
python -m pip install -e .
echo "    Installing BoundFlow Python package (editable)..."
cd "${ROOT_DIR}"
python -m pip install -e .

# -----------------------------------------------------------------------------
# 6. Final Setup
# -----------------------------------------------------------------------------
echo ""
echo ">>> [6/6] Setting up Environment Hooks..."
cd "${ROOT_DIR}"
bash scripts/setup_hooks.sh

echo ""
echo "----------------------------------------------------------------"
echo ">>> Installation Complete!"
echo ">>> Please verify by running:"
echo "    conda activate ${ENV_NAME}"
echo "    python tests/test_env.py"
echo "----------------------------------------------------------------"
