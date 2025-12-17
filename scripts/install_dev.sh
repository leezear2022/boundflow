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
CORES=$(nproc)

echo ">>> BoundFlow Installer"
echo ">>> Root Dir: ${ROOT_DIR}"

# -----------------------------------------------------------------------------
# 1. Submodules
# -----------------------------------------------------------------------------
echo ""
echo ">>> [1/6] Updating Git Submodules (Recursive)..."
git submodule update --init --recursive
# Also update nested submodules specifically to be safe
(cd ${TVM_DIR} && git submodule update --init --recursive)
(cd ${TVM_FFI_DIR} && git submodule update --init --recursive)

# -----------------------------------------------------------------------------
# 2. Conda Environment
# -----------------------------------------------------------------------------
echo ""
echo ">>> [2/6] Configuring Conda Environment '${ENV_NAME}'..."
if conda env list | grep -q "${ENV_NAME}"; then
    echo "    Environment exists. Updating..."
    conda env update -n ${ENV_NAME} -f environment.yaml
else
    echo "    Creating new environment..."
    conda env create -f environment.yaml
fi

# We need to run the rest of the commands inside the conda environment
# We use a recursive call to this script with a flag if we are not already in it
if [[ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    echo ">>> Switching to conda environment '${ENV_NAME}' for build steps..."
    # Execute the build steps in a new shell with the environment activated
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

# -----------------------------------------------------------------------------
# 3. Build & Install TVM-FFI
# -----------------------------------------------------------------------------
echo ""
echo ">>> [3/6] Building TVM-FFI..."
cd ${TVM_FFI_DIR}
mkdir -p build
cd build
# Explicitly build with CMake to ensure .so is generated correctly
cmake .. -G Ninja
ninja
# Copy the compiled library to the python package source directory
# This ensures pip install -e works and finds the library immediately
find . -name "*.so" -exec cp {} ../python/tvm_ffi/ \;
echo "    Installing TVM-FFI Python package (editable)..."
cd ../python
pip install -e .

# -----------------------------------------------------------------------------
# 4. Build & Install TVM
# -----------------------------------------------------------------------------
echo ""
echo ">>> [4/6] Building TVM..."
cd ${TVM_DIR}
mkdir -p build
cp cmake/config.cmake build/
cd build

# Modify config.cmake to enable CUDA and LLVM
# We use sed to enable these options. Adjust based on your system needs.
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/g' config.cmake
sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g' config.cmake
# Optional: Enable CUDNN if needed
# sed -i 's/set(USE_CUDNN OFF)/set(USE_CUDNN ON)/g' config.cmake

echo "    Configuring CMake (LLVM=ON, CUDA=ON)..."
cmake .. -G Ninja
ninja

echo "    Installing TVM Python package (editable)..."
cd ../python
pip install -e .

# -----------------------------------------------------------------------------
# 5. Install Auto_LiRPA
# -----------------------------------------------------------------------------
echo ""
echo ">>> [5/6] Installing Auto_LiRPA..."
cd ${LIRPA_DIR}
echo "    Installing Auto_LiRPA Python package (editable)..."
pip install -e .

# -----------------------------------------------------------------------------
# 6. Final Setup
# -----------------------------------------------------------------------------
echo ""
echo ">>> [6/6] Setting up Environment Hooks..."
cd ${ROOT_DIR}
bash scripts/setup_hooks.sh

echo ""
echo "----------------------------------------------------------------"
echo ">>> Installation Complete!"
echo ">>> Please verify by running:"
echo "    conda activate ${ENV_NAME}"
echo "    python tests/test_env.py"
echo "----------------------------------------------------------------"
