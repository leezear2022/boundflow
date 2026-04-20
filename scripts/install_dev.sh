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
if [[ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    echo ">>> Switching to conda environment '${ENV_NAME}' for build steps..."
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

CONDA_PYTHON=$(which python)
echo "    Using Python: ${CONDA_PYTHON}"

# -----------------------------------------------------------------------------
# 3. Build & Install TVM-FFI
# -----------------------------------------------------------------------------
echo ""
echo ">>> [3/6] Building TVM-FFI..."
cd ${TVM_FFI_DIR}
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

# Create minimal pyproject.toml if missing (required for importlib.metadata
# to resolve the 'apache-tvm-ffi' package name used by tvm_ffi.libinfo)
if [ ! -f ../python/pyproject.toml ]; then
    echo "    Creating pyproject.toml for apache-tvm-ffi..."
    cat > ../python/pyproject.toml <<'PYPROJECT'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "apache-tvm-ffi"
version = "0.0.0"

[tool.setuptools.packages.find]
include = ["tvm_ffi*"]

[tool.setuptools.package-data]
tvm_ffi = ["*.so", "*.dylib", "*.dll"]
PYPROJECT
fi

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

# Enable LLVM and CUDA backends
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/g' config.cmake
sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g' config.cmake

# Disable GTest to avoid issues with system-only static libraries
# (CMake IMPORTED_LOCATION error when only libgtest.a is available)
sed -i 's/set(USE_GTEST AUTO)/set(USE_GTEST OFF)/g' config.cmake

echo "    Configuring CMake (LLVM=ON, CUDA=ON, GTEST=OFF)..."
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
