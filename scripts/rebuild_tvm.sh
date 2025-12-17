#!/bin/bash
set -e
set -o pipefail

# Configuration
ENV_NAME="boundflow"
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TVM_DIR="${ROOT_DIR}/boundflow/3rdparty/tvm"

echo ">>> BoundFlow TVM Rebuilder"

# Check if we are in the conda environment
if [[ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    echo ">>> Activating conda environment '${ENV_NAME}'..."
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

echo ">>> Rebuilding TVM..."
cd ${TVM_DIR}/build

# Incremental build using Ninja
ninja

echo ">>> Rebuild Complete!"
echo "    Note: Python changes in 'python/' are reflected immediately (installed as editable)."
echo "    C++ changes require this rebuild."
