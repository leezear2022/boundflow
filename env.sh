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

echo "BoundFlow environment configured."
echo "TVM_HOME=$TVM_HOME"
