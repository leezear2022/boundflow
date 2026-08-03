#!/usr/bin/env python3
"""Fetch and verify the pinned VNN-COMP inputs for native IR replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from boundflow.runtime.abcrown_adapter import file_sha256

VNNCOMP_URL = "https://github.com/VNN-COMP/vnncomp2021.git"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
MODEL_RELATIVE_PATH = "benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
VNNLIB_RELATIVE_PATH = (
    "benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/"
    "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
)
VNNLIB_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _run(*command: str) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    """Clone the frozen source repository and verify both workload inputs."""

    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _run("git", "clone", "--filter=blob:none", VNNCOMP_URL, str(output_dir))
    _run("git", "-C", str(output_dir), "checkout", "--detach", VNNCOMP_COMMIT)
    model = output_dir / MODEL_RELATIVE_PATH
    vnnlib = output_dir / VNNLIB_RELATIVE_PATH
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("fetched VNN-COMP model digest differs")
    if file_sha256(vnnlib) != VNNLIB_SHA256:
        raise ValueError("fetched VNN-COMP VNNLIB digest differs")
    print(
        json.dumps(
            {
                "status": "ok",
                "commit": VNNCOMP_COMMIT,
                "model": str(model),
                "model_sha256": MODEL_SHA256,
                "vnnlib": str(vnnlib),
                "vnnlib_sha256": VNNLIB_SHA256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
