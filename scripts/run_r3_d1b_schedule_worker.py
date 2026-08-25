#!/usr/bin/env python3
"""Run one fresh D1-B fixed-winner isolated timing worker."""

# mypy: disable-error-code=import-untyped
# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

from boundflow.backends.tvm.r3_d1b_serial_schedule import (
    compile_r3d1b_serial_candidate,
    compile_r3d1b_v1_baseline,
)
from scripts.probe_r3_d1b_serial_schedule import (
    _load,
    _measure_candidate,
    RESIDUAL11,
    RESIDUAL6,
)

ROOT = Path(__file__).resolve().parents[1]
WINNER_THREADS = 256


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def run(run_index: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-D1B formal worker requires CUDA")
    raw11 = _load(RESIDUAL11 / "raw/run-00.pt")
    raw6 = _load(RESIDUAL6 / "raw/run-00.pt")
    baseline = compile_r3d1b_v1_baseline()
    candidate = compile_r3d1b_serial_candidate(WINNER_THREADS)
    row = _measure_candidate(torch.cuda.Stream(), baseline, candidate, raw11, raw6)
    props = torch.cuda.get_device_properties(0)
    payload: dict[str, Any] = {
        "schema_version": "boundflow.r3-d1b-schedule-worker/v1",
        "run_index": run_index,
        "source_git_head": _git("rev-parse", "HEAD"),
        "residual11_manifest_sha256": _file_hash(RESIDUAL11 / "manifest.json"),
        "residual6_manifest_sha256": _file_hash(RESIDUAL6 / "manifest.json"),
        "calibration_sha256": _file_hash(
            ROOT / "artifacts/r3-structured-owner/r3-d1b-serial-calibration-v1.json"
        ),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
        },
        "measurement": row,
        "winner_frozen": True,
        "isolated_opportunity_gate": 15.5,
        "isolated_performance_claimed": True,
        "wrapper_performance_claimed": False,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.run_index < 0:
        raise ValueError("R3-D1B run index differs")
    payload = run(args.run_index)
    args.result.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    row = payload["measurement"]
    print(
        f"R3-D1B formal run={args.run_index} speedup={row['speedup']:.4f}x "
        "wrapper_performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
