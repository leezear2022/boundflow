#!/usr/bin/env python
"""Freeze the validated-reduced PR-11 evidence and its experiment contract."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Optional, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "boundflow.pr11-evidence-freeze/v1"
DEFAULT_SOURCES = (
    "pr11-static-v3-agg-calibration-s32-d8-20260712",
    "pr11-static-v3-agg-mini2-calibration-s128-d8-20260712",
    "pr11-static-v3-agg-mini-heldout-s32-d8-20260712",
    "pr11-static-v3-agg-mini-heldout-s128-d8-20260712",
    "pr11-static-v3-agg-branched-heldout-s32-d8-20260712",
    "pr11-static-v3-agg-ridge-factor-loo-20260712",
    "pr11-static-v3-agg-final-default-mini-s32-d8-20260712",
    "pr11-static-v3-agg-final-default-mini-s128-d8-20260712",
    "pr11-static-v3-agg-final-default-branched-s32-d8-20260712",
    "pr11-real-oom-retry-static-v3-final-380mib-20260712",
    "pr11-final-regret-attribution-20260713",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _source_files(source: Path) -> dict[str, str]:
    return {
        str(path.relative_to(_REPO_ROOT)): _sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }


def _workload_hashes(sources: Sequence[Path]) -> dict[str, str]:
    """Hash workload topology independently of measured timings and query scale."""

    signatures: dict[str, dict[str, Any]] = {}
    excluded = {
        "coefficient_bytes",
        "coefficient_elements",
        "domain_batch_size",
        "estimated_dense_flops",
        "spec_batch_size",
    }
    for source in sources:
        raw_path = source / "raw.jsonl"
        if not raw_path.is_file():
            continue
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            workload = row.get("workload")
            barriers = row.get("static_barriers")
            if not isinstance(workload, dict) or not isinstance(barriers, list):
                continue
            name = str(workload["name"])
            signatures[name] = {
                "workload": workload,
                "static_barriers": [
                    {
                        key: value
                        for key, value in barrier.items()
                        if key not in excluded
                    }
                    for barrier in barriers
                ],
            }
    return {
        name: hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for name, payload in sorted(signatures.items())
    }


def _hardware_manifest() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    hardware: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda,
    }
    if cuda:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        hardware.update(
            {
                "device_index": index,
                "device_name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": properties.total_memory,
            }
        )
    try:
        hardware["driver_version"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        hardware["driver_version"] = None
    return hardware


def build_manifest(artifact_root: Path, *, tag: str) -> dict[str, Any]:
    """Build a content-addressed snapshot without modifying source artifacts."""

    artifact_root = artifact_root.resolve()
    sources = [artifact_root / name for name in DEFAULT_SOURCES]
    missing = [str(path) for path in sources if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing frozen evidence sources: {missing}")
    tag_commit = _git("rev-list", "-n", "1", tag)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "tag": tag,
            "tag_commit": tag_commit,
            "dirty_tracked_files": _git(
                "status", "--porcelain", "--untracked-files=no"
            ),
        },
        "schemas": {
            "static_barrier": "boundflow.materialization_static_barrier/v2",
            "profile": "boundflow.pr11-barrier-placement-profile/v3",
            "replicate_aggregation": "independent_profile_median_v1",
            "placement_plan": "boundflow.materialization_placement/v1",
            "cost_model": "boundflow.materialization_placement_cost_model/v3",
            "evaluation": "boundflow.pr11-barrier-placement-eval/v4",
            "regret_attribution": "boundflow.pr11-regret-attribution/v1",
        },
        "experiment_contract": {
            "workload_split": {
                "calibration": [
                    "mlp_chain",
                    "cnn_chain",
                    "residual_block",
                    "add_concat_dag",
                    "mini_resnet2",
                ],
                "held_out": ["mini_resnet", "branched_resnet"],
            },
            "random_seeds": {
                "profile_repetitions": [0, 11, 23],
                "model": 0,
                "input": 1,
            },
            "oracle": (
                "lowest replicated-median latency among measured correct "
                "placements feasible under the same memory budget"
            ),
            "regret": (
                "selected replicated-median latency divided by oracle "
                "replicated-median latency"
            ),
            "correctness": (
                "finite bounds, lower <= upper, and equality to the dense "
                "reference within the recorded tolerance"
            ),
            "high_regret_threshold": 1.5,
        },
        "hardware": _hardware_manifest(),
        "workload_hashes": _workload_hashes(sources),
        "source_hashes": {source.name: _source_files(source) for source in sources},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the immutable evidence snapshot to a new output directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/phase7a-pr11")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", default="pr11-validated-reduced")
    args = parser.parse_args(argv)
    manifest = build_manifest(args.artifact_root, tag=args.tag)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
