#!/usr/bin/env python
"""Freeze the PR-12 base reference and pre-implementation held-out split."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Optional, Sequence

import torch

BASE_TAG = "pr11-validated-reduced"
BASE_COMMIT = "fee6cc0d3229ff19229ebfb239ce46cc42e1cab5"
START_SCHEMA_VERSION = "boundflow.pr12-start/v1"
HELDOUT_SPLIT_ID = "pr12-final-heldout-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _hardware() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "device_name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": properties.total_memory,
            }
        )
    return payload


def _heldout_split() -> dict[str, Any]:
    return {
        "schema_version": "boundflow.pr12-heldout-split/v1",
        "split_id": HELDOUT_SPLIT_ID,
        "frozen_before_kernel_implementation": True,
        "development": {
            "source": "pr11-final-regret-attribution-20260713",
            "backend_gap_case_count": 7,
            "purpose": "motivation_and_debug_only",
        },
        "calibration": [
            {"family": "linear", "domain": 2, "spec": 8, "current": 16, "previous": 12},
            {
                "family": "linear",
                "domain": 4,
                "spec": 32,
                "current": 64,
                "previous": 48,
            },
            {
                "family": "conv2d",
                "domain": 2,
                "spec": 8,
                "channels": 8,
                "height": 16,
                "width": 16,
            },
        ],
        "final_heldout": [
            {
                "case_id": "linear-unseen-shape-a",
                "family": "linear",
                "domain": 3,
                "spec": 17,
                "current": 29,
                "previous": 13,
                "budget_mib": 256,
            },
            {
                "case_id": "linear-unseen-shape-b",
                "family": "linear",
                "domain": 5,
                "spec": 65,
                "current": 127,
                "previous": 61,
                "budget_mib": 192,
            },
            {
                "case_id": "linear-memory-sensitive",
                "family": "linear",
                "domain": 8,
                "spec": 257,
                "current": 1024,
                "previous": 512,
                "budget_mib": 64,
            },
            {
                "case_id": "conv-unseen-width",
                "family": "conv2d",
                "domain": 5,
                "spec": 48,
                "channels": 12,
                "height": 16,
                "width": 16,
                "kernel": 3,
                "budget_mib": 512,
            },
            {
                "case_id": "mini-resnet-unseen-width",
                "family": "mini_resnet",
                "domain": 5,
                "spec": 48,
                "width": 12,
                "blocks": 3,
                "budget_mib": 768,
            },
        ],
        "policy": "final_heldout is evaluation-only; changing it requires a new split_id",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write baseline, Planner reference, and held-out manifests once."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tests-file", type=Path, required=True)
    parser.add_argument(
        "--planner-freeze-manifest",
        type=Path,
        default=Path(
            "artifacts/phase7a-pr11/pr11-validated-reduced-freeze-20260713/manifest.json"
        ),
    )
    args = parser.parse_args(argv)
    if _git("rev-list", "-n", "1", BASE_TAG) != BASE_COMMIT:
        raise RuntimeError("PR-11 base tag moved; refusing to create PR-12 baseline")
    for path in (args.tests_file, args.planner_freeze_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heldout = _heldout_split()
    heldout_path = args.out_dir / "heldout_split.json"
    heldout_path.write_text(
        json.dumps(heldout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    planner_ref = {
        "base_tag": BASE_TAG,
        "base_commit": BASE_COMMIT,
        "planner_model": "pr11-v1-frozen",
        "candidate_set": ["dense_eager", "structured_eager", "reduce_batch"],
        "freeze_manifest": str(args.planner_freeze_manifest),
        "freeze_manifest_sha256": _sha256(args.planner_freeze_manifest),
    }
    planner_path = args.out_dir / "planner_freeze_ref.json"
    planner_path.write_text(
        json.dumps(planner_ref, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": START_SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_tag": BASE_TAG,
        "base_commit": BASE_COMMIT,
        "branch_head_at_generation": _git("rev-parse", "HEAD"),
        "planner_model": "pr11-v1-frozen",
        "pr11_candidate_set": ["dense_eager", "structured_eager", "reduce_batch"],
        "pr12_candidate_schema": "boundflow.backend_candidate/v1.0",
        "pr12_backend_profile_schema": "boundflow.backend_profile/v2.0",
        "high_regret_cases": 9,
        "candidate_not_available_cases": 9,
        "backend_gap_hypotheses": 7,
        "heldout_split_id": HELDOUT_SPLIT_ID,
        "hardware": _hardware(),
        "outputs": {
            "tests.txt": _sha256(args.tests_file),
            "planner_freeze_ref.json": _sha256(planner_path),
            "heldout_split.json": _sha256(heldout_path),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
