#!/usr/bin/env python
"""Freeze the PR-12M compile-aware calibration and final held-out split."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

SCHEMA_VERSION = "boundflow.pr12-compile-aware-split/v1"
SPLIT_ID = "pr12-compile-aware-final-heldout-v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split(parent: dict[str, Any]) -> dict[str, Any]:
    """Promote consumed v2 final rows and define untouched v3 cases."""

    calibration = [dict(record) for record in parent["final_heldout"]]
    final_heldout = [
        {
            "case_id": "linear-compile-aware-width-v3",
            "family": "linear",
            "domain": 5,
            "spec": 113,
            "current": 319,
            "previous": 157,
            "budget_mib": 128,
        },
        {
            "case_id": "linear-compile-aware-memory-v3",
            "family": "linear",
            "domain": 6,
            "spec": 173,
            "current": 1280,
            "previous": 640,
            "budget_mib": 64,
        },
        {
            "case_id": "linear-compile-aware-small-v3",
            "family": "linear",
            "domain": 2,
            "spec": 29,
            "current": 71,
            "previous": 53,
            "budget_mib": 32,
        },
        {
            "case_id": "conv-compile-aware-aspect-v3",
            "family": "conv2d",
            "domain": 2,
            "spec": 83,
            "channels": 12,
            "height": 10,
            "width": 18,
            "kernel": 3,
            "budget_mib": 128,
        },
        {
            "case_id": "mini-resnet-compile-aware-v3",
            "family": "mini_resnet",
            "domain": 2,
            "spec": 48,
            "width": 18,
            "blocks": 3,
            "budget_mib": 128,
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "split_id": SPLIT_ID,
        "parent_split_id": parent["split_id"],
        "candidate_set": [
            "pytorch_eager",
            "pytorch_structured",
            "pytorch_chunked",
            "tvm_tir_unfused",
            "tvm_fused_tir",
        ],
        "calibration": calibration,
        "final_heldout": final_heldout,
        "chunk_rows": int(parent.get("chunk_rows", 512)),
        "budget_mib_sweep": [16, 32, 64, 128, None],
        "reuse_policies": [
            {
                "policy_id": "cold_single",
                "expected_reuse_queries": 1,
                "memory_cache_hit_probability": 0.0,
                "disk_cache_hit_probability": 0.0,
            },
            {
                "policy_id": "mixed_q32",
                "expected_reuse_queries": 32,
                "memory_cache_hit_probability": 0.70,
                "disk_cache_hit_probability": 0.20,
            },
            {
                "policy_id": "warm_q1024",
                "expected_reuse_queries": 1024,
                "memory_cache_hit_probability": 0.95,
                "disk_cache_hit_probability": 0.04,
            },
        ],
        "freeze_policy": (
            "parent v2 final is consumed calibration; v3 final is evaluated once "
            "after model/policies are frozen and must not alter candidates or costs"
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the immutable split and hash manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    parent = json.loads(args.parent_split.read_text(encoding="utf-8"))
    split = build_split(parent)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    split_path = args.out_dir / "heldout_split.json"
    split_path.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "boundflow.pr12-compile-aware-freeze/v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split_id": SPLIT_ID,
        "parent": {
            "path": str(args.parent_split),
            "sha256": _sha256(args.parent_split),
        },
        "calibration_cases": len(split["calibration"]),
        "final_heldout_cases": len(split["final_heldout"]),
        "outputs": {"heldout_split.json": _sha256(split_path)},
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
