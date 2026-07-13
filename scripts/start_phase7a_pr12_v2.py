#!/usr/bin/env python
"""Freeze PR-12G calibration-v2 and a new final held-out split."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

SPLIT_ID = "pr12-multibackend-final-heldout-v2"
SCHEMA_VERSION = "boundflow.pr12-multibackend-split/v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split(v1_split: dict[str, Any]) -> dict[str, Any]:
    """Promote consumed v1 cases to calibration and create unseen v2 cases."""

    calibration = [dict(record) for record in v1_split["calibration"]]
    calibration.extend(dict(record) for record in v1_split["final_heldout"])
    final_heldout = [
        {
            "case_id": "linear-unseen-shape-c-v2",
            "family": "linear",
            "domain": 6,
            "spec": 97,
            "current": 255,
            "previous": 129,
            "budget_mib": 128,
        },
        {
            "case_id": "linear-memory-sensitive-v2",
            "family": "linear",
            "domain": 7,
            "spec": 193,
            "current": 1536,
            "previous": 768,
            "budget_mib": 64,
        },
        {
            "case_id": "linear-small-unseen-v2",
            "family": "linear",
            "domain": 1,
            "spec": 37,
            "current": 83,
            "previous": 41,
            "budget_mib": 128,
        },
        {
            "case_id": "conv-unseen-aspect-v2",
            "family": "conv2d",
            "domain": 3,
            "spec": 73,
            "channels": 10,
            "height": 12,
            "width": 20,
            "kernel": 3,
            "budget_mib": 384,
        },
        {
            "case_id": "mini-resnet-unseen-v2",
            "family": "mini_resnet",
            "domain": 3,
            "spec": 64,
            "width": 14,
            "blocks": 4,
            "budget_mib": 512,
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "split_id": SPLIT_ID,
        "parent_split_id": v1_split["split_id"],
        "candidate_set": [
            "pytorch_eager",
            "pytorch_chunked_r512",
            "tvm_fused_tir",
        ],
        "calibration": calibration,
        "final_heldout": final_heldout,
        "freeze_policy": (
            "v1 final is consumed calibration; v2 final may be evaluated once and "
            "must not be used to alter chunk_rows or planner thresholds"
        ),
        "chunk_rows": 512,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the split and a hash manifest into a new artifact directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    v1_split = json.loads(args.v1_split.read_text(encoding="utf-8"))
    split = build_split(v1_split)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    split_path = args.out_dir / "heldout_split.json"
    split_path.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "boundflow.pr12-multibackend-freeze/v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split_id": SPLIT_ID,
        "parent": {
            "path": str(args.v1_split),
            "sha256": _sha256(args.v1_split),
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
