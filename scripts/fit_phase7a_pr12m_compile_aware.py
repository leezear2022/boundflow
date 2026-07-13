#!/usr/bin/env python
"""Fit and freeze the PR-12M Planner before consuming final held-out rows."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

from boundflow.planner.fused_crown_backend import CompileAwareFusedCrownPlanner
from scripts.replay_phase7a_pr12m_compile_aware import (
    _disk_priors,
    _observations,
    _read_jsonl,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the deterministic model and its calibration-only provenance."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--amortization", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    priors = _disk_priors(_read_jsonl(args.amortization))
    planner = CompileAwareFusedCrownPlanner.fit(
        _observations(split, _read_jsonl(args.calibration), priors)
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.out_dir / "planner_model.json"
    model_path.write_text(
        json.dumps(planner.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "boundflow.pr12m-compile-aware-fit/v1",
        "split_id": split["split_id"],
        "final_heldout_consumed": False,
        "disk_setup_priors_ms": priors,
        "inputs": {
            "split": _sha256(args.split_file),
            "calibration": _sha256(args.calibration),
            "amortization": _sha256(args.amortization),
        },
        "outputs": {"planner_model.json": _sha256(model_path)},
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
