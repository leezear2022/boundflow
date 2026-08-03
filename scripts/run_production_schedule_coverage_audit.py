#!/usr/bin/env python3
"""Generate or replay the production Schedule IR P0 coverage artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from boundflow.planner.production_schedule_coverage import (
    canonical_json,
    generate_production_schedule_coverage_artifact,
    replay_production_schedule_coverage_artifact,
)


def main() -> None:
    """Run artifact generation or exact semantic replay."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--ir5-artifact-dir", type=Path, required=True)
    parser.add_argument("--rvir-artifact-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    operation = (
        replay_production_schedule_coverage_artifact
        if args.replay
        else generate_production_schedule_coverage_artifact
    )
    coverage = operation(
        args.artifact_dir,
        ir5_artifact_dir=args.ir5_artifact_dir,
        rvir_artifact_dir=args.rvir_artifact_dir,
    )
    print(
        canonical_json(
            {
                "schema_version": coverage["schema_version"],
                "verdict": coverage["verdict"],
                "failed_gate_ids": coverage["failed_gate_ids"],
            }
        )
    )


if __name__ == "__main__":
    main()
