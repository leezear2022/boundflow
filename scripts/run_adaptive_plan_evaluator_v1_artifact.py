"""Generate/replay the deterministic IR-5B policy-evaluator contract artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from boundflow.planner.adaptive_plan_evaluator import (
    AdaptiveEvaluationContext,
    AdaptivePlanObservation,
    evaluate_adaptive_plan_policies,
    summarize_adaptive_outcomes,
)

SCHEMA = "boundflow.adaptive-plan-evaluator-artifact/v1"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def build_artifact() -> dict[str, object]:
    """Evaluate frozen synthetic observations used only as a contract oracle."""

    contexts = (
        AdaptiveEvaluationContext("cold", 160, 1),
        AdaptiveEvaluationContext("repeated", 160, 100),
        AdaptiveEvaluationContext("low-memory", 80, 10),
        AdaptiveEvaluationContext("warm", 160, 1, ("compiled:fused",)),
    )
    observations = (
        AdaptivePlanObservation(
            "fixed-dense",
            _hash("fixed-dense"),
            2.0,
            0.0,
            2.0,
            (1.9, 2.0, 2.1, 2.2, 2.0),
            0.0,
            120,
        ),
        AdaptivePlanObservation(
            "compiled-fused",
            _hash("compiled-fused"),
            0.8,
            18.0,
            0.8,
            (0.7, 0.8, 0.9, 1.0, 0.8),
            20.0,
            140,
            "compiled:fused",
        ),
        AdaptivePlanObservation(
            "structured-low-memory",
            _hash("structured-low-memory"),
            3.0,
            0.0,
            3.0,
            (2.8, 3.0, 3.2, 3.4, 3.0),
            0.0,
            64,
        ),
    )
    outcomes = evaluate_adaptive_plan_policies(
        contexts, observations, fixed_plan_id="fixed-dense"
    )
    return {
        "schema_version": SCHEMA,
        "evidence_scope": "synthetic_contract_only_not_heldout_performance",
        "outcomes": [item.to_dict() for item in outcomes],
        "summary": summarize_adaptive_outcomes(outcomes),
    }


def _canonical(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a canonical artifact or reject replay drift."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--out", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    encoded = _canonical(build_artifact()) + "\n"
    if args.command == "generate":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
        return 0
    if args.artifact.read_text(encoding="utf-8") != encoded:
        raise ValueError("IR-5B adaptive evaluator artifact replay mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
