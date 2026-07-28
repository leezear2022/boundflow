"""IR-5B fair adaptive-policy evaluator contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from boundflow.planner.adaptive_plan_evaluator import AdaptivePlanPolicy
from scripts.run_adaptive_plan_evaluator_v1_artifact import build_artifact


def test_adaptive_policy_contract_exposes_expected_tradeoffs() -> None:
    artifact = build_artifact()
    outcomes = artifact["outcomes"]
    assert isinstance(outcomes, list)
    by_key = {(item["context_id"], item["policy"]): item for item in outcomes}

    assert (
        by_key[("cold", AdaptivePlanPolicy.GLOBAL.value)]["selected_plan_id"]
        == "fixed-dense"
    )
    assert (
        by_key[("repeated", AdaptivePlanPolicy.GLOBAL.value)]["selected_plan_id"]
        == "compiled-fused"
    )
    assert (
        by_key[("warm", AdaptivePlanPolicy.GLOBAL.value)]["selected_plan_id"]
        == "compiled-fused"
    )
    assert by_key[("low-memory", AdaptivePlanPolicy.FIXED.value)]["feasible"] is False
    assert (
        by_key[("low-memory", AdaptivePlanPolicy.GLOBAL.value)]["selected_plan_id"]
        == "structured-low-memory"
    )
    assert artifact["evidence_scope"] == (
        "synthetic_contract_only_not_heldout_performance"
    )


def test_adaptive_policy_artifact_replays_in_fresh_process(tmp_path: Path) -> None:
    artifact = tmp_path / "adaptive-evaluator.json"
    subprocess.run(
        (
            sys.executable,
            "scripts/run_adaptive_plan_evaluator_v1_artifact.py",
            "generate",
            "--out",
            str(artifact),
        ),
        check=True,
    )
    subprocess.run(
        (
            sys.executable,
            "scripts/run_adaptive_plan_evaluator_v1_artifact.py",
            "replay",
            "--artifact",
            str(artifact),
        ),
        check=True,
    )
    parsed = json.loads(artifact.read_text(encoding="utf-8"))
    assert parsed == build_artifact()
