"""Repository replay gates for the frozen MR7 attribution artifact."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_mr7_launch_materialization_formal import replay_artifact

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "artifacts/measurement-recovery/mr7-launch-materialization-attribution-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_mr7_artifact_replays_and_keeps_claim_closed() -> None:
    summary = replay_artifact(ARTIFACT)
    assert summary["status"] in {
        "MR7_A_COMPILED_REGION_ARENA_FFI",
        "MR7_B_PER_SITE_SCHEDULE",
        "MR7_C_CROSS_SITE_STEP_EXECUTION_GRAPH",
        "NO_GO_CURRENT_CONV_REPLACEMENT",
        "INVALID_MR7_ATTRIBUTION",
    }
    assert summary["performance_claimed"] is False


def test_frozen_mr7_artifact_has_six_fresh_and_exact_closure() -> None:
    summary = _load("summary.json")
    assert summary["run_count"] == 6
    assert summary["pair_count"] == 3
    assert len(summary["pair_metrics"]) == 3
    assert summary["gates"]["host_closure"] is True
    assert summary["gates"]["device_envelope_closure"] is True
    assert summary["gates"]["semantic_exact"] is True
    assert summary["gates"]["launch_counts"] is True


def test_frozen_mr7_tamper_and_paths_are_exact() -> None:
    tamper = _load("tamper_report.json")
    assert tamper["attack_count"] == 11
    assert tamper["rejected_count"] == 11
    assert tamper["all_rejected"] is True
    for path in ARTIFACT.rglob("*"):
        if path.is_file():
            assert "/home/" not in path.read_text(encoding="utf-8")
