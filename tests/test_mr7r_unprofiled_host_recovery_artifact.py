"""Repository replay gates for the frozen MR7-R artifact."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts.run_mr7r_unprofiled_host_recovery_formal import replay_artifact

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/measurement-recovery/mr7r-unprofiled-host-recovery-v1"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.is_dir(), reason="MR7-R formal artifact not generated yet"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_mr7r_artifact_replays_and_keeps_timing_closed() -> None:
    summary = replay_artifact(ARTIFACT)
    assert summary["status"] in {
        "INVALID_MR7R_LEDGER_PERTURBATION",
        "VALIDATED_MR7R_HOST_BOUNDARY_OPPORTUNITY",
        "VALIDATED_NO_GO_MR7R_HOST_BOUNDARY",
    }
    assert summary["timing_open"] is False
    assert summary["performance_claimed"] is False


def test_frozen_mr7r_artifact_has_ten_fresh_and_five_pairs() -> None:
    summary = _load("summary.json")
    assert summary["run_count"] == 10
    assert summary["pair_count"] == 5
    assert len(cast(list[object], summary["pair_metrics"])) == 5
    assert summary["host_closure_qualifying_count"] == 5


def test_frozen_mr7r_tamper_and_paths_are_exact() -> None:
    tamper = _load("tamper_report.json")
    assert tamper["attack_count"] == 12
    assert tamper["rejected_count"] == 12
    assert tamper["all_rejected"] is True
    for path in ARTIFACT.rglob("*"):
        if path.is_file():
            assert "/home/" not in path.read_text(encoding="utf-8")
