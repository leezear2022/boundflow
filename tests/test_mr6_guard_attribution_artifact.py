"""Repository replay gates for the frozen MR6 guard attribution artifact."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import json
from pathlib import Path

from boundflow.runtime.mr6_guard_attribution import NO_GO_STATUS
from scripts.run_mr6_guard_attribution_formal import replay_artifact

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/measurement-recovery/mr6-hot-path-guard-attribution-v1"


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_mr6_guard_artifact_replays_to_no_go() -> None:
    summary = replay_artifact(ARTIFACT)
    assert summary["status"] == NO_GO_STATUS
    assert summary["safe_guard_fusion_open"] is False
    assert summary["performance_claimed"] is False


def test_frozen_mr6_guard_ratios_and_gates_are_exact() -> None:
    summary = _load("summary.json")
    assert summary["run_count"] == 9
    assert summary["triplet_count"] == 3
    assert summary["full_diagnostic_host_geomean"] == 1.0331256409896374
    assert summary["full_diagnostic_bootstrap_95_lower"] == 1.0084436545902995
    assert summary["full_diagnostic_worst"] == 1.0084436545902995
    assert summary["provider_diagnostic_host_geomean"] == 0.9030066500186469
    assert summary["provider_diagnostic_bootstrap_95_lower"] == 0.8526133303221223
    assert summary["provider_diagnostic_worst"] == 0.8526133303221223
    assert summary["provider_full_host_geomean"] == 0.874053081437076
    assert summary["host_event_direction_consistent_count"] == 9
    assert summary["gates"] == {
        "full_diagnostic_geomean": False,
        "guard_counts": True,
        "host_event_direction": True,
        "module_stability": True,
        "provider_diagnostic_geomean": False,
        "provider_diagnostic_worst": False,
        "semantic_exact": True,
        "triplet_count": True,
    }


def test_frozen_mr6_guard_counts_and_semantics_are_closed() -> None:
    raw = _load("raw.json")
    runs = raw["runs"]
    assert isinstance(runs, list) and len(runs) == 9
    observed = {
        row["mode"]: row["worker"]["guard_receipt"]["synchronizing_guards_executed"]
        for row in runs
    }
    assert observed == {"provider": 0, "full": 360, "diagnostic": 60}
    summary = _load("summary.json")
    metrics = summary["triplet_metrics"]
    assert isinstance(metrics, list) and len(metrics) == 3
    comparisons = ("provider_full", "provider_diagnostic", "full_diagnostic")
    assert all(
        row[name]["allclose"] and row[name]["sign_exact"]
        for row in metrics
        for name in comparisons
    )
    assert max(
        row[name]["semantic_maximum_absolute_difference"]
        for row in metrics
        for name in comparisons
    ) == (4.708766937255859e-06)


def test_frozen_mr6_guard_tamper_and_paths_are_exact() -> None:
    tamper = _load("tamper_report.json")
    assert tamper["attack_count"] == 12
    assert tamper["rejected_count"] == 12
    assert tamper["all_rejected"] is True
    for path in ARTIFACT.rglob("*"):
        if path.is_file():
            assert "/home/" not in path.read_text(encoding="utf-8")
