"""Repository replay gates for the frozen MR5 multi-Conv timing artifact."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import json
from pathlib import Path

from boundflow.runtime.mr5_multi_conv_timing import NO_GO_STATUS
from scripts.run_mr5_multi_conv_timing_formal import replay_artifact

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/measurement-recovery/mr5-multi-conv-timing-v1"


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_mr5_multi_conv_timing_replays_to_no_go() -> None:
    summary = replay_artifact(ARTIFACT)
    assert summary["status"] == NO_GO_STATUS
    assert summary["performance_claimed"] is False
    assert summary["same_solver_complete_query_timing_open"] is False


def test_frozen_mr5_multi_conv_timing_counts_and_gates_are_exact() -> None:
    summary = _load("summary.json")
    assert summary["run_count"] == 12
    assert summary["pair_count"] == 6
    assert summary["host_event_direction_consistent_count"] == 6
    assert summary["host_speedup_geomean"] == 0.8344066482697061
    assert summary["host_speedup_bootstrap_95_lower"] == 0.8284977279712585
    assert summary["host_speedup_worst_pair"] == 0.8246435080415597
    assert summary["absolute_peak_allocated_worst_ratio"] == 0.9945703600234308
    assert summary["absolute_peak_reserved_worst_ratio"] == 1.0
    assert summary["gates"] == {
        "absolute_peak_allocated": True,
        "absolute_peak_reserved": True,
        "bootstrap_lower": False,
        "correctness": True,
        "host_event_direction": True,
        "host_geomean": False,
        "module_stability": True,
        "pair_count": True,
        "worst_pair": False,
    }


def test_frozen_mr5_multi_conv_timing_semantics_and_receipts_are_closed() -> None:
    summary = _load("summary.json")
    metrics = summary["pair_metrics"]
    assert isinstance(metrics, list) and len(metrics) == 6
    assert all(metric["semantic_element_count"] == 9540 for metric in metrics)
    assert max(
        metric["semantic_maximum_absolute_difference"] for metric in metrics
    ) == (4.708766937255859e-06)
    raw = _load("raw.json")
    runs = raw["runs"]
    assert isinstance(runs, list) and len(runs) == 12
    bridge_workers = [row["worker"] for row in runs if row["mode"] == "bridge"]
    for worker in bridge_workers:
        receipt = worker["bridge_receipt"]
        assert receipt["forward_launches"] == {"C0": 10, "C1": 10, "C2": 10}
        assert receipt["backward_launches"] == {"C0": 9, "C1": 9, "C2": 9}
        assert receipt["cache_miss_count"] == {"C0": 0, "C1": 0, "C2": 0}
        assert receipt["cache_hit_count"] == {"C0": 10, "C1": 10, "C2": 10}
        assert receipt["fallback_count"] == 0
        assert receipt["eager_count"] == 0
        assert receipt["native_shadow_count"] == 0


def test_frozen_mr5_multi_conv_timing_tamper_and_paths_are_exact() -> None:
    tamper = _load("tamper_report.json")
    assert tamper["attack_count"] == 20
    assert tamper["rejected_count"] == 20
    assert tamper["all_rejected"] is True
    for path in ARTIFACT.rglob("*"):
        if path.is_file():
            assert "/home/" not in path.read_text(encoding="utf-8")
