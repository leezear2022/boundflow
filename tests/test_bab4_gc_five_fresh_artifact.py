"""Frozen symmetric prepared-GC BAB4 five-fresh artifact tests."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts import run_bab4_gc_five_fresh as formal

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/bab4-gc-five-fresh/resnet2b-prop0-v1"


def test_bab4_gc_five_fresh_replays_and_binds_source_identity() -> None:
    formal.configure()
    formal.implementation._replay(argparse.Namespace(artifact=ARTIFACT))
    protocol = json.loads((ARTIFACT / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["source_git_head"] == "83cdc9d7cbf2d541bd642fcaed3947becb359796"
    assert protocol["prepared_gc_isolation"] is True
    for relative, expected in protocol["code_revision"].items():
        frozen = subprocess.run(
            ("git", "show", f"{protocol['source_git_head']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(frozen).hexdigest() == expected


def test_bab4_gc_summary_qualifies_parity_but_not_research_gate() -> None:
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    assert summary["control_configuration"] == "B4-A-GC"
    assert summary["candidate_configuration"] == "BAB4-GC"
    assert summary["prepared_gc_isolation"] is True
    assert summary["pair_count"] == 5
    assert summary["worker_count"] == 10
    assert summary["all_environment_admitted"] is True
    assert summary["all_discrete_semantics_exact"] is True
    assert summary["lower_max_abs_diff"] <= 2e-4
    assert summary["lower_sign_exact"] is True
    assert summary["query_speedup_geomean"] >= 1.06
    assert summary["query_speedup_worst"] >= 1.0
    assert summary["query_parity_qualified"] is True
    assert summary["query_research_gate_qualified"] is False
    assert summary["core_speedup_geomean"] >= 1.15
    assert summary["core_research_gate_qualified"] is False
    assert summary["performance_claimed"] is False


def test_bab4_gc_both_sides_restore_gc_and_preserve_memory_boundary() -> None:
    allocated_ratios: list[float] = []
    reserved_ratios: list[float] = []
    for pair in range(5):
        workers = {}
        for configuration in ("B4-A-GC", "BAB4-GC"):
            worker = json.loads(
                (
                    ARTIFACT
                    / "raw"
                    / f"pair-{pair:02d}"
                    / configuration
                    / "worker.json"
                ).read_text(encoding="utf-8")
            )
            workers[configuration] = worker
            receipt = worker["diagnostics"]["prepared_gc_isolation"]
            assert receipt["query_collect_generation"] == 1
            assert receipt["query_collect_call_count"] == 1
            assert receipt["query_collection_preserved"] is True
            assert receipt["query_timing_excluded"] is True
            assert receipt["restored"] is True
            assert receipt["performance_claimed"] is False

        control = workers["B4-A-GC"]["run"]["metrics"]
        candidate = workers["BAB4-GC"]["run"]["metrics"]
        allocated_ratios.append(
            candidate["peak_allocated_bytes"] / control["peak_allocated_bytes"]
        )
        reserved_ratios.append(
            candidate["peak_reserved_bytes"] / control["peak_reserved_bytes"]
        )
    assert max(allocated_ratios) <= 1.01
    assert max(reserved_ratios) <= 1.02
