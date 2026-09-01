"""Frozen fully warm-matched BAB4 rfactor artifact tests."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts import run_bab4_rfactor_warm_five_fresh as warm

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/bab4-rfactor-warm-five-fresh/resnet2b-prop0-v1"


def test_bab4_rfactor_warm_five_fresh_replay_and_source_identity() -> None:
    warm.configure()
    warm.implementation._replay(argparse.Namespace(artifact=ARTIFACT))
    protocol = json.loads((ARTIFACT / "protocol.json").read_text(encoding="utf-8"))
    for relative, expected in protocol["code_revision"].items():
        frozen = subprocess.run(
            ("git", "show", f"{protocol['source_git_head']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(frozen).hexdigest() == expected


def test_bab4_rfactor_warm_summary_is_reduced_not_research_qualified() -> None:
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    assert summary["control_configuration"] == "B4-A-WARM"
    assert summary["candidate_configuration"] == "BAB4-WARM"
    assert summary["pair_count"] == 5
    assert summary["worker_count"] == 10
    assert summary["all_environment_admitted"] is True
    assert summary["all_discrete_semantics_exact"] is True
    assert summary["lower_max_abs_diff"] <= 2e-4
    assert summary["lower_sign_exact"] is True
    assert summary["query_speedup_geomean"] >= 1.0
    assert summary["query_speedup_worst"] >= 1.0
    assert summary["query_research_gate_qualified"] is False
    assert summary["core_speedup_geomean"] >= 1.15
    assert summary["core_research_gate_qualified"] is False
    assert summary["performance_claimed"] is False


def test_bab4_rfactor_warm_both_sides_bind_root_warmup() -> None:
    for pair in range(5):
        for configuration in ("B4-A-WARM", "BAB4-WARM"):
            worker = json.loads(
                (
                    ARTIFACT
                    / "raw"
                    / f"pair-{pair:02d}"
                    / configuration
                    / "worker.json"
                ).read_text(encoding="utf-8")
            )
            receipt = worker["diagnostics"]["prepared_root_optimizer_warmup"]
            assert receipt["exact_model_property_warmup"] is True
            assert receipt["query_timing_excluded"] is True
            assert receipt["performance_claimed"] is False
            expected_exact_call_count = int(configuration == "BAB4-WARM")
            assert len(worker["s4_exact_call_receipts"]) == expected_exact_call_count
