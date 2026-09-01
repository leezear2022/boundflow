"""Frozen BAB4 same-solver five-fresh artifact tests."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.probe_bab4_same_solver_five_fresh_tamper import run as run_tamper
from scripts import run_bab4_same_solver_five_fresh as bab4

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/bab4-same-solver-five-fresh/resnet2b-prop0-v1"


def test_bab4_five_fresh_replay_and_source_identity() -> None:
    bab4.configure()
    bab4.implementation._replay(argparse.Namespace(artifact=ARTIFACT))
    protocol = json.loads((ARTIFACT / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["source_git_head"] == "7bd28bdae4ac4f0093089d66510806ef09cf9028"
    for relative, expected in protocol["code_revision"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected


def test_bab4_five_fresh_summary_closes_query_research_gate() -> None:
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    assert summary["pair_count"] == 5
    assert summary["worker_count"] == 10
    assert summary["all_environment_admitted"] is True
    assert summary["all_discrete_semantics_exact"] is True
    assert summary["lower_max_abs_diff"] <= 2e-4
    assert summary["lower_sign_exact"] is True
    assert summary["query_speedup_geomean"] >= 1.15
    assert summary["query_speedup_worst"] >= 1.0
    assert summary["query_research_gate_qualified"] is True
    assert summary["core_research_gate_qualified"] is False
    assert summary["performance_claimed"] is False


def test_bab4_five_fresh_ten_outer_resigned_tampers_are_rejected() -> None:
    result = run_tamper(ARTIFACT)
    assert result["attack_count"] == 10
    assert result["rejected_count"] == 10
    assert result["attack_model"] == "raw-worker-mutation-plus-outer-manifest-resign"
    assert result["coherent-full-resign_claimed"] is False
