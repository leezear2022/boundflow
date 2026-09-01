"""BAB4 complete-verifier prelude attribution contracts."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import run_asplos27_s4_same_solver_worker as s4_worker
from scripts import run_bab4_complete_prelude_profile as profile
from scripts import run_fsg4_b3_same_solver_timing as b3_worker
from scripts import run_fsg4_b4a_same_solver_worker as b4a_worker


def _payload(*, query_ns: int, core_ns: int, pre_core_ns: int, phase_ns: int):
    aggregates = {
        name: {
            "call_count": 1,
            "inclusive_ns": phase_ns,
            "exclusive_ns": phase_ns,
        }
        for name in profile.PHASES
    }
    return {
        "run": {
            "environment": {"admitted": True},
            "metrics": {"query_wall_ns": query_ns, "core_wall_ns": core_ns},
        },
        "diagnostics": {
            "query_phase_timing": {"pre_core_ns": pre_core_ns},
            "complete_prelude_timings": {"aggregates": aggregates},
        },
    }


def test_complete_prelude_summary_uses_pairwise_candidate_minus_control() -> None:
    workers = {}
    for pair in range(3):
        workers[(pair, profile.CONTROL)] = _payload(
            query_ns=200, core_ns=100, pre_core_ns=50, phase_ns=20
        )
        workers[(pair, profile.CANDIDATE)] = _payload(
            query_ns=160, core_ns=80, pre_core_ns=55, phase_ns=23
        )
    summary = profile._summarize(workers)
    assert summary["query_speedup_geomean"] == 1.25
    assert summary["core_speedup_geomean"] == 1.25
    medians = summary["candidate_minus_control_median_ms"]
    assert medians["pre_core_ms"] == 5e-6
    assert medians["gc_collect_inclusive_ms"] == 3e-6
    assert summary["profile_timing_claimed"] is False
    assert summary["performance_claimed"] is False


def test_complete_prelude_flag_crosses_all_same_solver_adapters() -> None:
    args = argparse.Namespace(
        configuration="B4-A-WARM",
        mode="control",
        run_id="test",
        block_index=0,
        sequence_position=0,
        benchmark_root=Path("benchmark"),
        abcrown_root=Path("abcrown"),
        model=Path("model.onnx"),
        property=Path("property.vnnlib"),
        attribute_root_incomplete=False,
        attribute_complete_prelude=True,
    )
    s4 = s4_worker._base_namespace(args, Path("s4.json"))
    b4a = b4a_worker._b3_namespace(s4, Path("b4a.json"))
    b3 = b3_worker._base_namespace(b4a, Path("b3.json"))
    assert s4.attribute_complete_prelude is True
    assert b4a.attribute_complete_prelude is True
    assert b3.attribute_complete_prelude is True
