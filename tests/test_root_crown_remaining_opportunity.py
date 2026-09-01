"""Tests for the post-root-CROWN activation-BaB opportunity ledger."""

# pylint: disable=missing-function-docstring,too-many-arguments

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.analyze_root_crown_remaining_opportunity import analyze


def _write_pair(
    root: Path,
    pair_index: int,
    mode: str,
    *,
    query_ns: int,
    root_ns: int,
    bab_ns: int,
) -> None:
    candidate = mode == "candidate-full"
    payload = {
        "root_expanded_mode": mode,
        "cumulative_autograd_owner_count": 1 if candidate else 0,
        "root_suffix_receipt": {} if candidate else None,
        "root_projection_receipt": {} if candidate else None,
        "root_input_domain_receipt": {} if candidate else None,
        "run": {"metrics": {"query_wall_ns": query_ns}},
        "diagnostics": {
            "root_incomplete_timings": {
                "aggregates": {"root_incomplete": {"inclusive_ns": root_ns}}
            },
            "host_phase_timings": {"bab_solve": {"wall_ns": bab_ns}},
        },
    }
    (root / f"pair-{pair_index}-{mode}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_analyze_recomputes_amdahl_projection(tmp_path: Path) -> None:
    for pair_index in range(3):
        _write_pair(
            tmp_path,
            pair_index,
            "control",
            query_ns=1200,
            root_ns=700,
            bab_ns=240,
        )
        _write_pair(
            tmp_path,
            pair_index,
            "candidate-full",
            query_ns=1000,
            root_ns=600,
            bab_ns=200,
        )

    result = analyze(tmp_path)

    assert result["current_control_query_geomean"] == pytest.approx(1.2)
    assert result["candidate_bab_solve_share_min"] == pytest.approx(0.2)
    assert result["candidate_bab_solve_share_max"] == pytest.approx(0.2)
    projections = cast(list[dict[str, Any]], result["projection_summary"])
    assert projections[0]["candidate_query_geomean"] == pytest.approx(1 / 0.9)
    assert projections[1]["control_query_geomean"] == pytest.approx(1.2 / 0.85)
    assert projections[-1]["region_speedup"] == "infinite"
    assert projections[-1]["control_query_geomean"] == pytest.approx(1.5)
    interpretation = cast(dict[str, Any], result["interpretation"])
    assert interpretation["performance_claimed"] is False


def test_analyze_rejects_non_full_candidate(tmp_path: Path) -> None:
    _write_pair(tmp_path, 0, "control", query_ns=1200, root_ns=700, bab_ns=240)
    _write_pair(tmp_path, 0, "candidate-full", query_ns=1000, root_ns=600, bab_ns=200)
    path = tmp_path / "pair-0-candidate-full.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["root_input_domain_receipt"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cumulative full root owner"):
        analyze(tmp_path, pair_count=1)


def test_analyze_rejects_invalid_speedup(tmp_path: Path) -> None:
    _write_pair(tmp_path, 0, "control", query_ns=1200, root_ns=700, bab_ns=240)
    _write_pair(tmp_path, 0, "candidate-full", query_ns=1000, root_ns=600, bab_ns=200)

    with pytest.raises(ValueError, match="speedup"):
        analyze(tmp_path, pair_count=1, region_speedups=(1.0,))


def test_projection_outputs_are_finite(tmp_path: Path) -> None:
    _write_pair(tmp_path, 0, "control", query_ns=1200, root_ns=700, bab_ns=240)
    _write_pair(tmp_path, 0, "candidate-full", query_ns=1000, root_ns=600, bab_ns=200)

    result = analyze(tmp_path, pair_count=1)

    rows = cast(list[dict[str, Any]], result["projection_summary"])
    for row in rows:
        assert math.isfinite(float(row["candidate_query_geomean"]))
        assert math.isfinite(float(row["control_query_geomean"]))
