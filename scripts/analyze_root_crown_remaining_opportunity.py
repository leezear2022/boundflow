#!/usr/bin/env python3
"""Recompute the remaining activation-BaB opportunity after root CROWN fusion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "boundflow.root-crown-remaining-bab-opportunity/v1"
DEFAULT_REGION_SPEEDUPS = (2.0, 4.0, 10.0)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _read(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geomean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _metrics(payload: Mapping[str, Any], *, candidate: bool) -> tuple[int, int, int]:
    run = _mapping(payload.get("run"), name="run")
    metrics = _mapping(run.get("metrics"), name="run.metrics")
    diagnostics = _mapping(payload.get("diagnostics"), name="diagnostics")
    root_timings = _mapping(
        diagnostics.get("root_incomplete_timings"), name="root_incomplete_timings"
    )
    aggregates = _mapping(root_timings.get("aggregates"), name="aggregates")
    root = _mapping(aggregates.get("root_incomplete"), name="root_incomplete")
    host_phases = _mapping(
        diagnostics.get("host_phase_timings"), name="host_phase_timings"
    )
    bab_solve = _mapping(host_phases.get("bab_solve"), name="bab_solve")
    query_ns = _positive_int(metrics.get("query_wall_ns"), name="query_wall_ns")
    root_ns = _positive_int(root.get("inclusive_ns"), name="root_incomplete_ns")
    bab_ns = _positive_int(bab_solve.get("wall_ns"), name="bab_solve_ns")
    if root_ns >= query_ns or bab_ns >= query_ns:
        raise ValueError("timing scopes do not fit inside the query")
    if candidate:
        if (
            payload.get("root_expanded_mode") != "candidate-full"
            or payload.get("cumulative_autograd_owner_count") != 1
            or payload.get("root_suffix_receipt") is None
            or payload.get("root_projection_receipt") is None
            or payload.get("root_input_domain_receipt") is None
        ):
            raise ValueError("candidate is not the cumulative full root owner")
    return query_ns, root_ns, bab_ns


def _projection(
    *, control_query_ns: int, candidate_query_ns: int, bab_ns: int, speedup: float
) -> dict[str, float | str]:
    if speedup <= 1.0 and not math.isinf(speedup):
        raise ValueError("region speedup must exceed one")
    residual_ns = candidate_query_ns - bab_ns
    projected_query_ns = (
        float(residual_ns) if math.isinf(speedup) else residual_ns + bab_ns / speedup
    )
    return {
        "region_speedup": "infinite" if math.isinf(speedup) else speedup,
        "candidate_query_speedup": candidate_query_ns / projected_query_ns,
        "control_query_speedup": control_query_ns / projected_query_ns,
    }


def analyze(  # pylint: disable=too-many-locals
    artifact_dir: Path,
    *,
    pair_count: int = 3,
    region_speedups: tuple[float, ...] = DEFAULT_REGION_SPEEDUPS,
) -> dict[str, object]:
    """Return an independently recomputed opportunity ledger."""

    pairs: list[dict[str, object]] = []
    for pair_index in range(pair_count):
        control = _read(artifact_dir / f"pair-{pair_index}-control.json")
        candidate = _read(artifact_dir / f"pair-{pair_index}-candidate-full.json")
        control_query_ns, control_root_ns, _ = _metrics(control, candidate=False)
        candidate_query_ns, candidate_root_ns, candidate_bab_ns = _metrics(
            candidate, candidate=True
        )
        projections = [
            _projection(
                control_query_ns=control_query_ns,
                candidate_query_ns=candidate_query_ns,
                bab_ns=candidate_bab_ns,
                speedup=speedup,
            )
            for speedup in (*region_speedups, math.inf)
        ]
        pairs.append(
            {
                "pair_index": pair_index,
                "control_query_ns": control_query_ns,
                "candidate_query_ns": candidate_query_ns,
                "control_root_ns": control_root_ns,
                "candidate_root_ns": candidate_root_ns,
                "candidate_bab_solve_ns": candidate_bab_ns,
                "current_control_query_speedup": control_query_ns / candidate_query_ns,
                "candidate_bab_solve_share": candidate_bab_ns / candidate_query_ns,
                "projections": projections,
            }
        )

    current = _geomean(
        [
            _number(pair["current_control_query_speedup"], name="current speedup")
            for pair in pairs
        ]
    )
    projection_summary: list[dict[str, float | str]] = []
    for ordinal, speedup in enumerate((*region_speedups, math.inf)):
        candidate_values = []
        control_values = []
        for pair in pairs:
            projection = pair["projections"]
            if not isinstance(projection, list):
                raise TypeError("projection ledger differs")
            row = _mapping(projection[ordinal], name="projection")
            candidate_values.append(
                _number(row.get("candidate_query_speedup"), name="candidate speedup")
            )
            control_values.append(
                _number(row.get("control_query_speedup"), name="control speedup")
            )
        projection_summary.append(
            {
                "region_speedup": "infinite" if math.isinf(speedup) else speedup,
                "candidate_query_geomean": _geomean(candidate_values),
                "control_query_geomean": _geomean(control_values),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_dir": artifact_dir.name,
        "pair_count": pair_count,
        "current_control_query_geomean": current,
        "candidate_bab_solve_share_min": min(
            _number(pair["candidate_bab_solve_share"], name="BaB share")
            for pair in pairs
        ),
        "candidate_bab_solve_share_max": max(
            _number(pair["candidate_bab_solve_share"], name="BaB share")
            for pair in pairs
        ),
        "projection_summary": projection_summary,
        "pairs": pairs,
        "interpretation": {
            "current_owner_scope": "initial/root CROWN /49 to /input-1",
            "remaining_target": "activation-BaB update_bounds_core",
            "remaining_transaction": "10 evaluations, 9 optimizer mutations, active beta",
            "projection_is_claim": False,
            "performance_claimed": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--pair-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Print one canonical opportunity ledger."""

    args = _parse_args()
    result = analyze(args.artifact_dir.resolve(), pair_count=args.pair_count)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
