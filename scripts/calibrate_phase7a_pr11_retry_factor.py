#!/usr/bin/env python
"""Select bounded-retry candidate-budget inflation by workload-family LOO."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boundflow.planner.materialization_placement import PlacementRetryCandidate
from boundflow.planner.materialization_placement import (
    rank_bounded_placement_retry_candidates,
)
from boundflow.planner.materialization_placement_cost_model import (
    fit_placement_interaction_cost_model,
)
from scripts.evaluate_phase7a_pr11_barrier_placements import (
    _dense_baselines,
    _pattern,
    _percentile,
    _query_key,
    calibration_samples,
    features_for_row,
    load_profile,
)

CALIBRATION_SCHEMA_VERSION = "boundflow.pr11-retry-factor-calibration/v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _parse_factors(value: str) -> tuple[float, ...]:
    try:
        factors = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "factors must be comma-separated floats"
        ) from error
    if not factors or any(factor < 1.0 for factor in factors):
        raise argparse.ArgumentTypeError("factors must be >= 1")
    return factors


def _parse_ridges(value: str) -> tuple[float, ...]:
    try:
        ridges = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ridges must be comma-separated floats"
        ) from error
    if not ridges or any(ridge < 0.0 for ridge in ridges):
        raise argparse.ArgumentTypeError("ridges must be >= 0")
    return ridges


def _representative_budgets(rows: Sequence[dict], maximum: int = 8) -> tuple[int, ...]:
    peaks = sorted(
        {int(row["timing_trace_off"]["peak_cuda_allocated_bytes"]) for row in rows}
    )
    if len(peaks) <= maximum:
        return tuple(peaks)
    return tuple(
        peaks[round(index * (len(peaks) - 1) / (maximum - 1))]
        for index in range(maximum)
    )


def calibrate_factor(  # pylint: disable=too-many-locals
    rows: Sequence[dict],
    *,
    factors: Sequence[float],
    ridges: Sequence[float] = (1e-6,),
    max_attempts: int,
) -> tuple[float, float, list[dict[str, object]]]:
    """Run workload-family LOO and jointly select ridge plus retry factor."""

    query_keys = sorted({_query_key(row) for row in rows})
    if len(query_keys) < 2:
        raise ValueError(
            "retry factor calibration requires at least two query families"
        )
    output: list[dict[str, object]] = []
    for ridge in ridges:
        for factor in factors:
            output.append(
                _evaluate_calibration_pair(
                    rows,
                    query_keys=query_keys,
                    factor=float(factor),
                    ridge=float(ridge),
                    max_attempts=max_attempts,
                )
            )

    def numeric(row: dict[str, object], name: str) -> float:
        value = row[name]
        if not isinstance(value, (int, float)):
            raise TypeError(f"calibration metric {name} must be numeric")
        return float(value)

    selected = min(
        output,
        key=lambda row: (
            numeric(row, "unexpected_failures"),
            numeric(row, "latency_regret_p90"),
            numeric(row, "latency_regret_median"),
            numeric(row, "factor"),
            numeric(row, "ridge"),
        ),
    )
    return numeric(selected, "factor"), numeric(selected, "ridge"), output


def _evaluate_calibration_pair(  # pylint: disable=too-many-locals
    rows: Sequence[dict],
    *,
    query_keys: Sequence[tuple[str, int, int]],
    factor: float,
    ridge: float,
    max_attempts: int,
) -> dict[str, object]:
    """Evaluate one frozen ridge/factor pair across LOO query families."""

    regrets: list[float] = []
    unexpected = 0
    attempts: list[int] = []
    evaluated_budgets = 0
    for heldout_key in query_keys:
        heldout = [row for row in rows if _query_key(row) == heldout_key]
        training = [row for row in rows if _query_key(row) != heldout_key]
        model = fit_placement_interaction_cost_model(
            calibration_samples(training), ridge=float(ridge)
        )
        baseline = next(iter(_dense_baselines(heldout).values()))
        by_pattern = {_pattern(row): row for row in heldout}
        predictions = {
            pattern: model.predict(features_for_row(row, baseline))
            for pattern, row in by_pattern.items()
        }
        for budget in _representative_budgets(heldout):
            evaluated_budgets += 1
            candidates = (
                PlacementRetryCandidate(
                    candidate_id=pattern,
                    predicted_peak_bytes=prediction.peak_bytes,
                    predicted_latency_ms=prediction.latency_ms,
                    conservative="D" not in pattern,
                    structured_count=pattern.count("S"),
                    barrier_count=len(pattern),
                    action_transition_count=sum(
                        lhs != rhs for lhs, rhs in zip(pattern, pattern[1:])
                    ),
                )
                for pattern, prediction in predictions.items()
            )
            ladder = rank_bounded_placement_retry_candidates(
                candidates,
                memory_budget_bytes=round(float(budget) * float(factor)),
                max_attempts=max_attempts,
            )
            selected = None
            for attempt, pattern in enumerate(ladder, start=1):
                if (
                    int(
                        by_pattern[pattern]["timing_trace_off"][
                            "peak_cuda_allocated_bytes"
                        ]
                    )
                    <= budget
                ):
                    selected = by_pattern[pattern]
                    attempts.append(attempt)
                    break
            feasible = [
                row
                for row in heldout
                if int(row["timing_trace_off"]["peak_cuda_allocated_bytes"]) <= budget
            ]
            oracle = min(
                feasible,
                key=lambda row: float(row["timing_trace_off"]["latency_ms_median"]),
            )
            if selected is None:
                unexpected += 1
            else:
                regrets.append(
                    float(selected["timing_trace_off"]["latency_ms_median"])
                    / float(oracle["timing_trace_off"]["latency_ms_median"])
                )
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "factor": float(factor),
        "ridge": float(ridge),
        "query_families": len(query_keys),
        "evaluated_budgets": evaluated_budgets,
        "unexpected_failures": unexpected,
        "latency_regret_median": statistics.median(regrets),
        "latency_regret_p90": _percentile(regrets, 0.9),
        "latency_regret_max": max(regrets),
        "max_attempts_observed": max(attempts),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Calibrate the frozen retry factor and emit auditable raw evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-extra", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--factors",
        type=_parse_factors,
        default=_parse_factors("1.0,1.05,1.1,1.15,1.2,1.25,1.3"),
    )
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument(
        "--ridges",
        type=_parse_ridges,
        default=_parse_ridges("1e-6,1e-5,1e-4,1e-3,1e-2"),
    )
    args = parser.parse_args(argv)
    if args.max_attempts <= 0:
        parser.error("max-attempts must be positive")

    calibration_paths = [args.calibration, *args.calibration_extra]
    selected_factor, selected_ridge, rows = calibrate_factor(
        [row for path in calibration_paths for row in load_profile(path)],
        factors=args.factors,
        ridges=args.ridges,
        max_attempts=int(args.max_attempts),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    selection_path = args.out_dir / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "selection_rule": (
                    "min_unexpected_then_p90_then_median_then_factor_then_ridge"
                ),
                "retry_strategy": "latency_topology_density_stratified_v3",
                "selected_factor": selected_factor,
                "selected_ridge": selected_ridge,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "boundflow.pr11-retry-factor-calibration-manifest/v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "calibration": [
            {"path": str(path), "sha256": _sha256(path)} for path in calibration_paths
        ],
        "factors": list(args.factors),
        "ridges": list(args.ridges),
        "max_attempts": int(args.max_attempts),
        "selected_factor": selected_factor,
        "selected_ridge": selected_ridge,
        "retry_strategy": "latency_topology_density_stratified_v3",
        "outputs": {
            "raw.jsonl": _sha256(raw_path),
            "selection.json": _sha256(selection_path),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
