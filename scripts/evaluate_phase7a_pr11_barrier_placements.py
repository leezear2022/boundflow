#!/usr/bin/env python
"""Fit barrier interaction costs and evaluate held-out placement policies."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boundflow.planner.materialization_placement_cost_model import (
    PlacementCalibrationSample,
    PlacementFeatures,
    PlacementInteractionCostModel,
    fit_placement_interaction_cost_model,
    placement_features_from_static,
)
from boundflow.planner.materialization_placement import (
    PlacementRetryCandidate,
    rank_bounded_placement_retry_candidates,
)
from boundflow.planner.materialization_static_features import StaticBarrierSummary

EVAL_SCHEMA_VERSION = "boundflow.pr11-barrier-placement-eval/v4"
SUMMARY_SCHEMA_VERSION = "boundflow.pr11-barrier-placement-summary/v1"
PROFILE_SCHEMA_VERSION = "boundflow.pr11-barrier-placement-profile/v3"
POLICIES = (
    "always_dense",
    "always_structured",
    "memory_threshold",
    "local_predicted",
    "global_predicted",
    "global_retry",
    "global_bounded_retry",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _actions(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(item["action"] for item in row["placement"]["placements"])


def _pattern(row: dict[str, Any]) -> str:
    return "".join("D" if action == "dense" else "S" for action in _actions(row))


def _query_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["workload"]["name"]),
        int(row["spec_size"]),
        int(row["domain_batch_size"]),
    )


def features_for_row(
    row: dict[str, Any], dense_baseline: dict[str, Any]
) -> PlacementFeatures:
    """Derive candidate-independent static features without trace measurements."""

    actions = _actions(row)
    barriers = tuple(
        StaticBarrierSummary.from_dict(payload) for payload in row["static_barriers"]
    )
    if tuple(barrier.barrier_id for barrier in barriers) != tuple(row["barrier_ids"]):
        raise ValueError("static barrier identities do not match placement profile")
    return placement_features_from_static(
        barriers,
        actions,
        dense_baseline_peak_bytes=int(
            dense_baseline["timing_trace_off"]["peak_cuda_allocated_bytes"]
        ),
        dense_baseline_latency_ms=float(
            dense_baseline["timing_trace_off"]["latency_ms_median"]
        ),
    )


def load_profile(path: Path) -> list[dict[str, Any]]:
    """Load successful exhaustive-profile rows and preserve their measurements."""

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty barrier profile: {path}")
    versions = {row.get("schema_version") for row in rows}
    if versions != {PROFILE_SCHEMA_VERSION}:
        raise ValueError(f"unsupported barrier profile schema(s): {versions}")
    if any("static_barriers" not in row for row in rows):
        raise ValueError("barrier profile is missing static topology features")
    failures = [row["query_id"] for row in rows if row["status"] != "ok"]
    if failures:
        raise ValueError(f"barrier profile contains non-ok rows: {failures[:5]}")
    return rows


def _dense_baselines(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    baselines: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if "S" not in _pattern(row):
            key = _query_key(row)
            if key in baselines:
                raise ValueError(f"duplicate dense baseline for query={key}")
            baselines[key] = row
    return baselines


def calibration_samples(
    rows: Iterable[dict[str, Any]],
) -> list[PlacementCalibrationSample]:
    """Create calibration samples without mixing in held-out rows."""

    materialized = list(rows)
    baselines = _dense_baselines(materialized)
    samples: list[PlacementCalibrationSample] = []
    for row in materialized:
        baseline = baselines[_query_key(row)]
        samples.append(
            PlacementCalibrationSample(
                features=features_for_row(row, baseline),
                peak_bytes=int(row["timing_trace_off"]["peak_cuda_allocated_bytes"]),
                latency_ms=float(row["timing_trace_off"]["latency_ms_median"]),
            )
        )
    return samples


def _actual(row: Optional[dict[str, Any]], *, budget_bytes: int) -> dict[str, Any]:
    if row is None:
        return {
            "status": "no_predicted_plan",
            "feasible": False,
            "peak_bytes": None,
            "latency_ms": None,
        }
    peak = int(row["timing_trace_off"]["peak_cuda_allocated_bytes"])
    return {
        "status": "ok",
        "feasible": peak <= int(budget_bytes),
        "peak_bytes": peak,
        "latency_ms": float(row["timing_trace_off"]["latency_ms_median"]),
        "pattern": _pattern(row),
    }


def _local_pattern(
    rows_by_pattern: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> str:
    dense_pattern = next(pattern for pattern in rows_by_pattern if "S" not in pattern)
    actions = list(dense_pattern)
    dense_latency = float(predictions[dense_pattern]["latency_ms"])
    for index, _action in enumerate(actions):
        singleton = list(dense_pattern)
        singleton[index] = "S"
        singleton_pattern = "".join(singleton)
        if float(predictions[singleton_pattern]["latency_ms"]) < dense_latency:
            actions[index] = "S"
    return "".join(actions)


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
def evaluate_heldout(
    rows: Sequence[dict[str, Any]],
    *,
    model: PlacementInteractionCostModel,
    budgets_bytes: Sequence[int],
    peak_inflation: float,
    retry_max_attempts: int,
    retry_prediction_budget_factor: float,
    calibration_profiles: Sequence[str],
    heldout_profile: str,
) -> list[dict[str, Any]]:
    """Evaluate predicted policies against the measured exhaustive Oracle."""

    baselines = _dense_baselines(rows)
    if len(baselines) != 1:
        raise ValueError(
            "held-out evaluator currently expects exactly one query/workload"
        )
    baseline = next(iter(baselines.values()))
    rows_by_pattern = {_pattern(row): row for row in rows}
    predictions: dict[str, dict[str, Any]] = {}
    for pattern, row in rows_by_pattern.items():
        prediction = model.predict(features_for_row(row, baseline))
        predictions[pattern] = {
            "peak_bytes": int(prediction.peak_bytes),
            "inflated_peak_bytes": int(
                round(float(prediction.peak_bytes) * peak_inflation)
            ),
            "latency_ms": float(prediction.latency_ms),
        }
    dense_pattern = next(pattern for pattern in rows_by_pattern if "S" not in pattern)
    structured_pattern = next(
        pattern for pattern in rows_by_pattern if "D" not in pattern
    )
    local_pattern = _local_pattern(rows_by_pattern, predictions)

    output: list[dict[str, Any]] = []
    for budget in budgets_bytes:
        actual_feasible = [
            row
            for row in rows
            if int(row["timing_trace_off"]["peak_cuda_allocated_bytes"]) <= int(budget)
        ]
        oracle = (
            min(
                actual_feasible,
                key=lambda row: float(row["timing_trace_off"]["latency_ms_median"]),
            )
            if actual_feasible
            else None
        )
        predicted_feasible = [
            pattern
            for pattern, prediction in predictions.items()
            if int(prediction["inflated_peak_bytes"]) <= int(budget)
        ]
        global_pattern: Optional[str] = None
        if dense_pattern in predicted_feasible:
            global_pattern = dense_pattern
        elif predicted_feasible:
            global_pattern = min(
                predicted_feasible,
                key=lambda pattern: (
                    float(predictions[pattern]["latency_ms"]),
                    pattern,
                ),
            )
        retry_candidates = sorted(
            predicted_feasible,
            key=lambda pattern: (
                float(predictions[pattern]["latency_ms"]),
                pattern,
            ),
        )
        retry_attempted: list[str] = []
        retry_pattern: Optional[str] = None
        for candidate_pattern in retry_candidates:
            retry_attempted.append(candidate_pattern)
            candidate_peak = int(
                rows_by_pattern[candidate_pattern]["timing_trace_off"][
                    "peak_cuda_allocated_bytes"
                ]
            )
            if candidate_peak <= int(budget):
                retry_pattern = candidate_pattern
                break
        bounded_ladder = rank_bounded_placement_retry_candidates(
            (
                PlacementRetryCandidate(
                    candidate_id=pattern,
                    predicted_peak_bytes=int(prediction["inflated_peak_bytes"]),
                    predicted_latency_ms=float(prediction["latency_ms"]),
                    conservative="D" not in pattern,
                    structured_count=pattern.count("S"),
                    barrier_count=len(pattern),
                    action_transition_count=sum(
                        lhs != rhs for lhs, rhs in zip(pattern, pattern[1:])
                    ),
                )
                for pattern, prediction in predictions.items()
            ),
            memory_budget_bytes=int(
                round(float(budget) * float(retry_prediction_budget_factor))
            ),
            max_attempts=int(retry_max_attempts),
        )
        bounded_attempted: list[str] = []
        bounded_pattern: Optional[str] = None
        for candidate_pattern in bounded_ladder:
            bounded_attempted.append(candidate_pattern)
            candidate_peak = int(
                rows_by_pattern[candidate_pattern]["timing_trace_off"][
                    "peak_cuda_allocated_bytes"
                ]
            )
            if candidate_peak <= int(budget):
                bounded_pattern = candidate_pattern
                break
        memory_pattern: Optional[str] = None
        if dense_pattern in predicted_feasible:
            memory_pattern = dense_pattern
        elif structured_pattern in predicted_feasible:
            memory_pattern = structured_pattern
        selected: dict[str, Optional[str]] = {
            "always_dense": dense_pattern,
            "always_structured": structured_pattern,
            "memory_threshold": memory_pattern,
            "local_predicted": (
                local_pattern if local_pattern in predicted_feasible else None
            ),
            "global_predicted": global_pattern,
            "global_retry": retry_pattern,
            "global_bounded_retry": bounded_pattern,
        }
        oracle_actual = _actual(oracle, budget_bytes=int(budget))
        for policy in POLICIES:
            selected_pattern = selected[policy]
            selected_row = (
                rows_by_pattern.get(selected_pattern)
                if selected_pattern is not None
                else None
            )
            actual = _actual(selected_row, budget_bytes=int(budget))
            regret = None
            if actual["feasible"] and oracle_actual["feasible"]:
                regret = float(actual["latency_ms"]) / float(
                    oracle_actual["latency_ms"]
                )
            output.append(
                {
                    "schema_version": EVAL_SCHEMA_VERSION,
                    "status": "ok",
                    "split": {
                        "kind": "architecture_family_final_heldout",
                        "calibration_profiles": list(calibration_profiles),
                        "heldout_profile": heldout_profile,
                    },
                    "workload": rows[0]["workload"],
                    "spec_size": rows[0]["spec_size"],
                    "domain_batch_size": rows[0]["domain_batch_size"],
                    "budget_bytes": int(budget),
                    "peak_inflation": float(peak_inflation),
                    "feature_source": "static_topology_liveness_v1",
                    "policy": policy,
                    "selected_pattern": selected_pattern,
                    "prediction": (
                        predictions.get(selected_pattern)
                        if selected_pattern is not None
                        else None
                    ),
                    "retry": (
                        {
                            "feedback": "measured_budget_rejection_replay",
                            "bounded": policy == "global_bounded_retry",
                            "max_attempts": (
                                int(retry_max_attempts)
                                if policy == "global_bounded_retry"
                                else None
                            ),
                            "strategy": (
                                "latency_topology_density_stratified_v3"
                                if policy == "global_bounded_retry"
                                else None
                            ),
                            "prediction_budget_factor": (
                                float(retry_prediction_budget_factor)
                                if policy == "global_bounded_retry"
                                else None
                            ),
                            "attempt_count": len(
                                bounded_attempted
                                if policy == "global_bounded_retry"
                                else retry_attempted
                            ),
                            "attempted_patterns": list(
                                bounded_attempted
                                if policy == "global_bounded_retry"
                                else retry_attempted
                            ),
                        }
                        if policy in {"global_retry", "global_bounded_retry"}
                        else None
                    ),
                    "actual": actual,
                    "oracle": oracle_actual,
                    "metrics": {
                        "feasible": bool(actual["feasible"]),
                        "oracle_feasible": bool(oracle_actual["feasible"]),
                        "unexpected_failure": bool(
                            oracle_actual["feasible"] and not actual["feasible"]
                        ),
                        "latency_regret_ratio": regret,
                    },
                }
            )
    return output


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, object]]:
    """Summarize feasibility and regret once per policy."""

    output: list[dict[str, object]] = []
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        regrets = [
            float(row["metrics"]["latency_regret_ratio"])
            for row in selected
            if row["metrics"]["latency_regret_ratio"] is not None
        ]
        oracle_feasible = sum(
            bool(row["metrics"]["oracle_feasible"]) for row in selected
        )
        planner_feasible = sum(bool(row["metrics"]["feasible"]) for row in selected)
        output.append(
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "policy": policy,
                "rows": len(selected),
                "oracle_feasible": oracle_feasible,
                "planner_feasible": planner_feasible,
                "feasible_coverage": (
                    float(planner_feasible) / float(oracle_feasible)
                    if oracle_feasible
                    else None
                ),
                "unexpected_failures": sum(
                    bool(row["metrics"]["unexpected_failure"]) for row in selected
                ),
                "regret_samples": len(regrets),
                "latency_regret_median": (
                    statistics.median(regrets) if regrets else None
                ),
                "latency_regret_p90": _percentile(regrets, 0.9),
                "latency_regret_p99": _percentile(regrets, 0.99),
                "latency_regret_max": max(regrets) if regrets else None,
            }
        )
    return output


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(item.strip()) * 1024 * 1024 for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "budgets must be comma-separated MiB integers"
        ) from error
    if not budgets or any(value <= 0 for value in budgets):
        raise argparse.ArgumentTypeError("budgets must be positive")
    return budgets


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:  # pylint: disable=too-many-locals
    """Fit on calibration rows and emit final-heldout JSONL/CSV/manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-extra", type=Path, action="append", default=[])
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--budgets-mib",
        type=_parse_budgets,
        default=_parse_budgets("28,30,32,34,36,40,48,64"),
    )
    parser.add_argument("--peak-inflation", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--retry-max-attempts", type=int, default=6)
    parser.add_argument("--retry-prediction-budget-factor", type=float, default=1.3)
    args = parser.parse_args(argv)
    if args.peak_inflation <= 0.0:
        parser.error("peak-inflation must be > 0")
    if args.retry_max_attempts <= 0:
        parser.error("retry-max-attempts must be > 0")
    if args.retry_prediction_budget_factor < 1.0:
        parser.error("retry-prediction-budget-factor must be >= 1")

    calibration_paths = [args.calibration, *args.calibration_extra]
    calibration_rows = [row for path in calibration_paths for row in load_profile(path)]
    heldout_rows = load_profile(args.heldout)
    model = fit_placement_interaction_cost_model(
        calibration_samples(calibration_rows), ridge=float(args.ridge)
    )
    evaluated = evaluate_heldout(
        heldout_rows,
        model=model,
        budgets_bytes=tuple(args.budgets_mib),
        peak_inflation=float(args.peak_inflation),
        retry_max_attempts=int(args.retry_max_attempts),
        retry_prediction_budget_factor=float(args.retry_prediction_budget_factor),
        calibration_profiles=tuple(str(path) for path in calibration_paths),
        heldout_profile=str(args.heldout),
    )
    summary = summarize(evaluated)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evaluated),
        encoding="utf-8",
    )
    model_path = args.out_dir / "cost_model.json"
    model_path.write_text(
        json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path = args.out_dir / "summary.csv"
    _write_csv(summary_path, summary)
    manifest = {
        "schema_version": "boundflow.pr11-barrier-placement-eval-manifest/v4",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "calibration": [
            {"path": str(path), "sha256": _sha256(path)} for path in calibration_paths
        ],
        "heldout": {"path": str(args.heldout), "sha256": _sha256(args.heldout)},
        "budgets_bytes": list(args.budgets_mib),
        "peak_inflation": float(args.peak_inflation),
        "ridge": float(args.ridge),
        "retry_max_attempts": int(args.retry_max_attempts),
        "retry_strategy": "latency_topology_density_stratified_v3",
        "retry_prediction_budget_factor": float(args.retry_prediction_budget_factor),
        "feature_source": "static_topology_liveness_v1",
        "row_count": len(evaluated),
        "outputs": {
            "raw.jsonl": _sha256(raw_path),
            "cost_model.json": _sha256(model_path),
            "summary.csv": _sha256(summary_path),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
