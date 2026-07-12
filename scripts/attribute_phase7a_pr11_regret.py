#!/usr/bin/env python
"""Attribute high-regret PR-11 held-out cases before PR-12 lowering."""

# pylint: disable=wrong-import-position,too-many-arguments,duplicate-code

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boundflow.planner.materialization_placement_cost_model import (
    PlacementInteractionCostModel,
)
from scripts.evaluate_phase7a_pr11_barrier_placements import (
    _dense_baselines,
    _pattern,
    features_for_row,
    load_profile,
)

ATTRIBUTION_SCHEMA_VERSION = "boundflow.pr11-regret-attribution/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def classify_regret(
    *,
    oracle_pattern: str,
    attempted_patterns: Sequence[str],
    selected_predicted_latency: float,
    oracle_predicted_latency: float,
    selected_structured_count: int,
    oracle_structured_count: int,
    measurement_variance: bool,
    used_fallback: bool,
) -> tuple[str, tuple[str, ...]]:
    """Return one primary cause plus non-exclusive diagnostic flags."""

    flags: list[str] = []
    if oracle_pattern not in attempted_patterns:
        primary = "CANDIDATE_NOT_AVAILABLE"
    elif used_fallback:
        primary = "OOM_FALLBACK"
    elif selected_predicted_latency <= oracle_predicted_latency:
        primary = "COST_MODEL_MISRANK"
    else:
        primary = "PROFILE_EXTRAPOLATION"
    if measurement_variance:
        flags.append("MEASUREMENT_VARIANCE")
    if selected_structured_count > oracle_structured_count:
        # A hypothesis for PR-12, not proof that fused lowering will close the gap.
        flags.append("BACKEND_GAP")
    return primary, tuple(flags)


def _variance_ratio(row: dict[str, Any]) -> float:
    aggregation = row.get("replicate_aggregation", {})
    minimum = float(aggregation.get("latency_ms_min", 0.0))
    maximum = float(aggregation.get("latency_ms_max", 0.0))
    return maximum / minimum if minimum > 0.0 else 1.0


def attribute_evaluation(  # pylint: disable=too-many-locals
    evaluation_dir: Path,
    profile_path: Path,
    *,
    regret_threshold: float,
    variance_ratio_threshold: float,
) -> list[dict[str, Any]]:
    """Join final evaluator rows with static predictions and aggregate profiles."""

    profile_rows = load_profile(profile_path)
    baselines = _dense_baselines(profile_rows)
    if len(baselines) != 1:
        raise ValueError("regret attribution expects one held-out query per profile")
    baseline = next(iter(baselines.values()))
    by_pattern = {_pattern(row): row for row in profile_rows}
    model = PlacementInteractionCostModel.from_dict(
        json.loads((evaluation_dir / "cost_model.json").read_text(encoding="utf-8"))
    )
    predictions = {
        pattern: model.predict(features_for_row(row, baseline))
        for pattern, row in by_pattern.items()
    }
    evaluated = [
        json.loads(line)
        for line in (evaluation_dir / "raw.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    output: list[dict[str, Any]] = []
    for row in evaluated:
        regret = row["metrics"]["latency_regret_ratio"]
        if row["policy"] != "global_bounded_retry" or regret is None:
            continue
        if float(regret) < float(regret_threshold):
            continue
        selected_pattern = str(row["selected_pattern"])
        oracle_pattern = str(row["oracle"]["pattern"])
        selected_profile = by_pattern[selected_pattern]
        oracle_profile = by_pattern[oracle_pattern]
        attempted = tuple(str(value) for value in row["retry"]["attempted_patterns"])
        selected_prediction = predictions[selected_pattern]
        oracle_prediction = predictions[oracle_pattern]
        selected_variance = _variance_ratio(selected_profile)
        oracle_variance = _variance_ratio(oracle_profile)
        used_fallback = "D" not in selected_pattern and len(attempted) > 1
        primary, flags = classify_regret(
            oracle_pattern=oracle_pattern,
            attempted_patterns=attempted,
            selected_predicted_latency=selected_prediction.latency_ms,
            oracle_predicted_latency=oracle_prediction.latency_ms,
            selected_structured_count=selected_pattern.count("S"),
            oracle_structured_count=oracle_pattern.count("S"),
            measurement_variance=(
                max(selected_variance, oracle_variance) >= variance_ratio_threshold
            ),
            used_fallback=used_fallback,
        )
        budget = int(row["budget_bytes"])
        output.append(
            {
                "schema_version": ATTRIBUTION_SCHEMA_VERSION,
                "case_id": (
                    f"{row['workload']['name']}:CROWN:s{row['spec_size']}:"
                    f"d{row['domain_batch_size']}:budget{budget}"
                ),
                "workload_family": row["workload"]["name"],
                "bound_method": "CROWN",
                "requires_grad": False,
                "memory_budget_bytes": budget,
                "selected_plan": selected_pattern,
                "oracle_plan": oracle_pattern,
                "attempted_plans": list(attempted),
                "predicted_latency_selected_ms": selected_prediction.latency_ms,
                "predicted_latency_oracle_ms": oracle_prediction.latency_ms,
                "actual_latency_selected_ms": row["actual"]["latency_ms"],
                "actual_latency_oracle_ms": row["oracle"]["latency_ms"],
                "selected_peak_memory_bytes": row["actual"]["peak_bytes"],
                "oracle_peak_memory_bytes": row["oracle"]["peak_bytes"],
                "selected_measurement_max_min_ratio": selected_variance,
                "oracle_measurement_max_min_ratio": oracle_variance,
                "regret": float(regret),
                "attribution": primary,
                "attribution_flags": list(flags),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    flattened = [
        {
            **row,
            "attempted_plans": json.dumps(row["attempted_plans"]),
            "attribution_flags": json.dumps(row["attribution_flags"]),
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Attribute one or more paired final evaluation/profile artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--regret-threshold", type=float, default=1.5)
    parser.add_argument("--variance-ratio-threshold", type=float, default=1.25)
    args = parser.parse_args(argv)
    if len(args.evaluation) != len(args.profile):
        parser.error("evaluation/profile counts must match")
    if args.regret_threshold < 1.0 or args.variance_ratio_threshold < 1.0:
        parser.error("regret and variance thresholds must be >= 1")

    rows = [
        row
        for evaluation, profile in zip(args.evaluation, args.profile)
        for row in attribute_evaluation(
            evaluation,
            profile,
            regret_threshold=float(args.regret_threshold),
            variance_ratio_threshold=float(args.variance_ratio_threshold),
        )
    ]
    if not rows:
        raise ValueError("no final cases met the regret threshold")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    csv_path = args.out_dir / "attribution.csv"
    _write_csv(csv_path, rows)
    counts = {
        attribution: sum(row["attribution"] == attribution for row in rows)
        for attribution in sorted({str(row["attribution"]) for row in rows})
    }
    manifest = {
        "schema_version": "boundflow.pr11-regret-attribution-manifest/v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "regret_threshold": float(args.regret_threshold),
        "variance_ratio_threshold": float(args.variance_ratio_threshold),
        "case_count": len(rows),
        "attribution_counts": counts,
        "sources": [
            {
                "evaluation": str(evaluation),
                "evaluation_manifest_sha256": _sha256(evaluation / "manifest.json"),
                "profile": str(profile),
                "profile_sha256": _sha256(profile),
            }
            for evaluation, profile in zip(args.evaluation, args.profile)
        ],
        "outputs": {
            "raw.jsonl": _sha256(raw_path),
            "attribution.csv": _sha256(csv_path),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
