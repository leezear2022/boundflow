#!/usr/bin/env python
"""Fit PR-12M from calibration and evaluate immutable multi-budget held-out rows."""

# pylint: disable=too-many-locals

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Optional, Sequence

from boundflow.planner.execution_candidate import BackendVariant
from boundflow.planner.fused_crown_backend import (
    CompileAwareBackendObservation,
    CompileAwareFusedCrownPlanner,
    CompileAwareQueryPolicy,
)
from scripts.benchmark_phase7a_pr12_runtime_pareto import _workload, _write_jsonl

SCHEMA_VERSION = "boundflow.pr12m-compile-aware-evaluation/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def _disk_priors(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    priors: dict[str, float] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        priors[str(row["workload"]["family"])] = float(
            row["amortization"]["disk_first_query_setup_ms"]
        )
    return priors


def _correct(row: dict[str, Any]) -> bool:
    correctness = row.get("correctness", {})
    return bool(
        row.get("status") == "ok"
        and correctness.get("allclose")
        and correctness.get("finite")
        and correctness.get("lower_le_upper")
    )


def _setup(row: dict[str, Any]) -> float:
    runtime = row["runtime"]
    return max(
        0.0,
        float(runtime["compile_first_run_wall_ms"])
        - float(runtime["host_group_per_query"]["median_ms"]),
    )


def _disk_setup(
    backend: BackendVariant, family: str, priors: dict[str, float]
) -> float | None:
    if backend in {
        BackendVariant.PYTORCH_EAGER,
        BackendVariant.PYTORCH_STRUCTURED,
        BackendVariant.PYTORCH_CHUNKED,
    }:
        return 0.0
    if backend == BackendVariant.TVM_FUSED_TIR:
        return priors.get(family)
    return None


def _observations(
    split: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    priors: dict[str, float],
) -> list[CompileAwareBackendObservation]:
    records = {str(row["case_id"]): row for row in split["calibration"]}
    result: list[CompileAwareBackendObservation] = []
    for row in rows:
        if row["benchmark_contract"]["level"] != "end_to_end_final_bound":
            continue
        if row["candidate"]["stream"] != "default":
            continue
        case_id = str(row["workload"]["case_id"])
        workload = _workload(records[case_id], split_role="calibration")
        backend = BackendVariant(row["candidate"]["backend"])
        fresh = _setup(row) if row.get("runtime") else 0.0
        result.append(
            CompileAwareBackendObservation(
                case_id=case_id,
                family=workload.planner_family,
                backend=backend,
                boundary_bytes=workload.boundary_bytes,
                region_count=max(1, workload.expected_regions),
                warm_latency_ms=float(
                    row["runtime"]["host_group_per_query"]["median_ms"]
                ),
                peak_allocated_bytes=int(row["memory"]["peak_allocated_delta_bytes"]),
                fresh_setup_ms=fresh,
                disk_setup_ms=_disk_setup(backend, workload.family, priors),
                eligible=True,
                correct=_correct(row),
            )
        )
    return result


def _actual_amortized_ms(
    row: dict[str, Any], policy: CompileAwareQueryPolicy, disk_setup: float | None
) -> float:
    warm = float(row["runtime"]["host_group_per_query"]["median_ms"])
    fresh = _setup(row)
    disk = fresh if disk_setup is None else disk_setup
    expected_setup = (
        policy.fresh_probability * fresh + policy.disk_cache_hit_probability * disk
    )
    return warm + expected_setup / policy.expected_reuse_queries


def _evaluation_rows(  # pylint: disable=too-many-locals
    split: dict[str, Any],
    planner: CompileAwareFusedCrownPlanner,
    heldout_rows: Sequence[dict[str, Any]],
    priors: dict[str, float],
) -> list[dict[str, Any]]:
    by_case: dict[str, dict[BackendVariant, dict[str, Any]]] = {}
    for row in heldout_rows:
        if (
            row["benchmark_contract"]["level"] == "end_to_end_final_bound"
            and row["candidate"]["stream"] == "default"
            and _correct(row)
        ):
            by_case.setdefault(str(row["workload"]["case_id"]), {})[
                BackendVariant(row["candidate"]["backend"])
            ] = row
    evaluations: list[dict[str, Any]] = []
    for record in split["final_heldout"]:
        workload = _workload(record, split_role="heldout")
        candidates = by_case[workload.case_id]
        eligible = tuple(candidates)
        for policy_record in split["reuse_policies"]:
            policy = CompileAwareQueryPolicy(
                expected_reuse_queries=int(policy_record["expected_reuse_queries"]),
                memory_cache_hit_probability=float(
                    policy_record["memory_cache_hit_probability"]
                ),
                disk_cache_hit_probability=float(
                    policy_record["disk_cache_hit_probability"]
                ),
            )
            for budget_mib in split["budget_mib_sweep"]:
                budget_bytes = (
                    1 << 60 if budget_mib is None else int(budget_mib) * 1024 * 1024
                )
                decision = planner.decide(
                    family=workload.planner_family,
                    boundary_bytes=workload.boundary_bytes,
                    region_count=max(1, workload.expected_regions),
                    budget_bytes=budget_bytes,
                    eligible_backends=eligible,
                    policy=policy,
                )
                actual: dict[BackendVariant, float] = {}
                feasible: dict[BackendVariant, dict[str, Any]] = {}
                for backend, row in candidates.items():
                    disk = _disk_setup(backend, workload.family, priors)
                    actual[backend] = _actual_amortized_ms(row, policy, disk)
                    if int(row["memory"]["peak_allocated_delta_bytes"]) <= budget_bytes:
                        feasible[backend] = row
                oracle_pool = feasible or candidates
                oracle_backend = min(oracle_pool, key=actual.__getitem__)
                selected_row = candidates[decision.backend]
                selected_actual = actual[decision.backend]
                evaluations.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "split_id": split["split_id"],
                        "case_id": workload.case_id,
                        "family": workload.family,
                        "policy_id": policy_record["policy_id"],
                        "budget_mib": budget_mib,
                        "budget_bytes": budget_bytes,
                        "decision": decision.to_dict(),
                        "oracle_backend": oracle_backend.value,
                        "oracle_amortized_latency_ms": actual[oracle_backend],
                        "selected_actual_amortized_latency_ms": selected_actual,
                        "amortized_latency_regret": (
                            selected_actual / actual[oracle_backend]
                        ),
                        "any_measured_budget_feasible": bool(feasible),
                        "selected_measured_budget_feasible": int(
                            selected_row["memory"]["peak_allocated_delta_bytes"]
                        )
                        <= budget_bytes,
                        "selected_peak_allocated_bytes": int(
                            selected_row["memory"]["peak_allocated_delta_bytes"]
                        ),
                        "unsafe_backend": decision.backend not in eligible,
                    }
                )
    return evaluations


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:  # pylint: disable=too-many-locals
    """Fit once from calibration, then replay all frozen held-out regimes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--amortization", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    priors = _disk_priors(_read_jsonl(args.amortization))
    planner = CompileAwareFusedCrownPlanner.fit(
        _observations(split, _read_jsonl(args.calibration), priors)
    )
    evaluations = _evaluation_rows(split, planner, _read_jsonl(args.heldout), priors)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    planner_path = args.out_dir / "planner.jsonl"
    model_path = args.out_dir / "planner_model.json"
    _write_jsonl(planner_path, evaluations)
    model_path.write_text(
        json.dumps(planner.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    regrets = [float(row["amortized_latency_regret"]) for row in evaluations]
    opportunities = [row for row in evaluations if row["any_measured_budget_feasible"]]
    feasible_regrets = [float(row["amortized_latency_regret"]) for row in opportunities]
    selected_counts: dict[str, int] = {}
    for row in evaluations:
        backend = str(row["decision"]["backend"])
        selected_counts[backend] = selected_counts.get(backend, 0) + 1
    summary = {
        "schema_version": "boundflow.pr12m-compile-aware-summary/v1",
        "split_id": split["split_id"],
        "rows": len(evaluations),
        "median_regret": statistics.median(regrets),
        "p90_regret": _percentile(regrets, 0.90),
        "max_regret": max(regrets),
        "feasible_median_regret": statistics.median(feasible_regrets),
        "feasible_p90_regret": _percentile(feasible_regrets, 0.90),
        "feasible_max_regret": max(feasible_regrets),
        "no_measured_feasible_candidate": len(evaluations) - len(opportunities),
        "budget_feasible_opportunities": len(opportunities),
        "selected_budget_feasible": sum(
            bool(row["selected_measured_budget_feasible"]) for row in opportunities
        ),
        "unsafe_backend_count": sum(bool(row["unsafe_backend"]) for row in evaluations),
        "oracle_hits": sum(value <= 1.0 + 1e-9 for value in regrets),
        "selected_backend_counts": selected_counts,
        "disk_setup_priors_ms": priors,
        "inputs": {
            "split": _sha256(args.split_file),
            "calibration": _sha256(args.calibration),
            "heldout": _sha256(args.heldout),
            "amortization": _sha256(args.amortization),
        },
        "outputs": {
            "planner.jsonl": _sha256(planner_path),
            "planner_model.json": _sha256(model_path),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
