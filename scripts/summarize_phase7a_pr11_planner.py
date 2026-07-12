#!/usr/bin/env python
"""Summarize PR-11 held-out evaluation JSONL into a policy CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

SUMMARY_SCHEMA_VERSION = "boundflow.pr11-planner-summary/v1"
FIELDS = (
    "schema_version",
    "scope",
    "method",
    "policy",
    "rows",
    "oracle_feasible",
    "planner_feasible",
    "feasible_coverage",
    "unexpected_failures",
    "dense_actions",
    "structured_actions",
    "reduce_batch_actions",
    "regret_samples",
    "latency_regret_median",
    "latency_regret_p90",
    "latency_regret_p99",
    "latency_regret_max",
)


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * float(fraction)))),
    )
    return float(ordered[index])


def _aggregate(
    rows: Iterable[dict[str, Any]], *, scope: str, method: str, policy: str
) -> dict[str, object]:
    selected = list(rows)
    actions = [row["decision"]["plan"]["action"] for row in selected]
    regrets = [
        float(row["metrics"]["latency_regret_ratio"])
        for row in selected
        if row["metrics"]["latency_regret_ratio"] is not None
    ]
    oracle_feasible = sum(bool(row["metrics"]["oracle_feasible"]) for row in selected)
    planner_feasible = sum(bool(row["metrics"]["feasible"]) for row in selected)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "scope": scope,
        "method": method,
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
        "dense_actions": actions.count("dense"),
        "structured_actions": actions.count("structured"),
        "reduce_batch_actions": actions.count("reduce_batch"),
        "regret_samples": len(regrets),
        "latency_regret_median": _percentile(regrets, 0.5),
        "latency_regret_p90": _percentile(regrets, 0.9),
        "latency_regret_p99": _percentile(regrets, 0.99),
        "latency_regret_max": max(regrets) if regrets else None,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, object]]:
    """Aggregate every policy overall and once per bound method."""

    policies = sorted({row["decision"]["plan"]["policy"] for row in rows})
    methods = sorted({row["workload"]["method"] for row in rows})
    output: list[dict[str, object]] = []
    for policy in policies:
        policy_rows = [
            row for row in rows if row["decision"]["plan"]["policy"] == policy
        ]
        output.append(_aggregate(policy_rows, scope="all", method="ALL", policy=policy))
        for method in methods:
            output.append(
                _aggregate(
                    [row for row in policy_rows if row["workload"]["method"] == method],
                    scope="method",
                    method=method,
                    policy=policy,
                )
            )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read evaluation JSONL and write deterministic aggregate CSV."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
