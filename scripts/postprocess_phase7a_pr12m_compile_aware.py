#!/usr/bin/env python
"""Generate PR-12M CSV/figures/manifest from immutable Planner JSONL."""

# pylint: disable=duplicate-code,import-outside-toplevel,too-many-locals,too-many-statements

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write normalized decisions, backend counts, plots, and hash manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    rows = _read(args.planner)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    normalized = [
        {
            "case_id": row["case_id"],
            "family": row["family"],
            "policy_id": row["policy_id"],
            "budget_mib": (
                "unbounded" if row["budget_mib"] is None else row["budget_mib"]
            ),
            "selected_backend": row["decision"]["backend"],
            "oracle_backend": row["oracle_backend"],
            "amortized_latency_regret": row["amortized_latency_regret"],
            "selected_peak_allocated_bytes": row["selected_peak_allocated_bytes"],
            "any_measured_budget_feasible": row["any_measured_budget_feasible"],
            "selected_measured_budget_feasible": row[
                "selected_measured_budget_feasible"
            ],
            "selected_risk_tier": row["decision"]["selected_risk_tier"],
        }
        for row in rows
    ]
    counts: dict[tuple[str, str], int] = {}
    for row in normalized:
        key = (str(row["policy_id"]), str(row["selected_backend"]))
        counts[key] = counts.get(key, 0) + 1
    count_rows = [
        {"policy_id": policy, "backend": backend, "count": count}
        for (policy, backend), count in sorted(counts.items())
    ]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    decisions_path = args.out_dir / "decisions.csv"
    counts_path = args.out_dir / "backend_counts.csv"
    _write_csv(decisions_path, normalized)
    _write_csv(counts_path, count_rows)
    plots: list[Path] = []
    if not args.no_plots:
        import matplotlib.pyplot as plt

        policies = sorted({str(row["policy_id"]) for row in normalized})
        backends = sorted({str(row["selected_backend"]) for row in normalized})
        figure, axis = plt.subplots(figsize=(7.2, 3.6))
        bottom = [0] * len(policies)
        for backend in backends:
            values = [counts.get((policy, backend), 0) for policy in policies]
            axis.bar(policies, values, bottom=bottom, label=backend)
            bottom = [left + value for left, value in zip(bottom, values)]
        axis.set_ylabel("selected decisions")
        axis.legend(fontsize=7)
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        backend_plot = args.out_dir / "backend_selection_by_reuse.png"
        figure.savefig(backend_plot, dpi=180)
        plt.close(figure)
        plots.append(backend_plot)

        feasible = [row for row in normalized if row["any_measured_budget_feasible"]]
        figure, axis = plt.subplots(figsize=(7.2, 3.6))
        axis.scatter(
            range(len(feasible)),
            [float(row["amortized_latency_regret"]) for row in feasible],
            s=12,
        )
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_ylabel("amortized latency regret")
        axis.set_xlabel("feasible held-out regime")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        regret_plot = args.out_dir / "feasible_regret.png"
        figure.savefig(regret_plot, dpi=180)
        plt.close(figure)
        plots.append(regret_plot)
    outputs = {
        path.name: _sha256(path) for path in (decisions_path, counts_path, *plots)
    }
    manifest = {
        "schema_version": "boundflow.pr12m-compile-aware-report/v1",
        "inputs": {
            "planner.jsonl": _sha256(args.planner),
            "summary.json": _sha256(args.summary),
        },
        "summary": summary,
        "backend_counts": count_rows,
        "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
