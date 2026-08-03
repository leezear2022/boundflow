#!/usr/bin/env python
"""Build PR-12G three-candidate Pareto tables, figures, and manifest."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Optional, Sequence

BACKEND_ORDER = ("pytorch_eager", "pytorch_chunked", "tvm_fused_tir")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    defaults = [
        row
        for row in rows
        if row["status"] == "ok" and row["candidate"]["stream"] == "default"
    ]
    eager_by_case = {
        row["workload"]["case_id"]: row
        for row in defaults
        if row["candidate"]["backend"] == "pytorch_eager"
    }
    table = []
    for row in sorted(
        defaults,
        key=lambda item: (
            item["split"]["role"],
            item["workload"]["case_id"],
            BACKEND_ORDER.index(item["candidate"]["backend"]),
        ),
    ):
        eager = eager_by_case[row["workload"]["case_id"]]
        latency = float(row["runtime"]["host_group_per_query"]["median_ms"])
        eager_latency = float(eager["runtime"]["host_group_per_query"]["median_ms"])
        peak = int(row["memory"]["peak_allocated_delta_bytes"])
        eager_peak = int(eager["memory"]["peak_allocated_delta_bytes"])
        table.append(
            {
                "role": row["split"]["role"],
                "case_id": row["workload"]["case_id"],
                "family": row["workload"]["family"],
                "backend": row["candidate"]["backend"],
                "eligible": row["candidate"]["eligible"],
                "boundary_bytes": row["workload"]["boundary_bytes"],
                "budget_bytes": row["workload"]["budget_bytes"],
                "warm_latency_ms": latency,
                "speedup_vs_eager": eager_latency / latency,
                "peak_allocated_bytes": peak,
                "peak_over_eager": peak / eager_peak,
                "compile_overhead_ms": row["runtime"]["estimated_compile_overhead_ms"],
                "max_abs_diff": row["correctness"]["max_abs_diff"],
                "max_rel_diff": row["correctness"].get("max_rel_diff", 0.0),
            }
        )
    return table


def _planner_table(
    planner_rows: Sequence[dict[str, Any]], candidates: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(row["case_id"], row["backend"]): row for row in candidates}
    output = []
    selected_speedups = []
    selected_backends: Counter[str] = Counter()
    regrets = []
    fallback_controls = 0
    for row in planner_rows:
        selected = row["decision"]["backend"]
        candidate = by_key[(row["case_id"], selected)]
        speedup = float(candidate["speedup_vs_eager"])
        if row["role"] == "heldout":
            selected_backends[selected] += 1
            selected_speedups.append(speedup)
            regrets.append(float(row["latency_regret"]))
        else:
            fallback_controls += 1
        output.append(
            {
                "role": row["role"],
                "case_id": row["case_id"],
                "family": row["family"],
                "selected_backend": selected,
                "oracle_backend": row["oracle_backend"],
                "latency_regret": row["latency_regret"],
                "selected_speedup_vs_eager": speedup,
                "selected_peak_allocated_bytes": row["selected_peak_allocated_bytes"],
                "budget_bytes": row["budget_bytes"],
                "budget_feasible": row["selected_budget_feasible"],
                "unsafe_fusion": row["unsafe_fusion"],
                "planner_overhead_ms": row["planner_overhead_ms"],
            }
        )
    ordered_regrets = sorted(regrets)
    p90_index = min(len(ordered_regrets) - 1, math.ceil(0.9 * len(ordered_regrets)) - 1)
    summary = {
        "heldout_cases": len(regrets),
        "oracle_hits": sum(regret <= 1.0 + 1e-9 for regret in regrets),
        "budget_feasible": sum(
            row["budget_feasible"] for row in output if row["role"] == "heldout"
        ),
        "unsafe_fusion_count": sum(row["unsafe_fusion"] for row in output),
        "median_latency_regret": statistics.median(regrets),
        "p90_latency_regret": ordered_regrets[p90_index],
        "geomean_selected_speedup_vs_eager": math.exp(
            statistics.fmean(math.log(value) for value in selected_speedups)
        ),
        "selected_backend_counts": dict(sorted(selected_backends.items())),
        "fallback_controls": fallback_controls,
        "max_planner_overhead_ms": max(row["planner_overhead_ms"] for row in output),
    }
    return output, summary


def _plot(candidates: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    data = [row for row in candidates if row["role"] == "heldout"]
    latency_memory = out_dir / "multibackend_latency_vs_peak.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for backend, marker in zip(BACKEND_ORDER, ("o", "s", "^")):
        subset = [row for row in data if row["backend"] == backend]
        axis.scatter(
            [row["peak_allocated_bytes"] / 1048576 for row in subset],
            [row["warm_latency_ms"] for row in subset],
            marker=marker,
            label=backend,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Peak allocated delta (MiB)")
    axis.set_ylabel("Warm host latency (ms)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(latency_memory, dpi=180)
    plt.close(figure)

    scaling = out_dir / "multibackend_speedup_vs_boundary.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for backend, marker in zip(BACKEND_ORDER[1:], ("s", "^")):
        subset = [row for row in data if row["backend"] == backend]
        axis.scatter(
            [row["boundary_bytes"] / 1048576 for row in subset],
            [row["speedup_vs_eager"] for row in subset],
            marker=marker,
            label=backend,
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlabel("Elided scaled-A logical bytes (MiB)")
    axis.set_ylabel("Speedup vs PyTorch eager")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(scaling, dpi=180)
    plt.close(figure)
    return [latency_memory, scaling]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate the complete v2 report from immutable raw evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    candidates = _candidate_table(
        [*_read_jsonl(args.calibration), *_read_jsonl(args.heldout)]
    )
    planner_rows = _read_jsonl(args.planner)
    planner_table, summary = _planner_table(planner_rows, candidates)
    candidate_path = args.out_dir / "candidates.csv"
    planner_path = args.out_dir / "planner.csv"
    summary_path = args.out_dir / "summary.json"
    _write_csv(candidate_path, candidates)
    _write_csv(planner_path, planner_table)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plots = [] if args.no_plots else _plot(candidates, args.out_dir)
    outputs = {
        path.name: _sha256(path)
        for path in (candidate_path, planner_path, summary_path, *plots)
    }
    manifest = {
        "schema_version": "boundflow.pr12-multibackend-report/v2",
        "inputs": {
            "calibration": _sha256(args.calibration),
            "heldout": _sha256(args.heldout),
            "planner": _sha256(args.planner),
        },
        "candidate_rows": len(candidates),
        "planner_rows": len(planner_table),
        "summary": summary,
        "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
