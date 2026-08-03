#!/usr/bin/env python
"""Convert PR-12E/F raw JSONL into CSV summaries and Pareto figures."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

SCHEMA_VERSION = "boundflow.pr12-runtime-pareto-report/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _paired_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["candidate"]["stream"] != "default" or row["status"] != "ok":
            continue
        grouped.setdefault(row["workload"]["case_id"], {})[
            row["candidate"]["backend"]
        ] = row
    output: list[dict[str, Any]] = []
    for case_id, candidates in sorted(grouped.items()):
        eager = candidates.get("pytorch_eager")
        fused = candidates.get("tvm_fused_tir")
        if eager is None or fused is None:
            continue
        eager_ms = float(eager["runtime"]["host_group_per_query"]["median_ms"])
        fused_ms = float(fused["runtime"]["host_group_per_query"]["median_ms"])
        eager_peak = int(eager["memory"]["peak_allocated_delta_bytes"])
        fused_peak = int(fused["memory"]["peak_allocated_delta_bytes"])
        compile_ms = float(fused["runtime"]["estimated_compile_overhead_ms"])
        break_even = compile_ms / (eager_ms - fused_ms) if eager_ms > fused_ms else None
        output.append(
            {
                "role": eager["split"]["role"],
                "case_id": case_id,
                "family": eager["workload"]["family"],
                "boundary_bytes": int(eager["workload"]["boundary_bytes"]),
                "budget_bytes": int(eager["workload"]["budget_bytes"]),
                "fused_eligible": bool(fused["candidate"]["eligible"]),
                "eager_latency_ms": eager_ms,
                "fused_latency_ms": fused_ms,
                "warm_speedup": eager_ms / fused_ms,
                "eager_peak_bytes": eager_peak,
                "fused_peak_bytes": fused_peak,
                "fused_over_eager_peak": fused_peak / eager_peak,
                "compile_overhead_ms": compile_ms,
                "break_even_queries": break_even,
                "max_abs_diff": float(fused["correctness"]["max_abs_diff"]),
                "max_rel_diff": float(fused["correctness"].get("max_rel_diff", 0.0)),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty summary")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    data = [row for row in rows if row["role"] != "fallback_control"]
    latency_memory = out_dir / "latency_vs_peak_memory.png"
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    for backend, marker, latency_key, memory_key in (
        ("PyTorch eager", "o", "eager_latency_ms", "eager_peak_bytes"),
        ("TVM fused TIR", "^", "fused_latency_ms", "fused_peak_bytes"),
    ):
        axis.scatter(
            [row[memory_key] / (1024 * 1024) for row in data],
            [row[latency_key] for row in data],
            marker=marker,
            label=backend,
        )
    for row in data:
        axis.annotate(
            row["case_id"],
            (row["fused_peak_bytes"] / (1024 * 1024), row["fused_latency_ms"]),
            fontsize=6,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Peak allocated delta (MiB, log scale)")
    axis.set_ylabel("Warm host latency (ms, log scale)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(latency_memory, dpi=180)
    plt.close(figure)

    speedup = out_dir / "speedup_vs_boundary_bytes.png"
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.scatter(
        [row["boundary_bytes"] / (1024 * 1024) for row in data],
        [row["warm_speedup"] for row in data],
    )
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    for row in data:
        axis.annotate(
            row["case_id"],
            (row["boundary_bytes"] / (1024 * 1024), row["warm_speedup"]),
            fontsize=6,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Elided scaled-A logical bytes (MiB, log scale)")
    axis.set_ylabel("Warm speedup (eager / fused)")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(speedup, dpi=180)
    plt.close(figure)
    return [latency_memory, speedup]


def _planner_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    heldout = [row for row in rows if row["role"] == "heldout"]
    regrets = sorted(float(row["latency_regret"]) for row in heldout)
    profitable_or_required = sum(
        row["decision"]["reason"]
        in {
            "calibration_predicts_fused_faster",
            "fused_only_budget_feasible",
            "calibration_predicts_eager_faster",
            "eager_only_budget_feasible",
        }
        and (
            float(row["latency_regret"]) <= 1.0 + 1e-9
            or row["decision"]["reason"].endswith("only_budget_feasible")
        )
        for row in heldout
    )
    p90_index = min(len(regrets) - 1, math.ceil(0.9 * len(regrets)) - 1)
    return {
        "heldout_cases": len(heldout),
        "feasible_cases": sum(row["selected_budget_feasible"] for row in heldout),
        "unsafe_fusion_count": sum(row["unsafe_fusion"] for row in rows),
        "median_latency_regret": regrets[len(regrets) // 2],
        "p90_latency_regret": regrets[p90_index],
        "max_latency_regret": regrets[-1],
        "profitable_or_budget_required_selections": profitable_or_required,
        "fallback_controls": sum(row["role"] == "fallback_control" for row in rows),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write deterministic report files from immutable raw inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    calibration = _paired_rows(_read_jsonl(args.calibration))
    heldout = _paired_rows(_read_jsonl(args.heldout))
    combined = [*calibration, *heldout]
    summary_path = args.out_dir / "candidate_summary.csv"
    _write_csv(summary_path, combined)
    planner_rows = _read_jsonl(args.planner)
    planner_summary = _planner_summary(planner_rows)
    planner_path = args.out_dir / "planner_summary.json"
    planner_path.write_text(
        json.dumps(planner_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_paths = [] if args.no_plots else _plot(combined, args.out_dir)
    outputs = {
        summary_path.name: _sha256(summary_path),
        planner_path.name: _sha256(planner_path),
        **{path.name: _sha256(path) for path in plot_paths},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "calibration": {
                "path": str(args.calibration),
                "sha256": _sha256(args.calibration),
            },
            "heldout": {"path": str(args.heldout), "sha256": _sha256(args.heldout)},
            "planner": {"path": str(args.planner), "sha256": _sha256(args.planner)},
        },
        "candidate_rows": len(combined),
        "planner": planner_summary,
        "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
