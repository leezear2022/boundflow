#!/usr/bin/env python
"""Derive PR-12J compile-phase and repeated-query break-even evidence."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence


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
        raise ValueError("cannot write empty PR-12J CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _phase_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        phases = row["compile_phase_totals_ms"]
        restart_events = row["process_restart_disk_cache"]["cache_events"]
        fresh = row["amortization"]["fresh_break_even"]
        disk = row["amortization"]["disk_break_even"]
        process_restart = row["amortization"]["process_restart_break_even"]
        table.append(
            {
                "case_id": row["workload"]["case_id"],
                "family": row["workload"]["family"],
                "unique_modules": row["unique_compiled_modules"],
                "planner_ir_construction_ms": row["runtime"][
                    "planner_ir_construction_ms"
                ],
                "cache_lookup_ms": phases["cache_lookup_ms"],
                "tir_generation_ms": phases["tir_generation_ms"],
                "schedule_ms": phases["schedule_ms"],
                "tvm_compile_ms": phases["tvm_compile_ms"],
                "serialization_ms": phases["serialization_ms"],
                "first_query_wall_ms": row["runtime"]["first_query_wall_ms"],
                "warm_query_ms": row["amortization"]["candidate_warm_ms"],
                "disk_module_load_ms": sum(
                    float(event["module_load_ms"])
                    for event in restart_events
                    if event["event"] == "disk_hit"
                ),
                "restart_first_query_wall_ms": row["process_restart_disk_cache"][
                    "first_query_wall_ms"
                ],
                "restart_process_wall_ms": row["process_restart_disk_cache"][
                    "process_wall_ms"
                ],
                "fresh_vs_eager_status": fresh["pytorch_eager"]["status"],
                "fresh_vs_eager_queries": fresh["pytorch_eager"]["queries"],
                "disk_vs_eager_status": disk["pytorch_eager"]["status"],
                "disk_vs_eager_queries": disk["pytorch_eager"]["queries"],
                "process_vs_eager_status": process_restart["pytorch_eager"]["status"],
                "process_vs_eager_queries": process_restart["pytorch_eager"]["queries"],
                "correct": row["status"] == "ok",
            }
        )
    return table


def _query_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        case_id = row["workload"]["case_id"]
        fresh = row["amortization"]["fresh_query_totals"]
        disk = row["amortization"]["disk_query_totals"]
        process_restart = row["amortization"]["process_restart_query_totals"]
        for fresh_point, disk_point, process_point in zip(fresh, disk, process_restart):
            queries = int(fresh_point["queries"])
            values = {
                "tvm_fused_fresh": fresh_point["fresh_or_disk_candidate_ms"],
                "tvm_fused_disk_restart": disk_point["fresh_or_disk_candidate_ms"],
                "tvm_fused_memory_hit": fresh_point["memory_cache_candidate_ms"],
                "tvm_fused_process_restart": process_point[
                    "fresh_or_disk_candidate_ms"
                ],
                **fresh_point["baselines_ms"],
            }
            for mode, total_ms in values.items():
                table.append(
                    {
                        "case_id": case_id,
                        "queries": queries,
                        "mode": mode,
                        "total_ms": total_ms,
                    }
                )
    return table


def _plots(query_table: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    outputs = []
    for case_id in sorted({str(row["case_id"]) for row in query_table}):
        subset = [row for row in query_table if row["case_id"] == case_id]
        output = out_dir / f"{case_id}_amortization.png"
        figure, axis = plt.subplots(figsize=(7.4, 4.8))
        for mode in sorted({str(row["mode"]) for row in subset}):
            points = [row for row in subset if row["mode"] == mode]
            axis.plot(
                [int(row["queries"]) for row in points],
                [float(row["total_ms"]) for row in points],
                marker="o",
                markersize=3,
                label=mode,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Repeated queries")
        axis.set_ylabel("Modeled total time (ms)")
        axis.set_title(case_id)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs.append(output)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build all PR-12J tables/figures from immutable raw JSONL."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = _read_jsonl(args.raw)
    phases = _phase_table(rows)
    queries = _query_table(rows)
    phase_path = args.out_dir / "compile_phases.csv"
    query_path = args.out_dir / "amortization.csv"
    _write_csv(phase_path, phases)
    _write_csv(query_path, queries)
    summary = {
        "rows": len(rows),
        "correct_rows": sum(row["status"] == "ok" for row in rows),
        "fresh_amortizable_vs_eager": sum(
            row["fresh_vs_eager_status"] != "not_amortizable" for row in phases
        ),
        "disk_amortizable_vs_eager": sum(
            row["disk_vs_eager_status"] != "not_amortizable" for row in phases
        ),
        "process_restart_amortizable_vs_eager": sum(
            row["process_vs_eager_status"] != "not_amortizable" for row in phases
        ),
        "phase_rows": phases,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plots = [] if args.no_plots else _plots(queries, args.out_dir)
    outputs = {
        path.name: _sha256(path)
        for path in (phase_path, query_path, summary_path, *plots)
    }
    manifest = {
        "schema_version": "boundflow.pr12j-amortization-report/v1",
        "inputs": {"raw.jsonl": _sha256(args.raw)},
        "summary": summary,
        "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
