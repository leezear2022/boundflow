#!/usr/bin/env python
"""Build PR-12I fair-baseline tables, Pareto figures and manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
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
        raise ValueError("cannot write an empty PR-12I table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _is_pareto(row: dict[str, Any], peers: Sequence[dict[str, Any]]) -> bool:
    latency = float(row["warm_latency_ms"])
    peak = int(row["peak_allocated_bytes"])
    return not any(
        float(peer["warm_latency_ms"]) <= latency
        and int(peer["peak_allocated_bytes"]) <= peak
        and (
            float(peer["warm_latency_ms"]) < latency
            or int(peer["peak_allocated_bytes"]) < peak
        )
        for peer in peers
        if peer is not row
    )


def _table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    default_rows = [
        row
        for row in rows
        if row["candidate"]["stream"] == "default" and row["status"] == "ok"
    ]
    eager = {
        (row["benchmark_contract"]["level"], row["workload"]["case_id"]): row
        for row in default_rows
        if row["candidate"]["backend"] == "pytorch_eager"
    }
    table = []
    for row in default_rows:
        key = (row["benchmark_contract"]["level"], row["workload"]["case_id"])
        reference = eager[key]
        latency = float(row["runtime"]["host_group_per_query"]["median_ms"])
        eager_latency = float(reference["runtime"]["host_group_per_query"]["median_ms"])
        peak = int(row["memory"]["peak_allocated_delta_bytes"])
        eager_peak = int(reference["memory"]["peak_allocated_delta_bytes"])
        table.append(
            {
                "contract": key[0],
                "case_id": key[1],
                "family": row["workload"]["family"],
                "backend": row["candidate"]["backend"],
                "warm_latency_ms": latency,
                "speedup_vs_eager": eager_latency / latency,
                "peak_allocated_bytes": peak,
                "peak_ratio_vs_eager": peak / eager_peak,
                "compile_overhead_ms": row["runtime"]["estimated_compile_overhead_ms"],
                "max_abs_diff": row["correctness"]["max_abs_diff"],
                "max_rel_diff": row["correctness"]["max_rel_diff"],
                "pareto": False,
            }
        )
    for row in table:
        peers = [
            peer
            for peer in table
            if peer["contract"] == row["contract"] and peer["case_id"] == row["case_id"]
        ]
        row["pareto"] = _is_pareto(row, peers)
    return sorted(
        table, key=lambda row: (row["contract"], row["case_id"], row["backend"])
    )


def _summary(
    table: Sequence[dict[str, Any]], raw: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    contracts: dict[str, Any] = {}
    for contract in sorted({str(row["contract"]) for row in table}):
        subset = [row for row in table if row["contract"] == contract]
        backends: dict[str, Any] = {}
        for backend in sorted({str(row["backend"]) for row in subset}):
            backend_rows = [row for row in subset if row["backend"] == backend]
            speedups = [float(row["speedup_vs_eager"]) for row in backend_rows]
            backends[backend] = {
                "cases": len(backend_rows),
                "geomean_speedup_vs_eager": math.exp(
                    statistics.fmean(math.log(value) for value in speedups)
                ),
                "median_peak_ratio_vs_eager": statistics.median(
                    float(row["peak_ratio_vs_eager"]) for row in backend_rows
                ),
                "pareto_cases": sum(bool(row["pareto"]) for row in backend_rows),
            }
        contracts[contract] = backends
    statuses: dict[str, int] = {}
    for row in raw:
        status = str(row["status"])
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "rows": len(raw),
        "status_counts": statuses,
        "correctness_failures": sum(row["status"] == "fail" for row in raw),
        "contracts": contracts,
    }


def _plots(table: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    outputs = []
    for contract in sorted({str(row["contract"]) for row in table}):
        subset = [row for row in table if row["contract"] == contract]
        output = out_dir / f"{contract}_latency_vs_peak.png"
        figure, axis = plt.subplots(figsize=(7.4, 4.8))
        for backend in sorted({str(row["backend"]) for row in subset}):
            points = [row for row in subset if row["backend"] == backend]
            axis.scatter(
                [float(row["peak_allocated_bytes"]) / 1048576 for row in points],
                [float(row["warm_latency_ms"]) for row in points],
                label=backend,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Peak allocated delta (MiB)")
        axis.set_ylabel("Warm host latency (ms)")
        axis.set_title(contract)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs.append(output)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate all PR-12I derived evidence from one immutable raw JSONL."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    raw = _read_jsonl(args.raw)
    table = _table(raw)
    summary = _summary(table, raw)
    table_path = args.out_dir / "baseline.csv"
    summary_path = args.out_dir / "summary.json"
    _write_csv(table_path, table)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plots = [] if args.no_plots else _plots(table, args.out_dir)
    outputs = {path.name: _sha256(path) for path in (table_path, summary_path, *plots)}
    manifest = {
        "schema_version": "boundflow.pr12i-baseline-report/v1",
        "inputs": {"raw.jsonl": _sha256(args.raw)},
        "table_rows": len(table),
        "summary": summary,
        "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
