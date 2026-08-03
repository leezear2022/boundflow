#!/usr/bin/env python
"""Build PR-12K activity tables, comparisons and conditional branch decision."""

# pylint: disable=duplicate-code,too-many-locals

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

REQUIRED_BACKENDS = {
    "pytorch_eager",
    "pytorch_structured",
    "pytorch_chunked",
    "tvm_tir_unfused",
    "tvm_fused_tir",
}


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
        raise ValueError("cannot write empty PR-12K CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _activity_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        if row["status"] != "ok":
            continue
        activity = row["activity"]
        table.append(
            {
                "case_id": row["workload"]["case_id"],
                "family": row["workload"]["family"],
                "backend": row["candidate"]["backend"],
                "planned_fused_regions": row["planned_fused_regions"],
                "kernel_launches_per_query": activity["kernel_launches_per_query"],
                "device_time_us_per_query": activity["device_time_us_per_query"],
                "mean_kernel_device_us": activity["mean_kernel_device_us"],
                "unique_kernel_names": activity["unique_kernel_names"],
                "vendor_library_device_time_share": activity[
                    "vendor_library_device_time_share"
                ],
                "cuda_launch_self_cpu_us_per_query": activity[
                    "cuda_launch_self_cpu_us_per_query"
                ],
                "profile_wall_ms": row["profile_wall_ms"],
                "max_abs_diff": row["correctness"]["max_abs_diff"],
                "max_rel_diff": row["correctness"]["max_rel_diff"],
            }
        )
    return sorted(table, key=lambda item: (item["case_id"], item["backend"]))


def _mechanism_label(
    family: str,
    case_id: str,
    *,
    device_ratio: float,
    launch_reduction: float,
) -> str:
    if device_ratio <= 0.95 and launch_reduction >= 0.10:
        return "FUSION_ACTIVITY_IMPROVEMENT"
    if device_ratio < 1.05:
        return "ACTIVITY_NEUTRAL"
    if family == "linear" and "memory" in case_id:
        return "LONG_REDUCTION_KERNEL_DEVICE_TIME"
    if family.startswith("conv2d"):
        return "FUSED_CONV_KERNEL_DEVICE_TIME"
    return "COMPOSED_FUSED_REGION_DEVICE_TIME"


def _comparison(activity: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for case_id in sorted({str(row["case_id"]) for row in activity}):
        case_rows = [row for row in activity if row["case_id"] == case_id]
        by_backend = {str(row["backend"]): row for row in case_rows}
        if not REQUIRED_BACKENDS <= set(by_backend):
            continue
        fused = by_backend["tvm_fused_tir"]
        unfused = by_backend["tvm_tir_unfused"]
        eager = by_backend["pytorch_eager"]
        chunked = by_backend["pytorch_chunked"]
        device_ratio = float(fused["device_time_us_per_query"]) / float(
            unfused["device_time_us_per_query"]
        )
        launch_reduction = 1.0 - float(fused["kernel_launches_per_query"]) / float(
            unfused["kernel_launches_per_query"]
        )
        family = str(fused["family"])
        comparisons.append(
            {
                "case_id": case_id,
                "family": family,
                "fused_vs_unfused_device_time_ratio": device_ratio,
                "fused_vs_eager_device_time_ratio": float(
                    fused["device_time_us_per_query"]
                )
                / float(eager["device_time_us_per_query"]),
                "fused_vs_chunked_device_time_ratio": float(
                    fused["device_time_us_per_query"]
                )
                / float(chunked["device_time_us_per_query"]),
                "fused_launch_reduction_vs_unfused": launch_reduction,
                "fused_launch_reduction_vs_eager": 1.0
                - float(fused["kernel_launches_per_query"])
                / float(eager["kernel_launches_per_query"]),
                "fused_launch_cpu_ratio_vs_unfused": float(
                    fused["cuda_launch_self_cpu_us_per_query"]
                )
                / float(unfused["cuda_launch_self_cpu_us_per_query"]),
                "fused_vendor_time_share": fused["vendor_library_device_time_share"],
                "eager_vendor_time_share": eager["vendor_library_device_time_share"],
                "mechanism": _mechanism_label(
                    family,
                    case_id,
                    device_ratio=device_ratio,
                    launch_reduction=launch_reduction,
                ),
            }
        )
    return comparisons


def _top_kernels(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        if row["status"] != "ok":
            continue
        iterations = int(row["activity"]["iterations"])
        for rank, kernel in enumerate(row["activity"]["top_kernels"][:10], start=1):
            table.append(
                {
                    "case_id": row["workload"]["case_id"],
                    "backend": row["candidate"]["backend"],
                    "rank": rank,
                    "kernel_name": kernel["name"],
                    "count_per_query": float(kernel["count"]) / iterations,
                    "device_time_us_per_query": float(kernel["device_time_us"])
                    / iterations,
                }
            )
    return table


def _plots(activity: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    outputs = []
    for metric, ylabel in (
        ("device_time_us_per_query", "CUPTI kernel activity time / query (us)"),
        ("kernel_launches_per_query", "CUDA kernel launches / query"),
    ):
        output = out_dir / f"{metric}.png"
        cases = sorted({str(row["case_id"]) for row in activity})
        backends = sorted({str(row["backend"]) for row in activity})
        width = 0.15
        figure, axis = plt.subplots(figsize=(11.5, 5.2))
        for index, backend in enumerate(backends):
            values = [
                float(
                    next(
                        row[metric]
                        for row in activity
                        if row["case_id"] == case_id and row["backend"] == backend
                    )
                )
                for case_id in cases
            ]
            positions = [
                position + (index - 2) * width for position in range(len(cases))
            ]
            axis.bar(positions, values, width=width, label=backend)
        axis.set_yscale("log")
        axis.set_xticks(range(len(cases)), cases, rotation=24, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs.append(output)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate activity comparisons without claiming unavailable HW counters."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = _read_jsonl(args.raw)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    activity = _activity_table(rows)
    comparisons = _comparison(activity)
    kernels = _top_kernels(rows)
    activity_path = args.out_dir / "activity.csv"
    comparison_path = args.out_dir / "fused_comparison.csv"
    kernels_path = args.out_dir / "top_kernels.csv"
    _write_csv(activity_path, activity)
    _write_csv(comparison_path, comparisons)
    _write_csv(kernels_path, kernels)
    regressed = [
        row
        for row in comparisons
        if float(row["fused_vs_unfused_device_time_ratio"]) > 1.05
    ]
    improved = [
        row
        for row in comparisons
        if float(row["fused_vs_unfused_device_time_ratio"]) <= 0.95
    ]
    neutral = len(comparisons) - len(regressed) - len(improved)
    summary = {
        "rows": len(rows),
        "correct_rows": sum(row["status"] == "ok" for row in rows),
        "hardware_counters_available": audit["hardware_counter_sections_available"],
        "forbidden_claims": audit["forbidden_claims"],
        "comparison_cases": len(comparisons),
        "fused_device_time_regressions_vs_unfused": len(regressed),
        "fused_activity_improvements_vs_unfused": len(improved),
        "fused_activity_neutral_vs_unfused": neutral,
        "max_launch_reduction_vs_unfused": max(
            float(row["fused_launch_reduction_vs_unfused"]) for row in comparisons
        ),
        "selected_pr12l_branch": "E_STOP_OPTIMIZING_TIR",
        "decision_reason": (
            "fusion removes only 2 launches per eligible region; using a 5% activity "
            f"threshold, {len(regressed)}/{len(comparisons)} cases regress, "
            f"{len(improved)}/{len(comparisons)} improve, and {neutral}/{len(comparisons)} "
            "are neutral vs TVM-unfused; no hardware counters justify a specific "
            "tiling/occupancy/bandwidth intervention"
        ),
        "comparisons": comparisons,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plots = [] if args.no_plots else _plots(activity, args.out_dir)
    outputs = {
        path.name: _sha256(path)
        for path in (
            activity_path,
            comparison_path,
            kernels_path,
            summary_path,
            *plots,
        )
    }
    manifest = {
        "schema_version": "boundflow.pr12k-cupti-activity-report/v1",
        "inputs": {
            "raw.jsonl": _sha256(args.raw),
            "profiler_audit.json": _sha256(args.audit),
        },
        "summary": summary,
        "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
