"""Mechanism classification checks for PR-12K activity postprocessing."""

import json
from pathlib import Path

from scripts.postprocess_phase7a_pr12k_cupti import main


def _row(backend: str, device_us: float, launches: float) -> dict[str, object]:
    return {
        "status": "ok",
        "workload": {"case_id": "linear-memory-sensitive", "family": "linear"},
        "candidate": {"backend": backend},
        "planned_fused_regions": int(
            backend not in {"pytorch_eager", "pytorch_structured"}
        ),
        "profile_wall_ms": 1.0,
        "activity": {
            "iterations": 1,
            "kernel_launches_per_query": launches,
            "device_time_us_per_query": device_us,
            "mean_kernel_device_us": device_us / launches,
            "unique_kernel_names": 1,
            "vendor_library_device_time_share": 0.5,
            "cuda_launch_self_cpu_us_per_query": 10.0,
            "top_kernels": [
                {"name": "kernel", "count": 1, "device_time_us": device_us}
            ],
        },
        "correctness": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    }


def test_postprocess_selects_stop_branch_without_hardware_counters(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.jsonl"
    rows = [
        _row("pytorch_eager", 80.0, 100),
        _row("pytorch_structured", 90.0, 120),
        _row("pytorch_chunked", 70.0, 130),
        _row("tvm_tir_unfused", 100.0, 102),
        _row("tvm_fused_tir", 120.0, 100),
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "hardware_counter_sections_available": False,
                "forbidden_claims": ["achieved occupancy"],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "report"

    assert (
        main(
            [
                "--raw",
                str(raw),
                "--audit",
                str(audit),
                "--out-dir",
                str(out_dir),
                "--no-plots",
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    comparison = summary["comparisons"][0]
    assert summary["selected_pr12l_branch"] == "E_STOP_OPTIMIZING_TIR"
    assert summary["fused_device_time_regressions_vs_unfused"] == 1
    assert comparison["mechanism"] == "LONG_REDUCTION_KERNEL_DEVICE_TIME"
    assert comparison["fused_vs_unfused_device_time_ratio"] == 1.2
