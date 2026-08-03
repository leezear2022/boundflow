"""Artifact contract for PR-12M compile-aware held-out replay."""

import json
from pathlib import Path

from scripts.replay_phase7a_pr12m_compile_aware import main


def _row(case_id: str, backend: str, warm: float, first: float, peak: int) -> dict:
    return {
        "status": "ok",
        "benchmark_contract": {"level": "end_to_end_final_bound"},
        "workload": {"case_id": case_id},
        "candidate": {"backend": backend, "stream": "default"},
        "runtime": {
            "compile_first_run_wall_ms": first,
            "host_group_per_query": {"median_ms": warm},
        },
        "memory": {"peak_allocated_delta_bytes": peak},
        "correctness": {
            "allclose": True,
            "finite": True,
            "lower_le_upper": True,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compile_aware_replay_scans_budget_and_reuse_without_leakage(
    tmp_path: Path,
) -> None:
    record = {
        "family": "linear",
        "domain": 1,
        "spec": 4,
        "current": 8,
        "previous": 6,
        "budget_mib": 16,
    }
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "split_id": "compile-aware-test",
                "calibration": [{"case_id": "cal", **record}],
                "final_heldout": [{"case_id": "held", **record}],
                "budget_mib_sweep": [1, None],
                "reuse_policies": [
                    {
                        "policy_id": "cold",
                        "expected_reuse_queries": 1,
                        "memory_cache_hit_probability": 0.0,
                        "disk_cache_hit_probability": 0.0,
                    },
                    {
                        "policy_id": "warm",
                        "expected_reuse_queries": 1024,
                        "memory_cache_hit_probability": 1.0,
                        "disk_cache_hit_probability": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(
        calibration,
        [
            _row("cal", "pytorch_eager", 2.0, 2.0, 2000),
            _row("cal", "tvm_fused_tir", 1.0, 101.0, 1000),
        ],
    )
    _write_jsonl(
        heldout,
        [
            _row("held", "pytorch_eager", 2.0, 2.0, 2000),
            _row("held", "tvm_fused_tir", 1.0, 101.0, 1000),
        ],
    )
    amortization = tmp_path / "amortization.jsonl"
    _write_jsonl(
        amortization,
        [
            {
                "status": "ok",
                "workload": {"family": "linear"},
                "amortization": {"disk_first_query_setup_ms": 10.0},
            }
        ],
    )
    output = tmp_path / "replay"

    assert (
        main(
            [
                "--split-file",
                str(split),
                "--calibration",
                str(calibration),
                "--heldout",
                str(heldout),
                "--amortization",
                str(amortization),
                "--out-dir",
                str(output),
            ]
        )
        == 0
    )
    rows = [
        json.loads(line) for line in (output / "planner.jsonl").read_text().splitlines()
    ]
    summary = json.loads((output / "summary.json").read_text())

    assert len(rows) == 4
    assert {row["decision"]["backend"] for row in rows} == {
        "pytorch_eager",
        "tvm_fused_tir",
    }
    assert summary["unsafe_backend_count"] == 0
    assert (
        summary["selected_budget_feasible"] == summary["budget_feasible_opportunities"]
    )
