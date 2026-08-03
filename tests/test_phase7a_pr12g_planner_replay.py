"""Artifact contract for calibration-only PR-12G planner replay."""

import json
from pathlib import Path

from scripts.replay_phase7a_pr12_planner import main


def _row(
    case_id: str,
    backend: str,
    *,
    latency: float,
    peak: int,
    boundary: int,
    eligible: bool = True,
) -> dict:
    return {
        "status": "ok",
        "workload": {
            "case_id": case_id,
            "planner_family": "linear",
            "boundary_bytes": boundary,
            "expected_fused_regions": 1,
        },
        "candidate": {
            "backend": backend,
            "stream": "default",
            "eligible": eligible,
        },
        "runtime": {"host_group_per_query": {"median_ms": latency}},
        "memory": {"peak_allocated_delta_bytes": peak},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_planner_replay_hashes_inputs_and_preserves_fallback(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "split_id": "v2-test",
                "final_heldout": [
                    {
                        "case_id": "held",
                        "family": "linear",
                        "domain": 2,
                        "spec": 8,
                        "current": 16,
                        "previous": 12,
                        "budget_mib": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.jsonl"
    _write_jsonl(
        calibration,
        [
            _row("cal", backend, latency=latency, peak=peak, boundary=2048)
            for backend, latency, peak in (
                ("pytorch_eager", 2.0, 2000),
                ("pytorch_chunked", 1.0, 1000),
                ("tvm_fused_tir", 3.0, 500),
            )
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    rows = [
        _row("held", backend, latency=latency, peak=peak, boundary=2048)
        for backend, latency, peak in (
            ("pytorch_eager", 2.0, 2000),
            ("pytorch_chunked", 1.0, 1000),
            ("tvm_fused_tir", 3.0, 500),
        )
    ]
    rows.extend(
        _row(
            "fanout-graph-fallback-control",
            backend,
            latency=1.0,
            peak=100,
            boundary=11832,
            eligible=backend == "pytorch_eager",
        )
        for backend in ("pytorch_eager", "pytorch_chunked", "tvm_fused_tir")
    )
    _write_jsonl(candidates, rows)
    out_dir = tmp_path / "replay"

    assert (
        main(
            [
                "--split-file",
                str(split),
                "--calibration",
                str(calibration),
                "--candidates",
                str(candidates),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    evaluations = [
        json.loads(line)
        for line in (out_dir / "planner.jsonl").read_text().splitlines()
    ]
    assert evaluations[0]["decision"]["backend"] == "pytorch_chunked"
    assert evaluations[1]["decision"]["backend"] == "pytorch_eager"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["oracle_hits"] == 1
    assert manifest["unsafe_fusion_count"] == 0
    assert manifest["outputs"]["planner.jsonl"]
