"""Contract test for calibration-to-heldout barrier placement evaluation."""

import csv
import json

from scripts.evaluate_phase7a_pr11_barrier_placements import (
    EVAL_SCHEMA_VERSION,
    POLICIES,
    main,
)


def _row(pattern: str, *, peak: int, latency: float, workload: str) -> dict:
    persistent = pattern.count("D") * 200_000
    ephemeral = pattern.count("S") * 200_000
    return {
        "schema_version": "boundflow.pr11-barrier-placement-profile/v3",
        "status": "ok",
        "query_id": f"{workload}:{pattern}",
        "workload": {"name": workload, "tier": "test"},
        "method": "CROWN",
        "spec_size": 1,
        "domain_batch_size": 1,
        "barrier_ids": ["h1", "h2"],
        "static_barriers": [
            {
                "schema_version": "boundflow.materialization_static_barrier/v2",
                "barrier_id": name,
                "relu_output_id": f"{name}_relu",
                "producer_op_type": "linear" if index == 0 else "add",
                "topo_index": index * 2 + 1,
                "value_shape_per_domain": [4],
                "value_numel_per_domain": 4,
                "spec_batch_size": 1,
                "domain_batch_size": 1,
                "element_size_bytes": 50_000,
                "coefficient_elements": 4,
                "coefficient_bytes": 200_000,
                "estimated_dense_flops": 4,
                "reuse_count": 2 if index == 0 else 1,
                "direct_consumer_count": 2 if index == 0 else 1,
                "direct_live_span": 2 if index == 0 else 1,
                "downstream_depth": 3 - index,
                "downstream_merge_count": 1 if index == 0 else 0,
                "downstream_branch_count": 0,
                "downstream_path_count": 2 if index == 0 else 1,
                "is_merge_output": index == 1,
                "is_branch_source": index == 0,
            }
            for index, name in enumerate(("h1", "h2"))
        ],
        "placement": {
            "placements": [
                {
                    "barrier_id": name,
                    "action": "dense" if action == "D" else "structured",
                }
                for name, action in zip(("h1", "h2"), pattern)
            ]
        },
        "timing_trace_off": {
            "peak_cuda_allocated_bytes": peak,
            "peak_cuda_reserved_bytes": peak,
            "latency_ms_median": latency,
            "latency_ms_p90": latency,
        },
        "trace_on": {
            "materialization": {
                "by_lifetime_class": {
                    "persistent": {"logical_bytes": persistent},
                    "ephemeral": {"logical_bytes": ephemeral},
                }
            }
        },
    }


def _write(path, *, workload: str, scale: float) -> None:
    rows = [
        _row("DD", peak=int(1_500_000 * scale), latency=1.0 * scale, workload=workload),
        _row("DS", peak=int(1_000_000 * scale), latency=2.0 * scale, workload=workload),
        _row("SD", peak=int(1_100_000 * scale), latency=2.2 * scale, workload=workload),
        _row("SS", peak=int(800_000 * scale), latency=4.0 * scale, workload=workload),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_barrier_evaluator_keeps_calibration_and_heldout_artifacts_separate(
    tmp_path,
) -> None:
    calibration = tmp_path / "calibration.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    out_dir = tmp_path / "eval"
    _write(calibration, workload="calibration_net", scale=1.0)
    _write(heldout, workload="heldout_net", scale=1.1)

    result = main(
        [
            "--calibration",
            str(calibration),
            "--heldout",
            str(heldout),
            "--out-dir",
            str(out_dir),
            "--budgets-mib",
            "1,2",
        ]
    )

    assert result == 0
    rows = [
        json.loads(line) for line in (out_dir / "raw.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(POLICIES) * 2
    assert {row["schema_version"] for row in rows} == {EVAL_SCHEMA_VERSION}
    assert {row["workload"]["name"] for row in rows} == {"heldout_net"}
    assert all(
        row["split"]["heldout_profile"] not in row["split"]["calibration_profiles"]
        for row in rows
    )
    bounded = [row for row in rows if row["policy"] == "global_bounded_retry"]
    assert bounded
    assert all(row["retry"]["bounded"] for row in bounded)
    assert all(row["retry"]["attempt_count"] <= 6 for row in bounded)
    with (out_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == len(POLICIES)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["row_count"] == len(rows)
    assert manifest["retry_max_attempts"] == 6
    assert manifest["retry_strategy"] == "latency_topology_density_stratified_v3"
    assert manifest["retry_prediction_budget_factor"] == 1.3
    assert manifest["ridge"] == 0.001
    assert manifest["feature_source"] == "static_topology_liveness_v1"
    assert manifest["outputs"]["raw.jsonl"]
    assert manifest["outputs"]["cost_model.json"]
    assert manifest["outputs"]["summary.csv"]
