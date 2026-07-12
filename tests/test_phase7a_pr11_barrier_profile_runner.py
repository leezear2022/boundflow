"""Smoke test for measured PR-11 barrier placement profiling."""

import json

from scripts.profile_phase7a_pr11_barrier_placements import (
    PROFILE_SCHEMA_VERSION,
    _make_placement_workload,
    main,
)

import torch


def test_branched_resnet_is_an_independent_non_toy_profile_workload() -> None:
    workload = _make_placement_workload("branched_resnet", torch.device("cpu"))

    output = workload.model(torch.zeros((1, *workload.input_shape)))

    assert workload.tier == "non_toy_heldout_topology"
    assert output.shape == (1, 10)


def test_barrier_profile_runner_enumerates_all_two_relu_combinations(tmp_path) -> None:
    run_id = "test-barrier-profile"
    result = main(
        [
            "--run-id",
            run_id,
            "--out-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--workloads",
            "mlp_chain",
            "--spec-sizes",
            "1",
            "--domain-batches",
            "1",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
    )

    assert result == 0
    out_dir = tmp_path / run_id
    rows = [
        json.loads(line) for line in (out_dir / "raw.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert {row["schema_version"] for row in rows} == {PROFILE_SCHEMA_VERSION}
    assert {tuple(row["barrier_ids"]) for row in rows} == {("linear", "linear_1")}
    assert all(
        [item["barrier_id"] for item in row["static_barriers"]] == row["barrier_ids"]
        for row in rows
    )
    assert all(
        all(item["coefficient_bytes"] > 0 for item in row["static_barriers"])
        for row in rows
    )
    assert {
        tuple(item["action"] for item in row["placement"]["placements"]) for row in rows
    } == {
        ("dense", "dense"),
        ("dense", "structured"),
        ("structured", "dense"),
        ("structured", "structured"),
    }
    assert all(row["correctness"]["allclose_dense"] for row in rows)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["row_count"] == 4
    assert manifest["status_counts"] == {"ok": 4}
