import json

from scripts.bench_phase7b_crossover_matrix import main


def test_phase7b_crossover_matrix_schema_smoke(capsys) -> None:
    rc = main(
        [
            "--device",
            "cpu",
            "--workloads",
            "permute_reshape_linear",
            "--scales",
            "smoke",
            "--policies",
            "structured,auto",
            "--warmup",
            "1",
            "--iters",
            "1",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["meta"]["schema_version"] == "phase7b_crossover_matrix.v1"
    assert payload["meta"]["scales"] == ["smoke"]
    assert payload["meta"]["policies"] == ["structured", "auto"]
    assert "device_meta" in payload["meta"]
    assert payload["meta"]["capability_table"]["schema_version"] == 1
    assert len(payload["rows"]) == 2
    assert len(payload["summary"]) == 1

    by_policy = {row["policy_request"]: row for row in payload["rows"]}
    assert by_policy["structured"]["planner_decision"]["final_concretization_policy"] == "structured"
    assert by_policy["auto"]["planner_decision"]["final_concretization_policy"] == "dense_barrier"

    for row in payload["rows"]:
        metrics = row["metrics"]
        assert metrics["unknown_materialization_calls"] == 0
        assert metrics["split_pos_neg_dense_total"] == 0
        assert "structured_ms_p50" in metrics
        assert "materialized_bytes" in metrics

    summary = payload["summary"][0]
    assert summary["workload"] == "permute_reshape_linear"
    assert summary["scale_id"] == "smoke"
    assert summary["auto_final_concretization_policy"] == "dense_barrier"
