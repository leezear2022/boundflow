import json

from scripts.postprocess_phase7b_cost_model import build_cost_model, main


def _row(workload: str, scale_id: str, policy: str, final_policy: str, ms: float) -> dict:
    return {
        "workload": workload,
        "scale_id": scale_id,
        "policy_request": policy,
        "compare_target": "layout_only" if workload == "permute_reshape_linear" else "relu_barrier",
        "metrics": {
            "structured_ms_p50": ms,
            "baseline_ms_p50": ms * 0.8,
            "speedup": 0.8,
            "materialized_bytes": 1024,
            "materialized_numel": 256,
            "unknown_materialization_calls": 0,
            "right_matmul_exact_bytes": 512 if workload != "permute_reshape_linear" else 0,
            "final_bound_concretization_bytes": 1024 if final_policy == "structured" else 0,
            "final_bound_dense_barrier_bytes": 1024 if final_policy == "dense_barrier" else 0,
            "cache_hits": 1,
            "cache_misses": 2,
            "split_pos_neg_dense_total": 0,
            "planner_final_concretization_policy": final_policy,
            "planner_reason": "test",
        },
    }


def _matrix_payload() -> dict:
    return {
        "meta": {
            "schema_version": "phase7b_crossover_matrix.v1",
            "git_sha": "abc123",
            "device": "cpu",
            "dtype": "float32",
            "timer": "perf_counter",
            "warmup": 1,
            "iters": 3,
            "capability_table": {"schema_version": 1, "operators": {}},
        },
        "rows": [
            _row("permute_reshape_linear", "smoke", "structured", "structured", 1.0),
            _row("permute_reshape_linear", "smoke", "dense_barrier", "dense_barrier", 0.7),
            _row("permute_reshape_linear", "smoke", "auto", "dense_barrier", 0.72),
            _row("relu_heavy_mlp", "smoke", "structured", "structured", 1.0),
            _row("relu_heavy_mlp", "smoke", "dense_barrier", "dense_barrier", 1.02),
            _row("relu_heavy_mlp", "smoke", "auto", "structured", 1.01),
        ],
    }


def test_phase7b_cost_model_builds_rules_from_matrix_payload() -> None:
    model = build_cost_model(_matrix_payload(), min_relative_margin=0.05)
    assert model["meta"]["schema_version"] == "phase7b_cost_model_v1"
    assert model["meta"]["source_schema_version"] == "phase7b_crossover_matrix.v1"
    assert len(model["rules"]) == 2

    by_workload = {rule["workload"]: rule for rule in model["rules"]}
    layout = by_workload["permute_reshape_linear"]
    assert layout["recommended_policy_request"] == "dense_barrier"
    assert layout["recommended_final_concretization_policy"] == "dense_barrier"
    assert layout["confidence"] == "high"
    assert layout["guardrails"]["unknown_materialization_calls"] == 0
    assert layout["guardrails"]["split_pos_neg_dense_total"] == 0

    relu = by_workload["relu_heavy_mlp"]
    assert relu["recommended_policy_request"] == "structured"
    assert relu["recommended_final_concretization_policy"] == "structured"
    assert relu["confidence"] == "low"

    summary = {item["workload"]: item for item in model["summary"]}
    assert summary["permute_reshape_linear"]["recommended_default_final_concretization_policy"] == "dense_barrier"
    assert summary["relu_heavy_mlp"]["recommended_default_final_concretization_policy"] == "structured"


def test_phase7b_cost_model_cli_schema_smoke(tmp_path, capsys) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(_matrix_payload()), encoding="utf-8")

    rc = main([str(matrix_path), "--min-relative-margin", "0.05"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["meta"]["schema_version"] == "phase7b_cost_model_v1"
    assert len(payload["rules"]) == 2
