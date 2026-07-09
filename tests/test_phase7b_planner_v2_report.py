import json

from scripts.report_phase7b_planner_v2_candidates import build_planner_v2_report, main


def _cost_model_payload() -> dict:
    return {
        "meta": {
            "schema_version": "phase7b_cost_model_v1",
            "source_git_sha": "abc123",
            "source_device": "cpu",
        },
        "rules": [
            {
                "workload": "permute_reshape_linear",
                "scale_id": "small",
                "recommended_final_concretization_policy": "structured",
                "confidence": "high",
                "relative_gap_to_second_best": 0.12,
                "guardrails": {"unknown_materialization_calls": 0, "split_pos_neg_dense_total": 0},
            },
            {
                "workload": "relu_heavy_mlp",
                "scale_id": "smoke",
                "recommended_final_concretization_policy": "structured",
                "confidence": "low",
                "relative_gap_to_second_best": 0.01,
                "guardrails": {"unknown_materialization_calls": 0, "split_pos_neg_dense_total": 0},
            },
        ],
    }


def test_phase7b_planner_v2_report_marks_embedded_high_confidence_rule_promoted() -> None:
    report = build_planner_v2_report(_cost_model_payload(), device="cpu", min_confidence="high")
    assert report["meta"]["schema_version"] == "phase7b_planner_v2_candidates.v1"
    assert report["summary"]["promoted_count"] == 1
    assert report["summary"]["missing_promotion_count"] == 0
    assert report["summary"]["held_back_count"] == 1
    assert report["promoted_rules"][0]["workload"] == "permute_reshape_linear"
    assert report["held_back_rules"][0]["workload"] == "relu_heavy_mlp"


def test_phase7b_planner_v2_report_cli_schema_smoke(tmp_path, capsys) -> None:
    path = tmp_path / "cost_model.json"
    path.write_text(json.dumps(_cost_model_payload()), encoding="utf-8")

    rc = main([str(path), "--device", "cpu", "--min-confidence", "high"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["meta"]["schema_version"] == "phase7b_planner_v2_candidates.v1"
    assert payload["summary"]["promoted_count"] == 1
