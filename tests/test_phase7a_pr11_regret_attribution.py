"""Unit tests for PR-11 regret attribution categories."""

from scripts.attribute_phase7a_pr11_regret import classify_regret


def test_missing_oracle_candidate_is_primary_attribution() -> None:
    primary, flags = classify_regret(
        oracle_pattern="DDS",
        attempted_patterns=("DDD", "SSS"),
        selected_predicted_latency=1.0,
        oracle_predicted_latency=2.0,
        selected_structured_count=3,
        oracle_structured_count=1,
        measurement_variance=True,
        used_fallback=False,
    )

    assert primary == "CANDIDATE_NOT_AVAILABLE"
    assert flags == ("MEASUREMENT_VARIANCE", "BACKEND_GAP")


def test_included_oracle_with_wrong_prediction_is_cost_model_misrank() -> None:
    primary, flags = classify_regret(
        oracle_pattern="DDS",
        attempted_patterns=("DSD", "DDS"),
        selected_predicted_latency=1.0,
        oracle_predicted_latency=2.0,
        selected_structured_count=1,
        oracle_structured_count=1,
        measurement_variance=False,
        used_fallback=False,
    )

    assert primary == "COST_MODEL_MISRANK"
    assert flags == ()
