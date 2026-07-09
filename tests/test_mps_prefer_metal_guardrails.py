from scripts.report_mps_prefer_metal_guardrails import _compare_bounds


def test_compare_bounds_detects_allclose_and_cert_match() -> None:
    default = {
        "bounds": {
            "lower": [[0.1, 0.2], [0.5, 0.1]],
            "upper": [[0.3, 0.4], [0.6, 0.2]],
        },
        "certified_decisions": [-1, -1],
        "certified_count": 0,
    }
    prefer = {
        "bounds": {
            "lower": [[0.100001, 0.2], [0.5, 0.1]],
            "upper": [[0.3, 0.4], [0.6, 0.2]],
        },
        "certified_decisions": [-1, -1],
        "certified_count": 0,
    }

    result = _compare_bounds(default, prefer, atol=1e-4, rtol=1e-4)
    assert result["allclose"] is True
    assert result["cert_decision_match"] is True
    assert result["max_abs_diff"] > 0.0
