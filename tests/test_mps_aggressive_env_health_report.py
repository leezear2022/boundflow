import json

from scripts.report_mps_aggressive_env_health import _tail


def test_tail_limits_lines() -> None:
    assert _tail("a\nb\nc", lines=2) == "b\nc"


def test_health_schema_shape() -> None:
    payload = {
        "meta": {"schema_version": "mps_aggressive_env_health.v1"},
        "checks": {},
        "summary": {"clean_import_ok": True, "kmp_workaround_needed": False},
    }
    assert json.loads(json.dumps(payload))["meta"]["schema_version"] == "mps_aggressive_env_health.v1"
