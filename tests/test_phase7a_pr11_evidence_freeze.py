"""Contract tests for the PR-11 evidence freeze manifest."""

from scripts.freeze_phase7a_pr11_evidence import DEFAULT_SOURCES, SCHEMA_VERSION


def test_freeze_contract_names_every_final_evidence_class() -> None:
    assert SCHEMA_VERSION == "boundflow.pr11-evidence-freeze/v1"
    assert any("calibration" in name for name in DEFAULT_SOURCES)
    assert sum("heldout" in name for name in DEFAULT_SOURCES) == 3
    assert sum("final-default" in name for name in DEFAULT_SOURCES) == 3
    assert any("real-oom" in name for name in DEFAULT_SOURCES)
    assert any("regret-attribution" in name for name in DEFAULT_SOURCES)
