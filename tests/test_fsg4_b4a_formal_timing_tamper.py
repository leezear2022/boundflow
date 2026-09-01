"""Tamper-suite contract tests for FSG4/B4-A formal timing."""

# pylint: disable=missing-function-docstring,duplicate-code

from scripts import probe_fsg4_b4a_formal_timing_tamper as tamper


def test_formal_tamper_inventory_is_fixed_and_unique() -> None:
    names = [name for name, _mutate in tamper.ATTACKS]
    assert len(names) == 14
    assert len(set(names)) == 14
    assert "control-latency-outer-resign" in names
    assert "export-payload-outer-resign" in names
    assert "candidate-profile-counter-outer-resign" in names
    assert "worker-protocol-outer-resign" in names
    assert "formal-preflight-outer-resign" in names
    assert "power-policy-outer-resign" in names
    assert "environment-counter-delta-outer-resign" in names
    assert "protocol-sequence-outer-resign" in names
    assert "paired-ratio-outer-resign" in names
    assert "summary-outer-resign" in names
