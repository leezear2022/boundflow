"""Artifact tests for MR1 static same-solver eligibility."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.probe_mr1_static_same_solver_eligibility_tamper import run as run_tamper
from scripts.run_mr1_static_same_solver_eligibility_artifact import (
    OUTPUT,
    generate,
    replay,
)


def test_generate_and_replay_temporary_artifact(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    generated = generate(output)
    replayed = replay(output)
    assert generated == replayed
    assert generated["eligible_target_model_call_count"] == 0


def test_formal_artifact_replays_when_present() -> None:
    if not OUTPUT.exists():
        pytest.skip("formal MR1 artifact not generated yet")
    assert replay(OUTPUT)["eligible_target_model_call_count"] == 0


def test_formal_artifact_tamper_probe_when_present() -> None:
    if not OUTPUT.exists():
        pytest.skip("formal MR1 artifact not generated yet")
    report = run_tamper(OUTPUT)
    assert report["rejected_count"] == report["case_count"] == 13


def test_artifact_contains_no_local_home_path_when_present() -> None:
    if not OUTPUT.exists():
        pytest.skip("formal MR1 artifact not generated yet")
    for path in OUTPUT.rglob("*"):
        if path.is_file():
            assert b"/home/" not in path.read_bytes()
