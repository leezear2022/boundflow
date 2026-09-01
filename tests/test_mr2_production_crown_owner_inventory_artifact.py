"""Artifact tests for the MR2 production CROWN owner inventory."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.probe_mr2_production_crown_owner_inventory_tamper import run as run_tamper
from scripts.run_mr2_production_crown_owner_inventory_artifact import (
    OUTPUT,
    generate,
    replay,
)


def test_generate_and_replay_temporary_artifact(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    generated = generate(output)
    assert replay(output) == generated
    assert generated["selected_site"] == "P:25/Conv_8"
    assert (output / "replay_stdout.txt").is_file()


def test_formal_artifact_replays_when_present() -> None:
    if not OUTPUT.exists():
        pytest.skip("formal MR2 artifact not generated yet")
    assert replay(OUTPUT)["selected_site"] == "P:25/Conv_8"


def test_formal_artifact_tamper_probe_when_present() -> None:
    if not OUTPUT.exists():
        pytest.skip("formal MR2 artifact not generated yet")
    report = run_tamper(OUTPUT)
    assert report["rejected_count"] == report["case_count"] == 12


def test_artifact_contains_no_local_home_path_when_present() -> None:
    if not OUTPUT.exists():
        pytest.skip("formal MR2 artifact not generated yet")
    for path in OUTPUT.rglob("*"):
        if path.is_file():
            assert b"/home/" not in path.read_bytes()
