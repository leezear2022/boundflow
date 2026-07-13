"""Freeze/isolation contracts for the PR-12M split."""

import json
from pathlib import Path

from scripts.start_phase7a_pr12_m import build_split, main


def _parent() -> dict:
    return {
        "split_id": "parent-v2",
        "chunk_rows": 512,
        "calibration": [{"case_id": "old-cal"}],
        "final_heldout": [
            {
                "case_id": "consumed-v2-final",
                "family": "linear",
                "domain": 1,
                "spec": 2,
                "current": 3,
                "previous": 4,
                "budget_mib": 16,
            }
        ],
    }


def test_pr12m_split_promotes_only_consumed_final_and_freezes_policies() -> None:
    split = build_split(_parent())

    assert [row["case_id"] for row in split["calibration"]] == ["consumed-v2-final"]
    assert "old-cal" not in {row["case_id"] for row in split["calibration"]}
    assert len(split["final_heldout"]) == 5
    assert split["budget_mib_sweep"] == [16, 32, 64, 128, None]
    assert {row["policy_id"] for row in split["reuse_policies"]} == {
        "cold_single",
        "mixed_q32",
        "warm_q1024",
    }


def test_pr12m_split_cli_hashes_frozen_output(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(_parent()), encoding="utf-8")
    output = tmp_path / "freeze"

    assert main(["--parent-split", str(parent), "--out-dir", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    split = json.loads((output / "heldout_split.json").read_text())

    assert manifest["split_id"] == split["split_id"]
    assert manifest["calibration_cases"] == 1
    assert manifest["final_heldout_cases"] == 5
