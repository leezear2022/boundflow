"""Freeze-contract tests for the PR-12G multi-backend split."""

import json
from pathlib import Path

from scripts.benchmark_phase7a_pr12_runtime_pareto import _frozen_candidate_backends
from scripts.start_phase7a_pr12_v2 import SCHEMA_VERSION, SPLIT_ID, build_split, main


def _v1() -> dict:
    return json.loads(
        Path("artifacts/phase7a-pr12/baseline/heldout_split.json").read_text()
    )


def test_v2_split_promotes_consumed_v1_without_heldout_overlap() -> None:
    split = build_split(_v1())

    assert split["schema_version"] == SCHEMA_VERSION
    assert split["split_id"] == SPLIT_ID
    assert split["chunk_rows"] == 512
    assert len(split["calibration"]) == 8
    assert len(split["final_heldout"]) == 5
    calibration_ids = {
        record.get("case_id") for record in split["calibration"] if "case_id" in record
    }
    heldout_ids = {record["case_id"] for record in split["final_heldout"]}
    assert calibration_ids.isdisjoint(heldout_ids)
    assert all(case_id.endswith("v2") for case_id in heldout_ids)
    assert _frozen_candidate_backends(split) == (
        "pytorch_eager",
        "pytorch_chunked",
        "tvm_fused_tir",
    )


def test_v2_split_writer_hashes_parent_and_output(tmp_path: Path) -> None:
    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps(_v1()), encoding="utf-8")
    out_dir = tmp_path / "freeze"

    assert main(["--v1-split", str(v1_path), "--out-dir", str(out_dir)]) == 0

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["split_id"] == SPLIT_ID
    assert manifest["calibration_cases"] == 8
    assert manifest["outputs"]["heldout_split.json"]
