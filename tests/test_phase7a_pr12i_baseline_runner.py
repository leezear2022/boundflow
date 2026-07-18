"""Contract smoke for the PR-12I five-baseline runner."""

import json
from pathlib import Path

import pytest
import torch

from scripts.benchmark_phase7a_pr12i_baselines import main
from tests.pr12_split_fixtures import write_pr12_v2_split


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pr12i_runner_emits_region_and_complete_e2e_contracts(tmp_path: Path) -> None:
    out_dir = tmp_path / "baseline"
    split_path = write_pr12_v2_split(tmp_path)

    assert (
        main(
            [
                "--split-file",
                str(split_path),
                "--out-dir",
                str(out_dir),
                "--case-ids",
                "linear-unseen-shape-a",
                "--streams",
                "default",
                "--warmup",
                "1",
                "--groups",
                "1",
                "--repeats",
                "1",
            ]
        )
        == 0
    )

    rows = [
        json.loads(line) for line in (out_dir / "raw.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(rows) == 12
    assert manifest["status_counts"] == {"not_applicable": 3, "ok": 9}
    assert all(row["benchmark_contract"]["compliant"] for row in rows)
    assert {row["benchmark_contract"]["contract_id"] for row in rows} == {
        "pr12-region-runtime-v1",
        "pr12-end-to-end-final-bound-v1",
    }
    structured = [
        row for row in rows if row["candidate"]["backend"] == "pytorch_structured"
    ]
    assert {row["status"] for row in structured} == {"not_applicable", "ok"}
    compiled = [row for row in rows if row["candidate"]["backend"] == "torch_compile"]
    assert {row["status"] for row in compiled} == {"not_applicable"}
    assert any("probe" in row for row in compiled)
    assert all(
        row["status"] == "ok"
        for row in rows
        if row not in structured and row not in compiled
    )
