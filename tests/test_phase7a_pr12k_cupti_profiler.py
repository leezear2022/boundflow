"""Smoke contract for the PR-12K CUPTI activity fallback profiler."""

import json
from pathlib import Path

import pytest
import torch

from scripts.profile_phase7a_pr12k_cupti import main
from tests.pr12_split_fixtures import write_pr12_v2_split


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pr12k_profiler_emits_kernel_activity_and_audit(tmp_path: Path) -> None:
    out_dir = tmp_path / "profile"
    split_path = write_pr12_v2_split(tmp_path)

    assert (
        main(
            [
                "--split-file",
                str(split_path),
                "--out-dir",
                str(out_dir),
                "--case-ids",
                "cal-linear-d2-s8-i16-j12",
                "--backends",
                "pytorch_eager,tvm_fused_tir",
                "--warmup",
                "1",
                "--iterations",
                "1",
                "--skip-counter-probe",
            ]
        )
        == 0
    )

    rows = [
        json.loads(line)
        for line in (out_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    audit = json.loads((out_dir / "profiler_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status_counts"] == {"ok": 2}
    assert not audit["hardware_counter_sections_available"]
    assert audit["cupti_library"]
    assert all(row["correctness"]["allclose"] for row in rows)
    assert all(row["activity"]["kernel_launches"] > 0 for row in rows)
    assert all((out_dir / "traces" / row["trace"]).is_file() for row in rows)
