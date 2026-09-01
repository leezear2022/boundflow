"""Frozen MR5 multi-Conv production bridge artifact checks."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
import lzma
from pathlib import Path

from boundflow.runtime.mr5_multi_conv_formal import STATUS
from scripts.run_mr5_multi_conv_formal import replay_artifact

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/measurement-recovery/mr5-multi-conv-production-bridge-v1"


def test_mr5_multi_conv_artifact_replays_from_raw() -> None:
    summary = replay_artifact(ARTIFACT)
    assert summary["status"] == STATUS
    assert summary["pair_count"] == 5
    assert summary["fresh_process_count"] == 10
    assert summary["candidate_forward_count"] == 150
    assert summary["candidate_backward_count"] == 135
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False


def test_mr5_multi_conv_artifact_numeric_closure_is_below_frozen_tolerances() -> None:
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    assert summary["general_maximum_absolute_difference"] <= 2.0e-4
    assert summary["optimizer_maximum_absolute_difference"] <= 2.0e-5
    assert all(metric["sign_exact"] for metric in summary["pair_metrics"])
    assert all(metric["allclose"] for metric in summary["pair_metrics"])


def test_mr5_multi_conv_artifact_rejects_all_fully_resigned_attacks() -> None:
    report = json.loads((ARTIFACT / "tamper_report.json").read_text(encoding="utf-8"))
    assert report["attack_count"] == 21
    assert report["rejected_count"] == 21
    assert all(result["rejected"] for result in report["results"])


def test_mr5_multi_conv_artifact_binds_three_site_specific_modules() -> None:
    raw = json.loads(
        lzma.decompress((ARTIFACT / "raw.json.xz").read_bytes()).decode("utf-8")
    )
    receipts = [
        wrapper["worker"]["bridge_receipt"]
        for wrapper in raw["runs"]
        if wrapper["mode"] == "bridge"
    ]
    assert len(receipts) == 5
    assert all(receipt == receipts[0] for receipt in receipts[1:])
    signatures = receipts[0]["signature_hashes"]
    assert set(signatures) == {"C0", "C1", "C2"}
    assert len(set(signatures.values())) == 3
    assert receipts[0]["cache_miss_count"] == {"C0": 1, "C1": 1, "C2": 1}
    assert receipts[0]["cache_hit_count"] == {"C0": 9, "C1": 9, "C2": 9}
