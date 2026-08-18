"""Formal artifact gates for B4-B0 five-fresh production capture."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

import json
from pathlib import Path
import copy

import pytest
import torch

from scripts import probe_fsg4_b4b_five_fresh_tamper as integrity
from scripts import run_fsg4_b4b_five_fresh_artifact as artifact

ARTIFACT = Path("artifacts/fsg4-b4b-five-fresh/resnet2b-prop0-v1")
TAMPER = ARTIFACT.parent / "resnet2b-prop0-v1-tamper-report.json"
SOURCE_CAPTURE = Path(
    "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path("../vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx")


def test_b4b0_five_fresh_artifact_replays_all_raw_captures() -> None:
    runs, summary, result = artifact._verify_static_artifact(ARTIFACT)

    assert result["status"] == "replay-passed"
    assert len(runs) == summary["run_count"] == 5
    assert summary["capture_count"] == 10
    assert summary["semantic_anchor_count"] == 5
    assert summary["performance_anchor_count"] == 5
    assert summary["all_discrete_structure_exact"] is True
    assert summary["all_numeric_within_tolerance"] is True
    assert summary["all_sign_exact"] is True
    assert summary["root_raw_replay_passed"] is True
    assert summary["maximum_absolute_difference"] <= artifact.ATOL
    assert summary["performance_claimed"] is False
    assert summary["tir_admitted"] is False


def test_b4b0_five_fresh_tamper_report_rejects_all_attacks() -> None:
    report = json.loads(TAMPER.read_text(encoding="utf-8"))

    assert report["attack_count"] == 9
    assert report["rejected_count"] == 9
    assert all(row["outer_resigned"] is True for row in report["rows"])
    assert all(row["rejected"] is True for row in report["rows"])
    assert report["performance_claimed"] is False
    assert report["tir_admitted"] is False


@pytest.mark.parametrize("field", ["topology", "lineage"])
def test_b4b0_replay_rejects_coordinated_all_run_identity_rewrite(
    field: str,
) -> None:
    protocol = artifact._load_json(ARTIFACT / "protocol.json")
    runs = [
        torch.load(ARTIFACT / name, map_location="cpu", weights_only=True)
        for name in artifact.RUN_FILES
    ]
    changed = copy.deepcopy(runs)
    for run in changed:
        for capture in run["captures"]:
            metadata = capture["metadata"]
            if field == "topology":
                metadata["topology_hash"] = "b" * 64
            else:
                lineage = metadata["production_lineage"]
                hashes = lineage["source_tensor_hashes"]
                hashes[sorted(hashes)[0]] = "b" * 64
                lineage_payload = dict(lineage)
                lineage_payload.pop("lineage_hash", None)
                lineage["lineage_hash"] = artifact._canonical_hash(lineage_payload)
            integrity._resign_capture(metadata)
    with pytest.raises(ValueError, match="frozen source identity differs"):
        artifact._summary(changed, protocol)


def test_b4b0_v2_protocol_binds_absolute_source_identity() -> None:
    protocol = artifact._protocol(SOURCE_CAPTURE, MODEL)
    artifact._validate_protocol(protocol)
    assert protocol["schema_version"] == artifact.PROTOCOL_SCHEMA
    assert protocol["frozen_source_identity"] == artifact.FROZEN_SOURCE_IDENTITY


def test_b4b0_v2_protocol_rejects_resigned_frozen_identity_rewrite() -> None:
    protocol = artifact._protocol(SOURCE_CAPTURE, MODEL)
    protocol["frozen_source_identity"]["topology_hash"] = "b" * 64
    payload = dict(protocol)
    payload.pop("protocol_hash")
    protocol["protocol_hash"] = artifact._canonical_hash(payload)
    with pytest.raises(ValueError, match="protocol differs"):
        artifact._validate_protocol(protocol)
