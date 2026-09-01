"""Cross-artifact source parity tests for RVIR-v4 V4-2B."""

# pylint: disable=missing-function-docstring

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    production_tensor_sha256,
)
from scripts.verify_rvir_v4_optimizer_step_source_parity import (
    build_report,
    compare_snapshots,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/rvir-v4-production-state/resnet2b-core-capture-v2"
CANDIDATE = ROOT / "artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1"
REPORT = CANDIDATE.parent / "resnet2b-core-step-trace-v1-source-parity.json"


def test_formal_optimizer_step_trace_matches_frozen_source_within_tolerance() -> None:
    report = build_report(BASELINE, CANDIDATE)

    assert report == json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "source-parity-passed"
    assert report["performance_claimed"] is False
    parity = report["parity"]
    assert isinstance(parity, dict)
    assert parity["source_protocol_solver_exact"] is True
    assert parity["call_topology_schema_exact"] is True
    assert parity["result_sign_exact"] is True
    assert parity["result_lower_maximum_absolute_difference"] < 2e-4


def test_source_parity_numeric_gate_rejects_internally_valid_out_of_tolerance() -> None:
    capture = torch.load(
        CANDIDATE / "production_capture.pt", map_location="cpu", weights_only=True
    )
    snapshot = production_snapshot_from_payload_v4(capture["cores"][0]["pre_snapshot"])
    tensor_index = next(
        index
        for index, tensor in enumerate(snapshot.tensors)
        if tensor.semantic_path == "alpha/%2F45/%2F49"
    )
    source = snapshot.tensors[tensor_index]
    value = source.value.clone()
    flat = value.reshape(-1)
    positive_index = int(torch.nonzero(flat > 0.1, as_tuple=False)[0].item())
    flat[positive_index] += 1e-2
    changed_tensors = list(snapshot.tensors)
    changed_tensors[tensor_index] = replace(
        source,
        value=value,
        content_sha256=production_tensor_sha256(value),
    )
    changed = replace(snapshot, tensors=tuple(changed_tensors))
    changed.validate()

    with pytest.raises(ValueError, match="numeric tolerance"):
        compare_snapshots(snapshot, changed)
