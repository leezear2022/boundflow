"""R3-1b0 trace artifact summary and frozen-hash tests."""

# pylint: disable=missing-function-docstring,protected-access

import copy
from pathlib import Path

import pytest

from scripts import run_r3_bounded_arena_trace_artifact as artifact

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)


def _trace():  # type: ignore[no-untyped-def]
    if not MODEL.is_file():
        pytest.skip("frozen ResNet2B checkout is unavailable")
    return artifact._trace_payload(CAPTURE, MODEL)


def test_r31b0_artifact_summary_only_opens_compiled_forward_stage() -> None:
    summary = artifact._summary(_trace())
    result = artifact._result(summary)

    assert summary["status"] == "validated-r3-1b0-trace-liveness"
    assert summary["trace_hash"] == artifact.EXPECTED_TRACE_HASH
    assert summary["step_count"] == 12
    assert summary["residual_region_count"] == 2
    assert summary["scratch_slot_count"] == 2
    assert summary["scratch_capacity_bytes_per_slot"] == 73_728
    assert summary["b1_open"] is True
    assert summary["compiled_region"] is False
    assert result["evidence_status"] == summary["status"]
    assert result["performance_claimed"] is False


def test_r31b0_fully_resigned_claim_and_shape_tamper_fail_closed() -> None:
    trace = _trace()
    changed = copy.deepcopy(trace)
    changed["compiled_region"] = True
    semantic = {name: changed[name] for name in changed if name != "trace_hash"}
    changed["trace_hash"] = artifact._canonical_hash(semantic)
    with pytest.raises(ValueError, match="frozen trace"):
        artifact._validate_trace_payload(changed)

    changed = copy.deepcopy(trace)
    changed["steps"][10]["output_shape"] = [6, 1, 3, 31, 32]
    semantic = {name: changed[name] for name in changed if name != "trace_hash"}
    changed["trace_hash"] = artifact._canonical_hash(semantic)
    with pytest.raises(ValueError, match="frozen trace"):
        artifact._validate_trace_payload(changed)
