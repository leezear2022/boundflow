"""Capture-ready tests for the RVIR-v4 V4-2C formal runner."""

# pylint: disable=missing-function-docstring,protected-access

from pathlib import Path

from scripts import run_rvir_v4_pre_state_artifact as artifact_runner

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1"
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"


def test_formal_pre_state_source_builds_real_native_scope_and_state() -> None:
    capture = artifact_runner._load_torch(SOURCE / "production_capture.pt")

    mapping, native_state, summary = artifact_runner._build_evidence(capture, MODEL)

    assert mapping["mapping_hash"] == artifact_runner.EXPECTED_MAPPING_HASH
    assert native_state["state_hash"] == summary["native_state_hash"]
    assert summary["relu_state_count"] == 6
    assert summary["round_trip_exact_count"] == 12
    assert summary["alpha_copy_through_receipt_count"] == 6
    assert summary["native_optimizer_update_count"] == 9
    assert summary["native_alpha_learning_rate"] == 0.01
    assert summary["native_beta_learning_rate"] == 0.05
    assert summary["optimizer_mutation_executed"] is False
    assert summary["performance_claimed"] is False
