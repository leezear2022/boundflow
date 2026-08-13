"""Capture-ready tests for the RVIR-v4 V4-2D formal runner."""

# pylint: disable=missing-function-docstring,protected-access

from pathlib import Path

from scripts import run_rvir_v4_native_optimizer_artifact as artifact_runner

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1"
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"


def test_formal_native_optimizer_builds_ten_independent_parity_steps() -> None:
    capture = artifact_runner._load_torch(SOURCE / "source_capture.pt")

    native, parity, summary = artifact_runner._build_evidence(capture, MODEL)

    assert native["evaluation_count"] == 10
    assert native["update_count"] == 9
    assert native["provider_callback_count"] == 0
    assert isinstance(parity["step_rows"], list)
    assert len(parity["step_rows"]) == 10
    assert summary["all_steps_allclose"] is True
    assert summary["all_steps_sign_exact"] is True
    for name in (
        "lower_maximum_absolute_difference",
        "alpha_maximum_absolute_difference",
        "beta_maximum_absolute_difference",
    ):
        value = summary[name]
        assert isinstance(value, float)
        assert value <= 2e-4
    assert summary["atomic_copy_out_executed"] is False
    assert summary["optimizer_replacement_admitted"] is False
    assert summary["performance_claimed"] is False
