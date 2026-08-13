"""Capture-ready test for the RVIR-v4 V4-2E formal runner."""

# pylint: disable=missing-function-docstring,protected-access

from pathlib import Path

from scripts import run_rvir_v4_atomic_copy_out_artifact as artifact_runner

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/rvir-v4-native-optimizer/resnet2b-core-step-parity-v1"
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"


def test_formal_atomic_copy_out_reexecutes_and_commits_twelve_paths() -> None:
    capture = artifact_runner._load_torch(SOURCE / "source_capture.pt")

    copy_out, commit, summary = artifact_runner._build_evidence(capture, MODEL)

    assert len(copy_out["path_receipts"]) == 12
    assert commit["committed_path_count"] == 12
    assert commit["atomic_commit"] is True
    assert summary["evaluation_count"] == 10
    assert summary["update_count"] == 9
    assert summary["optimizer_replacement_admitted"] is True
    assert summary["b2_same_solver_timing_admitted"] is False
    assert summary["provider_callback_count"] == 0
    assert summary["fallback_dispatch_count"] == 0
    assert summary["performance_claimed"] is False
