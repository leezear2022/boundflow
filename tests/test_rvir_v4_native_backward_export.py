"""Contracts for RVIR-v4 V4-3B native backward export."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from dataclasses import replace
import json
from pathlib import Path

import pytest

from boundflow.runtime.rvir_v4_native_backward_export import (
    NativeBackwardExportV4,
)
from scripts import run_rvir_v4_native_backward_export_artifact as artifact_runner

ATOMIC_CAPTURE = Path(
    "artifacts/rvir-v4-atomic-copy-out/resnet2b-core-copy-out-v1/source_capture.pt"
)
WHOLE_TRUTH = Path("artifacts/rvir-v4-whole-core-truth/resnet2b-core-v1/truth.pt")
MODEL = Path("../vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx")
ARTIFACT = Path("artifacts/rvir-v4-native-backward-export/resnet2b-core-v1")
TAMPER_REPORT = ARTIFACT.parent / "resnet2b-core-v1-tamper-report.json"


@pytest.fixture(scope="module")
def formal_export() -> tuple[dict[str, object], dict[str, object]]:
    return artifact_runner._build_evidence(
        artifact_runner._load_torch(ATOMIC_CAPTURE),
        artifact_runner._load_torch(WHOLE_TRUTH),
        MODEL,
    )


def test_native_backward_exports_all_six_l_as_and_intermediates(
    formal_export: tuple[dict[str, object], dict[str, object]],
) -> None:
    export, summary = formal_export

    assert set(export["lAs"]) == {  # type: ignore[arg-type]
        "/input-4",
        "/input-12",
        "/input-16",
        "/input-24",
        "/45",
        "/48",
    }
    assert len(export["intermediates"]) == 6  # type: ignore[arg-type]
    assert summary["l_a_maximum_absolute_difference"] < 1e-6
    assert summary["intermediate_maximum_absolute_difference"] < 1e-5
    assert summary["lower_maximum_absolute_difference"] < 1e-5
    assert summary["native_backward_export_admitted"] is True
    assert summary["whole_core_replacement_admitted"] is False
    assert summary["b2_same_solver_timing_admitted"] is False
    assert summary["performance_claimed"] is False


def test_native_backward_export_rejects_provider_callback(
    formal_export: tuple[dict[str, object], dict[str, object]],
) -> None:
    export, _summary = formal_export
    typed = artifact_runner._export_from_payload(export)
    assert isinstance(typed, NativeBackwardExportV4)

    with pytest.raises(ValueError, match="export contract differs"):
        replace(typed, provider_compute_bounds_callback_count=1).validate()


def test_formal_artifact_replays_and_rejects_all_tamper_probes() -> None:
    result = artifact_runner._replay(ARTIFACT, MODEL)
    report = json.loads(TAMPER_REPORT.read_text(encoding="utf-8"))

    assert result["status"] == "replay-passed"
    assert report["attack_count"] == 5
    assert report["fully_resigned_export_attack_count"] == 3
    assert report["all_rejected"] is True
    assert report["performance_claimed"] is False
