"""Candidate runner gates for the B4-B1a five-fresh artifact."""

# pylint: disable=missing-function-docstring,protected-access

from pathlib import Path

from scripts import probe_fsg4_b4b1_reference_capture_integrity as integrity
from scripts import run_fsg4_b4b1_reference_five_fresh_artifact as artifact

SOURCE_CAPTURE = Path(
    "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path("../vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx")
ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")
INTEGRITY_REPORT = ARTIFACT.parent / "resnet2b-prop0-v1-integrity-report.json"


def test_b4b1a_protocol_binds_b4b0_source_and_new_code_revision() -> None:
    protocol = artifact._protocol(SOURCE_CAPTURE.resolve(), MODEL.resolve())
    artifact._validate_protocol(protocol)
    assert protocol["schema_version"] == artifact.PROTOCOL_SCHEMA
    assert protocol["base_protocol_hash"] == protocol["base_protocol"]["protocol_hash"]
    assert set(protocol["code_revision"]) == set(artifact.CODE_PATHS)
    assert protocol["performance_claimed"] is False
    assert protocol["tir_admitted"] is False


def test_b4b1a_runner_and_integrity_inventory_is_frozen() -> None:
    assert artifact.RUN_COUNT == 5
    assert len(artifact.RUN_FILES) == 5
    assert len(integrity.CASES) == 8
    assert [name for name, _mutation in integrity.CASES] == [
        "incoming-lower-bias-outer-resigned",
        "operator-bias-value-outer-resigned",
        "operator-bias-presence-outer-resigned",
        "output-lower-a-gradient-outer-resigned",
        "output-bias-gradient-outer-resigned",
        "sparse-mapping-index-outer-resigned",
        "reference-attribute-outer-resigned",
        "base-topology-outer-resigned",
    ]
    assert all(
        (artifact.REPOSITORY_ROOT / path).is_file() for path in artifact.CODE_PATHS
    )


def test_b4b1a_formal_artifact_replays_all_amendment_raw() -> None:
    runs, summary, result = artifact._verify_static_artifact(ARTIFACT)
    assert len(runs) == summary["run_count"] == result["run_count"] == 5
    assert summary["capture_count"] == result["capture_count"] == 10
    assert summary["amendment_tensor_comparison_count"] == 90
    assert summary["amendment_element_comparison_count"] == 63645
    assert summary["maximum_amendment_absolute_difference"] == 0.0
    assert summary["all_amendment_sign_exact"] is True
    assert summary["bias_and_output_adjoint_present"] is True
    assert summary["sparse_mapping_raw_present"] is True
    assert summary["operator_bias_present"] == [True, True]
    assert result["status"] == "replay-passed"
    assert result["performance_claimed"] is False
    assert result["tir_admitted"] is False


def test_b4b1a_formal_integrity_report_rejects_all_registered_cases() -> None:
    report = artifact._load_json(INTEGRITY_REPORT)
    assert report["case_count"] == report["rejected_count"] == 8
    assert all(row["outer_resigned"] is True for row in report["rows"])
    assert all(row["rejected"] is True for row in report["rows"])
    assert "coordinated dynamic bias/adjoint rewrites" in report["known_limit"]
    assert report["performance_claimed"] is False
    assert report["tir_admitted"] is False
