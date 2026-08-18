"""Candidate runner gates for the B4-B1a five-fresh artifact."""

# pylint: disable=missing-function-docstring,protected-access

from pathlib import Path

from scripts import probe_fsg4_b4b1_reference_capture_integrity as integrity
from scripts import run_fsg4_b4b1_reference_five_fresh_artifact as artifact

SOURCE_CAPTURE = Path(
    "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path("../vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx")


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
