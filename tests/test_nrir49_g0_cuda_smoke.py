"""Fail-closed contracts for the post-reboot NRIR49 G0 CUDA smoke."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.run_nrir49_g0_cuda_smoke import (
    EVIDENCE_SCHEMA_VERSION,
    GATE_ORDER,
    JSON_MARKER,
    aggregate_gates,
    array_sha256,
    derive_cross_environment_gate,
    expected_vector_contract,
    generate_artifact,
    parse_marked_json,
    python_type_identity,
    replay_artifact,
    validate_evidence,
)


def _gate(status: str = "pass") -> dict[str, object]:
    return {
        "status": status,
        "performance_claimed": False,
        "facts": {},
        "error": None if status == "pass" else "unavailable",
    }


def _gate_with_facts(facts: dict[str, object]) -> dict[str, object]:
    value = _gate()
    value["facts"] = facts
    return value


def _evidence() -> dict[str, object]:
    gates = {name: _gate() for name in GATE_ORDER}
    admission = aggregate_gates(gates)
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": admission["status"],
        "performance_claimed": False,
        "source": {},
        "gates": gates,
        "admission": admission,
        "limitations": ["one", "two", "three"],
    }


def test_exact_six_pass_matrix_is_required_for_g1() -> None:
    evidence = _evidence()
    validate_evidence(evidence)
    assert evidence["admission"] == {
        "g0_cuda_ready": True,
        "blockers": [],
        "status": "ready_for_g1",
    }
    evidence["gates"]["tvm_ffi_custom_stream"] = _gate("blocked")
    evidence["admission"] = aggregate_gates(evidence["gates"])
    evidence["status"] = "blocked"
    validate_evidence(evidence)
    assert evidence["admission"]["blockers"] == ["tvm_ffi_custom_stream"]


def test_gate_set_and_manual_readiness_upgrade_are_rejected() -> None:
    evidence = _evidence()
    missing = dict(evidence["gates"])
    missing.pop("tvm_ffi_custom_stream")
    with pytest.raises(ValueError, match="gate set differs"):
        aggregate_gates(missing)
    evidence["gates"]["competitor_torch_cuda"] = _gate("fail")
    with pytest.raises(ValueError, match="admission derivation differs"):
        validate_evidence(evidence)


def test_performance_claim_is_rejected_at_header_and_gate() -> None:
    evidence = _evidence()
    evidence["performance_claimed"] = True
    with pytest.raises(ValueError, match="header differs"):
        validate_evidence(evidence)
    evidence = _evidence()
    evidence["gates"]["boundflow_torch_cuda"]["performance_claimed"] = True
    with pytest.raises(ValueError, match="attempted a performance claim"):
        validate_evidence(evidence)


def test_marked_json_parser_ignores_noisy_import_output() -> None:
    actual = parse_marked_json(
        "library log\n" + JSON_MARKER + '{"cuda_available":true,"device_count":1}\n'
    )
    assert actual == {"cuda_available": True, "device_count": 1}
    with pytest.raises(ValueError, match="marker is missing"):
        parse_marked_json("only logs")


def test_vector_contract_is_stable() -> None:
    assert expected_vector_contract() == {
        "dtype": "float32",
        "shape": [256],
        "input_sha256": "04441b72253f49384e853fb46a81657e5e28187f02187a47713eb9cd482f9a17",
        "output_sha256": "bdc0fa4648c549a709922fce396b871234a0e0707a14a918e9542b65835e2e2b",
    }


def test_python_type_identity_does_not_probe_dynamic_attributes() -> None:
    class DynamicModule:
        """Test double whose dynamic lookup must remain untouched."""

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"unexpected dynamic lookup: {name}")

    assert python_type_identity(DynamicModule()).endswith(".DynamicModule")


def test_cross_environment_gate_requires_device_and_all_digests(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    property_path = tmp_path / "property.vnnlib"
    model.write_bytes(b"model")
    property_path.write_bytes(b"property")
    oracle = expected_vector_contract()
    device = {
        "device_name": "GPU",
        "capability": [8, 9],
        "total_memory_bytes": 8 << 30,
        "input_sha256": oracle["input_sha256"],
        "output_sha256": oracle["output_sha256"],
    }
    competitor = {
        **device,
        "model_sha256": hashlib.sha256(b"model").hexdigest(),
        "property_sha256": hashlib.sha256(b"property").hexdigest(),
    }
    tvm_expected = np.arange(256, dtype="float32") + np.float32(1.0)
    tvm = {
        "input_sha256": oracle["input_sha256"],
        "output_sha256": array_sha256(tvm_expected),
        "expected_output_sha256": array_sha256(tvm_expected),
    }
    actual = derive_cross_environment_gate(
        boundflow=_gate_with_facts(device),
        tvm=_gate_with_facts(tvm),
        competitor=_gate_with_facts(competitor),
        model=model,
        property_path=property_path,
    )
    assert actual["status"] == "pass"
    competitor["capability"] = [9, 0]
    failed = derive_cross_environment_gate(
        boundflow=_gate_with_facts(device),
        tvm=_gate_with_facts(tvm),
        competitor=_gate_with_facts(competitor),
        model=model,
        property_path=property_path,
    )
    assert failed["status"] == "fail"
    assert failed["facts"]["checks"]["capability_match"] is False


def test_artifact_replay_rejects_file_and_semantic_tamper(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "cuda-smoke"
    evidence = _evidence()
    generate_artifact(artifact_dir, evidence)
    assert replay_artifact(artifact_dir) == evidence
    path = artifact_dir / "cuda_smoke.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        replay_artifact(artifact_dir)


def test_semantic_tamper_is_rejected_even_with_rewritten_manifest(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    tampered = deepcopy(evidence)
    tampered["gates"]["nvidia_driver_device"] = _gate("fail")
    artifact_dir = tmp_path / "tampered"
    with pytest.raises(ValueError, match="admission derivation differs"):
        validate_evidence(tampered)
    generate_artifact(artifact_dir, tampered)
    with pytest.raises(ValueError, match="admission derivation differs"):
        replay_artifact(artifact_dir)
