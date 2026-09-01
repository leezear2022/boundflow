"""Fail-closed tests for the NRIR49 G0 admission artifact."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.run_nrir49_g0_admission import (
    COMPLETE_QUERY_TARGET,
    EVIDENCE_SCHEMA_VERSION,
    MAX_REQUIRED_REGION_SPEEDUP,
    QUEUE_TARGET,
    absolute_without_resolving_symlink,
    canonical_json,
    generate_artifact,
    projected_scope_speedup,
    replay_artifact,
    required_region_speedup,
    validate_evidence,
)


def _evidence() -> dict[str, object]:
    gate_results = {
        "gpu_infrastructure_ready": False,
        "competitor_environment_ready": False,
        "frontend_observed_matrix_importable": True,
        "shared_non_unknown_workload_present": False,
    }
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "blocked",
        "performance_claimed": False,
        "source": {},
        "gpu": {"status": "blocked_reboot_required", "performance_claimed": False},
        "competitor": {
            "status": "blocked_missing_alpha_beta_crown_source",
            "performance_claimed": False,
        },
        "user_boundconv_40x": {
            "status": "not_auditable_source_missing",
            "performance_claimed": False,
        },
        "frontend": {"status": "validated_reduced", "performance_claimed": False},
        "solveability": {
            "status": "no_go",
            "performance_claimed": False,
            "workload_solver_status": {
                "resnet": {
                    "boundflow_native": "unknown",
                    "external_abcrown": "unknown",
                }
            },
            "shared_non_unknown_workloads": [],
            "qualification_candidate": None,
        },
        "memory_reachability": {
            "status": "not_auditable_gpu_unavailable",
            "performance_claimed": False,
        },
        "amdahl_preregistration": {
            "status": "not_auditable_gpu_unavailable",
            "performance_claimed": False,
            "formula_required": "s / (s + 1 / T - 1)",
            "maximum_required_region_speedup": MAX_REQUIRED_REGION_SPEEDUP,
        },
        "admission": {
            "g1_ready": False,
            "gate_results": gate_results,
            "blockers": [name for name, passed in gate_results.items() if not passed],
        },
        "limitations": [],
    }


def test_amdahl_formula_and_infeasible_boundary() -> None:
    assert required_region_speedup(0.25, QUEUE_TARGET) == pytest.approx(3.0)
    assert required_region_speedup(0.20, QUEUE_TARGET) == pytest.approx(6.0)
    assert required_region_speedup(0.171066, COMPLETE_QUERY_TARGET) == pytest.approx(
        4.211, rel=1e-3
    )
    assert required_region_speedup(0.10, QUEUE_TARGET) is None
    assert projected_scope_speedup(0.25, 3.0) == pytest.approx(QUEUE_TARGET)


def test_unavailable_gpu_cannot_create_opportunity_values() -> None:
    evidence = _evidence()
    validate_evidence(evidence)
    tampered = deepcopy(evidence)
    tampered["amdahl_preregistration"]["status"] = "pending_g1_measurement"
    with pytest.raises(ValueError, match="must not produce measured opportunity"):
        validate_evidence(tampered)


def test_shared_non_unknown_gate_is_derived_from_both_backends() -> None:
    evidence = _evidence()
    evidence["solveability"]["shared_non_unknown_workloads"] = ["resnet"]
    with pytest.raises(ValueError, match="solveability derivation"):
        validate_evidence(evidence)


def test_admission_gate_cannot_be_manually_upgraded() -> None:
    evidence = _evidence()
    evidence["admission"]["g1_ready"] = True
    with pytest.raises(ValueError, match="blocker derivation"):
        validate_evidence(evidence)


def test_artifact_replay_rejects_digest_tamper(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "g0"
    evidence = _evidence()
    generate_artifact(artifact_dir, evidence)
    assert replay_artifact(artifact_dir) == evidence
    path = artifact_dir / "admission.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        replay_artifact(artifact_dir)


def test_performance_claim_is_always_rejected() -> None:
    evidence = _evidence()
    evidence["performance_claimed"] = True
    with pytest.raises(ValueError, match="header differs"):
        validate_evidence(evidence)


def test_canonical_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        canonical_json({"bad": float("inf")})


def test_competitor_python_path_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "python-real"
    target.write_text("", encoding="utf-8")
    link = tmp_path / "venv-python"
    link.symlink_to(target)
    assert absolute_without_resolving_symlink(link) == link
    assert absolute_without_resolving_symlink(link) != link.resolve()


def test_qualification_candidate_cannot_disagree_between_backends() -> None:
    evidence = _evidence()
    candidate = {
        "performance_claimed": False,
        "selection_role": "solveability_qualification_only_not_performance_tuning",
        "workload_id": "public:1",
        "native_result": {
            "workload_id": "public:1",
            "solver_status": "verified",
            "performance_claimed": False,
        },
        "abcrown_result": {
            "workload_id": "public:1",
            "solver_status": "unsafe",
            "performance_claimed": False,
        },
    }
    evidence["solveability"]["qualification_candidate"] = candidate
    with pytest.raises(ValueError, match="qualification candidate contract"):
        validate_evidence(evidence)
