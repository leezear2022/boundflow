"""Frozen NRIR-27 production prepared verifier artifact contracts."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_production_prepared_verifier_artifact import (
    EVIDENCE_FILE,
    MANIFEST_FILE,
    canonical_hash,
    file_sha256,
    validate_evidence_structure,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "production-prepared-verifier"
    / "vnncomp21-three-topology-cpu-v1"
)
EVIDENCE_HASH = "7b650dce529d47c54eeadb168b2311e83a4346b47ffc341d5293b6468c6ac08b"


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_production_artifact_digest_ir_and_repeated_coverage() -> None:
    manifest = _load(MANIFEST_FILE)
    evidence = _load(EVIDENCE_FILE)
    validate_evidence_structure(evidence)

    assert manifest["evidence_hash"] == EVIDENCE_HASH == canonical_hash(evidence)
    assert manifest["performance_claimed"] is True
    assert manifest["files"] == {
        str(path.relative_to(ARTIFACT_DIR)): file_sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    ir = evidence["ir"]
    assert isinstance(ir, dict)
    assert len(ir["task"]["tasks"]) == 21
    assert len(ir["schedule"]["fresh_process_task_ids"]) == 6
    records = evidence["records"]
    assert isinstance(records, list) and len(records) == 27


def test_production_artifact_freezes_internal_speedups_and_full_query_status() -> None:
    evidence = _load(EVIDENCE_FILE)
    summaries = evidence["summaries"]
    assert isinstance(summaries, dict)
    expected = {
        "mnistfc:000": (1.3662832175083264, 14.834238849, 3.440155215940509),
        "cifar10_resnet:000": (
            2.4723310047591642,
            60.754404342,
            0.94636016658097,
        ),
        "oval21:000": (1.451084848684865, 11.963770558, 2.6429712163868984),
    }
    for workload_id, (speedup, full_seconds, competitor_ratio) in expected.items():
        summary = summaries[workload_id]
        clause = summary["clause0"]
        full = summary["full_production"]
        assert clause["internal_speedup"] == speedup
        assert clause["semantic_parity"] is True
        assert clause["performance_claimed"] is True
        assert len(clause["audit_raw_e2e_ns"]) == 3
        assert len(clause["production_raw_e2e_ns"]) == 3
        assert full["median_e2e_ns"] / 1e9 == full_seconds
        assert full["production_to_competitor_single_reference_ratio"] == (
            competitor_ratio
        )
        assert full["solver_statuses"] == ["unknown", "unknown", "unknown"]
        assert full["performance_claimed"] is False


def test_production_artifact_rejects_semantic_timing_and_disclosure_tamper() -> None:
    evidence = _load(EVIDENCE_FILE)
    changed_semantic = deepcopy(evidence)
    records = changed_semantic["records"]
    assert isinstance(records, list)
    production = next(row for row in records if row["mode"] == "clause_production")
    production["result"]["semantic_signature_hash"] = "0" * 64
    with pytest.raises(ValueError, match="semantic parity differs"):
        validate_evidence_structure(changed_semantic)

    changed_timing = deepcopy(evidence)
    records = changed_timing["records"]
    assert isinstance(records, list)
    records[0]["e2e_elapsed_ns"] += 1
    with pytest.raises(ValueError, match="summary replay differs"):
        validate_evidence_structure(changed_timing)

    changed_disclosure = deepcopy(evidence)
    records = changed_disclosure["records"]
    assert isinstance(records, list)
    production = next(row for row in records if row["mode"] == "full_production")
    production["result"]["clauses"][0]["selected_native_reexecution"] = True
    with pytest.raises(ValueError, match="production disclosure differs"):
        validate_evidence_structure(changed_disclosure)


def test_production_artifact_logs_and_competitor_boundary_are_bound() -> None:
    evidence = _load(EVIDENCE_FILE)
    records = evidence["records"]
    assert isinstance(records, list)
    for record in records:
        log_path = ARTIFACT_DIR / record["log_path"]
        assert file_sha256(log_path) == record["log_sha256"]
    competitor = evidence["competitor_reference"]
    assert competitor["single_observation_only"] is True
    assert competitor["performance_claimed"] is False
    assert evidence["environment"]["cuda_executed"] is False
