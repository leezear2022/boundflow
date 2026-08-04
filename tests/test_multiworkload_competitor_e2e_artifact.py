"""Contract and tamper tests for the frozen NRIR-18 artifact."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_multiworkload_competitor_e2e_artifact import (
    EVIDENCE_FILE,
    MANIFEST_FILE,
    canonical_hash,
    file_sha256,
    validate_evidence_structure,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "multiworkload-competitor-e2e"
    / "vnncomp21-three-topology-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_multiworkload_artifact_has_exact_ir_and_execution_coverage() -> None:
    manifest = _load(MANIFEST_FILE)
    evidence = _load(EVIDENCE_FILE)
    validate_evidence_structure(evidence)

    assert manifest["evidence_hash"] == canonical_hash(evidence)
    files = manifest["files"]
    assert isinstance(files, dict)
    assert files == {
        str(path.relative_to(ARTIFACT_DIR)): file_sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    records = evidence["records"]
    assert isinstance(records, list)
    statuses = {
        (record["workload_id"], record["backend"]): record["result"]["solver_status"]
        for record in records
    }
    assert statuses == {
        ("mnistfc:000", "boundflow_native"): "unknown",
        ("mnistfc:000", "external_abcrown"): "verified",
        ("cifar10_resnet:000", "boundflow_native"): "unknown",
        ("cifar10_resnet:000", "external_abcrown"): "unknown",
        ("oval21:000", "boundflow_native"): "unknown",
        ("oval21:000", "external_abcrown"): "verified",
    }


def test_multiworkload_artifact_rejects_ir_and_coverage_tamper() -> None:
    evidence = _load(EVIDENCE_FILE)
    changed_plan = deepcopy(evidence)
    changed_plan["ir"]["plan"]["workloads"][0]["query_ir_hash"] = "0" * 64
    with pytest.raises(ValueError, match="linkage differs"):
        validate_evidence_structure(changed_plan)

    missing_record = deepcopy(evidence)
    missing_record["records"].pop()
    with pytest.raises(ValueError, match="coverage differs"):
        validate_evidence_structure(missing_record)


def test_multiworkload_artifact_log_digests_are_semantically_linked() -> None:
    evidence = _load(EVIDENCE_FILE)
    records = evidence["records"]
    assert isinstance(records, list)
    for record in records:
        log_path = ARTIFACT_DIR / record["log_path"]
        assert log_path.is_file()
        assert file_sha256(log_path) == record["log_sha256"]
        assert record["result"]["performance_claimed"] is False
