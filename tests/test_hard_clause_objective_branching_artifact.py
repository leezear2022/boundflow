"""Frozen NRIR-17 objective branching artifact and tamper gates."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_hard_clause_objective_branching_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    MANIFEST_FILE,
    canonical_hash,
    validate_evidence,
)

ARTIFACT_DIR = Path(
    "artifacts/hard-clause-objective-branching/vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load() -> tuple[dict[str, object], dict[str, object]]:
    manifest = json.loads((ARTIFACT_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
    artifact = json.loads((ARTIFACT_DIR / ARTIFACT_FILE).read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    assert isinstance(artifact, dict)
    return manifest, artifact


def test_frozen_objective_branching_artifact_digest_and_claim_boundary() -> None:
    manifest, artifact = _load()
    evidence = artifact["evidence"]

    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert artifact["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert evidence["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert manifest["performance_claimed"] is False
    assert evidence["performance_claimed"] is False
    assert evidence["property_status"] == "unknown"
    assert manifest["files"] == {
        ARTIFACT_FILE: file_sha256(ARTIFACT_DIR / ARTIFACT_FILE)
    }
    assert manifest["evidence_hash"] == canonical_hash(evidence)
    validate_evidence(evidence)


def test_all_hard_clauses_improve_worst_leaf_without_claim_upgrade() -> None:
    _manifest, artifact = _load()
    evidence = artifact["evidence"]
    comparisons = evidence["comparisons"]

    assert set(comparisons) == {"0", "2", "4"}
    assert all(item["objective_not_weaker"] is True for item in comparisons.values())
    assert comparisons["0"]["objective_minus_widest_leaf_worst"] > 0.12
    assert comparisons["2"]["objective_minus_widest_leaf_worst"] > 0.07
    assert comparisons["4"]["objective_minus_widest_leaf_worst"] > 0.05
    assert all(
        evidence["clauses"][clause]["objective"]["summary"]["property_status"]
        == "unknown"
        for clause in ("0", "2", "4")
    )


def test_artifact_semantic_and_claim_tamper_fail_closed() -> None:
    _manifest, artifact = _load()
    evidence = artifact["evidence"]

    selected = deepcopy(evidence)
    trace = selected["clauses"]["0"]["objective"]["objective_branches"][0]["trace"]
    trace["selected_candidate_ordinal"] = (
        trace["selected_candidate_ordinal"] + 1
    ) % len(trace["scores"])
    with pytest.raises(ValueError, match="selection|IR evidence"):
        validate_evidence(selected)

    schedule = deepcopy(evidence)
    schedule["clauses"]["2"]["objective"]["objective_branches"][0]["schedule"][
        "actions"
    ].pop()
    with pytest.raises(ValueError, match="branch IR evidence"):
        validate_evidence(schedule)

    claim = deepcopy(evidence)
    claim["performance_claimed"] = True
    with pytest.raises(ValueError, match="header"):
        validate_evidence(claim)
