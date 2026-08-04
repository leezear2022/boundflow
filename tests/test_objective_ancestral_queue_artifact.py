"""Frozen NRIR-32 artifact replay-contract tests."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_objective_ancestral_queue_artifact import (
    ARTIFACT_DIR,
    EVIDENCE_FILE,
    _summary,
    validate_evidence_structure,
)


def _evidence() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / ARTIFACT_DIR / EVIDENCE_FILE
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_objective_ancestral_queue_frozen_evidence_passes() -> None:
    evidence = _evidence()
    validate_evidence_structure(evidence)

    summary = evidence["summary"]
    assert isinstance(summary, dict)
    assert summary["ancestral_accepted_nodes"] == [7, 7, 7]
    assert summary["worst_active_lower_deltas"] == [
        95.69998168945312,
        95.69998168945312,
        95.69998168945312,
    ]


def test_objective_ancestral_queue_evidence_rejects_parent_lineage_tamper() -> None:
    evidence = deepcopy(_evidence())
    records = evidence["records"]
    assert isinstance(records, list)
    refinements = records[0]["result"]["ancestral"]["node_refinements"]
    refinements[1]["source_intermediate_constraints_hash"] = "0" * 64

    with pytest.raises(ValueError, match="serialized parent lineage differs"):
        validate_evidence_structure(evidence)


def test_objective_ancestral_queue_evidence_rejects_committed_hash_drift() -> None:
    evidence = deepcopy(_evidence())
    records = evidence["records"]
    assert isinstance(records, list)
    records[1]["result"]["ancestral"]["trace"]["queue_trace_hash"] = "0" * 64
    evidence["summary"] = _summary(records)

    with pytest.raises(ValueError, match="repeated tightness gate differs"):
        validate_evidence_structure(evidence)
