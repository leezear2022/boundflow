"""Deterministic checks for the committed NRIR-31 artifact."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_objective_hard_clause_escalation_artifact import (
    _summaries,
    validate_evidence_structure,
)

ARTIFACT = Path(
    "artifacts/objective-hard-clause-escalation/"
    "vnncomp21-three-topology-cpu-v1/evidence.json"
)


def _evidence() -> dict[str, object]:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_objective_hard_clause_artifact_validates_and_covers_three_repeats() -> None:
    evidence = _evidence()

    validate_evidence_structure(evidence)
    assert len(evidence["records"]) == 9  # type: ignore[arg-type]
    assert evidence["property_status"] == "validated-reduced"
    summaries = evidence["summaries"]
    assert isinstance(summaries, dict)
    resnet = summaries["cifar10_resnet:000"]
    assert resnet["minimum_root_lower_delta"] == pytest.approx(81.5225830078125)
    assert resnet["maximum_root_lower_delta"] == pytest.approx(179.970458984375)


def test_objective_hard_clause_artifact_rejects_source_lineage_tamper() -> None:
    evidence = deepcopy(_evidence())
    records = evidence["records"]
    assert isinstance(records, list)
    records[0]["result"]["clauses"][0]["source_refinement_semantic_trace_hash"] = (
        "0" * 64
    )

    with pytest.raises(ValueError, match="provenance differs"):
        validate_evidence_structure(evidence)


def test_objective_hard_clause_artifact_rejects_synchronized_gate_tamper() -> None:
    evidence = deepcopy(_evidence())
    records = evidence["records"]
    assert isinstance(records, list)
    for record in records:
        if record["workload_id"] != "cifar10_resnet:000":
            continue
        for row in record["comparison"]["root_comparisons"]:
            row["root_lower_delta"] = 0.0
            row["strict_improvement_gt_1e_4"] = False
    evidence["summaries"] = _summaries(records)

    with pytest.raises(ValueError, match="strict-tightness gate differs"):
        validate_evidence_structure(evidence)
