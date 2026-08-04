"""Semantic nesting and closure-gate tests for the NRIR-29 artifact."""

import hashlib

from scripts import run_wall_clock_parametric_bab_scaling_artifact as artifact


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _result(
    budget_id: str,
    max_nodes: int,
    max_depth: int,
    domain_count: int,
    verified_count: int,
) -> dict[str, object]:
    clauses = []
    for clause_index in range(9):
        domains = [
            {
                "split_state_hash": _digest(f"clause-{clause_index}-domain-{index}"),
                "lower": float(clause_index) + index / 100.0,
            }
            for index in range(domain_count)
        ]
        clauses.append({"domains": domains})
    return {
        "budget": {
            "budget_id": budget_id,
            "max_nodes": max_nodes,
            "max_depth": max_depth,
        },
        "completed_clause_count": 9,
        "completed_clause_indices": list(range(9)),
        "pending_clause_indices": [],
        "verified_clause_indices": list(range(verified_count)),
        "clauses": clauses,
    }


def test_scaling_group_requires_nested_domains_and_strict_verified_gain(
    monkeypatch,
) -> None:
    monkeypatch.setattr(artifact, "_validate_worker_result", lambda _result: None)
    gate = artifact.validate_scaling_group(
        (
            _result("n7d2", 7, 2, 1, 6),
            _result("n31d4", 31, 4, 2, 8),
            _result("n127d6", 127, 6, 3, 8),
        )
    )

    assert gate["all_completed"] is True
    assert gate["domain_nested"] is True
    assert gate["no_verified_regression"] is True
    assert gate["strict_verified_increase"] is True
    assert gate["common_domain_lower_max_diff"] == 0.0


def test_scaling_group_detects_synced_digest_semantic_tamper(monkeypatch) -> None:
    monkeypatch.setattr(artifact, "_validate_worker_result", lambda _result: None)
    small = _result("n7d2", 7, 2, 1, 6)
    medium = _result("n31d4", 31, 4, 2, 8)
    large = _result("n127d6", 127, 6, 3, 8)
    medium["clauses"][0]["domains"][0]["lower"] += 1e-3

    gate = artifact.validate_scaling_group((small, medium, large))

    assert gate["domain_nested"] is False
    assert gate["common_domain_lower_max_diff"] > 1e-5
