"""Frozen NRIR-41 causal-attribution artifact and tamper gates."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_objective_branch_production_cost_attribution import (
    _canonical_hash,
    _formal_semantic,
    validate_attribution,
)

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/objective-branch-production-cost-attribution"
    / "vnncomp21-resnet2b-property0-three-repeat-cpu-v1/formal.json"
)


def _formal() -> dict[str, Any]:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_attribution_selects_scorer_ownership_route() -> None:
    """Both preregistered causal gates hold across the frozen evidence."""

    formal = _formal()
    validate_attribution(formal)
    decision = formal["decision"]
    assert decision["frontier_order_retained"] is True
    assert decision["scoring_cost_dominant"] is True
    assert decision["next_route"] == "optimize_scorer_ownership"
    assert min(decision["frontier_improvements"].values()) > 0.0
    assert min(decision["queue_ratios"].values()) >= 1.2
    assert min(decision["branch_program_shares"].values()) >= 0.2


def test_frozen_attribution_rejects_rehashed_prefix_tamper() -> None:
    """An outer rehash cannot replace the independently reconstructed prefix."""

    formal = deepcopy(_formal())
    formal["prefix_rows"][0]["worst_active_lower"] += 1.0
    formal["formal_hash"] = _canonical_hash(_formal_semantic(formal))
    with pytest.raises(ValueError, match="formal derived evidence differs"):
        validate_attribution(formal)
