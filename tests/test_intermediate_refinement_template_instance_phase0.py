"""Tests for NRIR46 Phase0 compiler-ownership attribution."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from typing import cast

import pytest

from scripts.run_intermediate_refinement_template_instance_phase0 import (
    STATIC_SHAREABLE_GATE_NS,
    _attribution_summary,
    _timer_ns,
)


def _row(calls: int, nanoseconds: int) -> dict[str, int]:
    return {
        "calls": calls,
        "inclusive_ns": nanoseconds,
        "exclusive_ns": nanoseconds,
    }


def test_template_instance_attribution_keeps_dynamic_target_selection() -> None:
    timers = {
        "prepared_compile_total": _row(60, 5_000_000_000),
        "select_targets": _row(124, 1_860_000_000),
        "module_validate": _row(120, 1_000_000),
        "policy_validate": _row(240, 1_000_000),
        "primal_graph_hash": _row(120, 40_000_000),
        "lower_legacy_ir": _row(60, 1_200_000_000),
        "lower_prepared_ir": _row(60, 20_000_000),
    }

    result = _attribution_summary(timers, program_count=60)

    assert result["semantic_target_selection_calls"] == 60
    assert result["redundant_target_selection_calls"] == 64
    assert result["static_topology_ns"] == 1_262_000_000
    assert result["strict_static_gate_passed"] is False
    assert (
        cast(int, result["template_instance_convertible_ns"]) > STATIC_SHAREABLE_GATE_NS
    )
    assert result["ownership_ceiling_gate_passed"] is True


def test_phase0_timer_lookup_fails_closed() -> None:
    with pytest.raises(ValueError, match="timer is absent"):
        _timer_ns({}, "missing", "calls")
    with pytest.raises(ValueError, match="timer differs"):
        _timer_ns({"bad": {"calls": -1}}, "bad", "calls")
