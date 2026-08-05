"""Contract tests for NRIR48 additive execution-cost attribution."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import copy

import pytest

from scripts import run_top2_production_execution_cost_attribution as nrir48


def _timer(value: int, calls: int = 1) -> dict[str, int]:
    return {"calls": calls, "inclusive_ns": value}


def _attribution() -> dict[str, object]:
    raw = {
        "child_refinement_compile": _timer(200, 30),
        "child_refinement_execute": _timer(400, 30),
        "optimizer_warm_state": _timer(10, 15),
        "optimizer_template_acquire": _timer(10, 16),
        "optimizer_instantiate": _timer(10, 16),
        "optimizer_ir_lower": _timer(10, 16),
        "optimizer_execute": _timer(100, 16),
        "branch_bind_score": _timer(100, 16),
        "queue_make_batch_commit": _timer(10, 16),
        "queue_task_emit": _timer(10, 33),
        "queue_schedule_lower": _timer(10),
        "execute_fast_validate": _timer(20, 90),
        "execute_target_select": _timer(40, 30),
        "execute_selected_crown": _timer(200, 30),
        "execute_propagate_forward": _timer(20, 30),
    }
    return {
        "queue_elapsed_ns": 1_000,
        "raw_timers": raw,
        "materialize_action_ns": 50,
        "top_categories": {
            "child_refinement_compile_ns": 200,
            "child_refinement_execute_ns": 400,
            "optimizer_prepare_ns": 40,
            "optimizer_execute_ns": 100,
            "branch_bind_score_ns": 100,
            "materialize_commit_ns": 80,
            "queue_control_residual_ns": 80,
        },
        "child_execute_categories": {
            "fast_validate_ns": 20,
            "runtime_target_select_ns": 40,
            "selected_crown_ns": 200,
            "propagate_forward_ns": 20,
            "refinement_hash_trace_residual_ns": 120,
        },
        "closure_error_ns": 0,
    }


def test_attribution_accepts_exact_mutually_exclusive_derivation() -> None:
    nrir48._validate_attribution(_attribution())


def test_attribution_rejects_synchronized_top_level_rehash_tamper() -> None:
    value = copy.deepcopy(_attribution())
    top = value["top_categories"]
    assert isinstance(top, dict)
    top["child_refinement_execute_ns"] += 1
    top["queue_control_residual_ns"] -= 1
    with pytest.raises(ValueError, match="closure|derivation"):
        nrir48._validate_attribution(value)


def test_attribution_rejects_child_category_overlap() -> None:
    value = copy.deepcopy(_attribution())
    child = value["child_execute_categories"]
    assert isinstance(child, dict)
    child["selected_crown_ns"] += 1
    with pytest.raises(ValueError, match="closure"):
        nrir48._validate_attribution(value)


def test_profiler_execute_only_scope_excludes_compile_calls() -> None:
    profiler = nrir48._Profiler()

    def operation() -> int:
        return 7

    execute_only = profiler.wrap(operation, "leaf", execute_only=True)
    assert execute_only() == 7
    assert profiler.calls("leaf") == 0
    execute_scope = profiler.wrap(execute_only, "parent", execute_scope=True)
    assert execute_scope() == 7
    assert profiler.calls("leaf") == 1
    assert profiler.calls("parent") == 1
