"""Tests for deterministic FSG2 production-state admission logic."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from scripts.run_fsg2_abcrown_state_inventory import derive_summary


def _state(count: int) -> list[dict[str, object]]:
    return [
        {
            "path": f"tensor.{index}",
            "shape": [2, 1, 6, 164],
            "rank": 4,
        }
        for index in range(count)
    ]


def test_realistic_alpha_beta_inventory_keeps_b2_fail_closed() -> None:
    inventory = {
        "calls": [
            {
                "phase": "alpha_optimize",
                "pre_module_state": {
                    "alpha": _state(6),
                    "sparse_beta": [],
                    "beta": [],
                    "split_beta": [],
                },
                "kwarg_state": {"intermediate_constr": []},
            },
            {
                "phase": "beta_split",
                "pre_module_state": {
                    "alpha": _state(6),
                    "sparse_beta": [],
                    "beta": [],
                    "split_beta": [],
                },
                "kwarg_state": {"intermediate_constr": _state(2)},
            },
        ]
    }

    summary = derive_summary(inventory)

    assert summary["production_alpha_state_observed"] is True
    assert summary["production_beta_phase_observed"] is True
    assert summary["production_beta_state_explicit_before_call"] is False
    assert summary["provider_nested_split_context_observed"] is True
    assert summary["alpha_beta_split_replacement_admitted"] is False
    assert summary["b2_same_solver_timing_admitted"] is False
    assert "split_context_is_nested_in_provider_kwargs" in summary["rejection_reasons"]


def test_empty_inventory_cannot_accidentally_admit_replacement() -> None:
    summary = derive_summary({"calls": []})

    assert summary["phase_call_counts"]["alpha_optimize"] == 0
    assert summary["alpha_beta_split_replacement_admitted"] is False
    assert summary["b2_same_solver_timing_admitted"] is False
