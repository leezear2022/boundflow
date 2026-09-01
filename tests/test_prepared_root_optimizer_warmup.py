"""Tests for exact root optimizer warmup preparation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from boundflow.runtime.prepared_root_optimizer_warmup import (
    prepare_root_optimizer_warmup_v1,
)


class _FakeSolver:
    def __init__(self) -> None:
        self.config = {"general": {"complete_verifier": "bab"}}
        self.constraint = "old-constraint"
        self.spec = "old-spec"
        self.vnnlib_handler = "old-handler"
        self.spec_handler_incomplete = "old-incomplete"
        self._last_result = "old-result"
        self.observed_policy = ""

    def verify(self, *, constraints: object) -> object:
        self.observed_policy = self.config["general"]["complete_verifier"]
        self.constraint = constraints
        self.spec = constraints
        self.vnnlib_handler = "mutated-handler"
        self.spec_handler_incomplete = "mutated-incomplete"
        self._last_result = "mutated-result"
        return SimpleNamespace(
            status="unknown",
            reference={"lower_bounds": {"a": torch.zeros(2), "b": torch.ones(3)}},
        )


def test_root_optimizer_warmup_restores_solver_state_and_accounts_output() -> None:
    solver = _FakeSolver()
    receipt = prepare_root_optimizer_warmup_v1(
        solver=solver, constraints="new-constraint", torch_module=torch
    ).to_dict()
    assert solver.observed_policy == "skip"
    assert solver.config["general"]["complete_verifier"] == "bab"
    assert solver.constraint == "old-constraint"
    assert solver.spec == "old-spec"
    assert solver.vnnlib_handler == "old-handler"
    assert solver.spec_handler_incomplete == "old-incomplete"
    assert solver._last_result == "old-result"
    assert receipt["lower_bound_tensor_count"] == 2
    assert receipt["lower_bound_element_count"] == 5
    assert receipt["query_timing_excluded"] is True
    assert receipt["performance_claimed"] is False


def test_root_optimizer_warmup_rejects_missing_complete_policy() -> None:
    solver = _FakeSolver()
    solver.config = {"general": {}}
    with pytest.raises(TypeError, match="general config"):
        prepare_root_optimizer_warmup_v1(
            solver=solver, constraints="new-constraint", torch_module=torch
        )
