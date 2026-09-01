"""Tests for AOT preparation at the solver request boundary."""

# pylint: disable=missing-function-docstring,protected-access
# pylint: disable=too-few-public-methods

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from boundflow.runtime.prepared_verification_request import (
    prepare_verification_request_v1,
)


class _Handler:
    def __init__(self) -> None:
        self.x = torch.tensor([1.0])
        self.rhs = torch.tensor([2.0])
        self.vnnlib = [([(-1.0, 1.0)], [])]
        self.current_index = 7
        self.all_specs = object()

    def _set_all_specs(self) -> None:
        self.all_specs = (self.x.clone(), self.rhs.clone(), self.current_index)


class _Solver:
    def __init__(self) -> None:
        self.config = {
            "name": "test",
            "specification": {"rhs_offset": None},
            "debug": {"sanity_check": False},
        }
        self.constraint = None
        self.spec = None
        self._runtime_spec = None
        self.environment_count = 0

    def _prepare_environment(self, device: str) -> None:
        assert device == "cuda"
        self.environment_count += 1

    def _prepare_model(self, device: str) -> object:
        return {"device": device}

    def _prepare_runtime_spec(self) -> object:
        self._runtime_spec = {"clauses": [[1]]}
        return self._runtime_spec

    def _build_vnnlib_handler(self, _runtime_spec: object) -> _Handler:
        return _Handler()


def test_prepared_request_reuses_model_and_clones_mutable_state() -> None:
    solver = _Solver()
    prepared = prepare_verification_request_v1(
        solver=solver,
        constraint={"property": "p0"},
        device="cuda",
        torch_module=torch,
        config_context=lambda _config: nullcontext(),
    )
    original_model_method = solver._prepare_model
    with prepared.activate():
        assert solver._prepare_model("cuda") is prepared.model
        runtime_spec = solver._prepare_runtime_spec()
        handler = solver._build_vnnlib_handler(runtime_spec)
        handler.x.add_(10)
        handler.rhs.add_(10)
        assert prepared.handler.x.tolist() == [1.0]
        assert prepared.handler.rhs.tolist() == [2.0]
        assert handler.current_index == 0
    assert solver._prepare_model.__func__ is original_model_method.__func__
    receipt = prepared.receipt().to_dict()
    assert receipt["model_reuse_count"] == 1
    assert receipt["runtime_spec_clone_count"] == 1
    assert receipt["handler_clone_count"] == 1
    assert receipt["model_reuse_ns"] > 0
    assert receipt["runtime_spec_clone_ns"] > 0
    assert receipt["handler_clone_ns"] > 0
    assert receipt["static_prepare_excluded_from_query"] is True
    assert receipt["performance_claimed"] is False


def test_copy_on_prune_policy_shares_only_immutable_initial_projection() -> None:
    solver = _Solver()
    prepared = prepare_verification_request_v1(
        solver=solver,
        constraint={"property": "p0"},
        device="cuda",
        torch_module=torch,
        config_context=lambda _config: nullcontext(),
        copy_on_prune_handler=True,
    )
    with prepared.activate():
        assert solver._prepare_model("cuda") is prepared.model
        runtime_spec = solver._prepare_runtime_spec()
        handler = solver._build_vnnlib_handler(runtime_spec)
        assert handler.all_specs is prepared.handler.all_specs
        assert handler.vnnlib is prepared.handler.vnnlib
        handler.x = handler.x[0:0]
        handler.rhs = handler.rhs[0:0]
        assert prepared.handler.x.tolist() == [1.0]
        assert prepared.handler.rhs.tolist() == [2.0]
    assert (
        prepared.receipt().handler_clone_policy
        == "share-immutable-initial-copy-on-prune"
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("specification", "rhs_offset", 0.1),
        ("debug", "sanity_check", "Full"),
    ),
)
def test_copy_on_prune_rejects_handler_mutation_modes(
    section: str, key: str, value: object
) -> None:
    solver = _Solver()
    solver.config[section][key] = value
    with pytest.raises(ValueError, match="not admissible"):
        prepare_verification_request_v1(
            solver=solver,
            constraint={"property": "p0"},
            device="cuda",
            torch_module=torch,
            config_context=lambda _config: nullcontext(),
            copy_on_prune_handler=True,
        )
