"""RVIR-v4 V4-2 production optimizer policy admission tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_beta_crown import run_alpha_beta_crown_mlp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    ProductionMutationPolicyV4,
)
from boundflow.runtime.rvir_v4_production_state import ProductionOptimizerPolicyV4
from boundflow.runtime.task_executor import InputSpec


def _production_policy() -> ProductionOptimizerPolicyV4:
    return ProductionOptimizerPolicyV4(
        iteration=10,
        alpha_learning_rate=0.01,
        beta_learning_rate=0.05,
        bound_lower=True,
        bound_upper=False,
        fix_intermediate_bounds=True,
        deterministic=False,
        stop_criterion_id=(
            "auto_LiRPA.utils.stop_criterion_batch_any.<locals>.<lambda>"
        ),
    )


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="rvir-v4-policy",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="rvir-v4-policy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0]]),
                "b1": torch.tensor([0.0]),
                "W2": torch.tensor([[1.0]]),
                "b2": torch.tensor([0.0]),
            }
        },
    )


def test_production_policy_maps_ten_evaluations_to_nine_updates() -> None:
    policy = ProductionMutationPolicyV4(_production_policy())

    assert policy.evaluation_count == 10
    assert policy.update_count == 9
    native = policy.to_native_policy()
    assert native.steps == 9
    assert native.lr == 0.01
    assert native.effective_beta_lr == 0.05
    assert native.to_dict()["beta_lr"] == 0.05
    assert len(policy.stable_hash()) == 64


def test_legacy_unified_learning_rate_payload_remains_compatible() -> None:
    policy = NativeAlphaBetaOptimizerPolicy(steps=3, lr=0.2)

    assert policy.effective_beta_lr == 0.2
    assert "beta_lr" not in policy.to_dict()


@pytest.mark.parametrize(
    "policy",
    [
        replace(_production_policy(), iteration=0),
        replace(_production_policy(), bound_upper=True),
        replace(_production_policy(), fix_intermediate_bounds=False),
        replace(_production_policy(), stop_criterion_id="always-false"),
    ],
)
def test_nonproduction_mutation_policy_is_rejected(
    policy: ProductionOptimizerPolicyV4,
) -> None:
    with pytest.raises(ValueError, match="not admitted"):
        ProductionMutationPolicyV4(policy).validate()


def test_optimizer_uses_distinct_alpha_and_beta_parameter_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[float] = []
    original = torch.optim.Adam

    def capture_adam(*args: Any, **kwargs: Any) -> Any:
        groups = list(args[0])
        observed.extend(float(group["lr"]) for group in groups)
        return original(groups, *args[1:], **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", capture_adam)
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[0.0]]), eps=1.0)
    run_alpha_beta_crown_mlp(
        _module(),
        spec,
        linear_spec_C=torch.tensor([[1.0]]),
        relu_split_state={"h1": torch.tensor([[1]], dtype=torch.int8)},
        steps=1,
        lr=0.01,
        beta_lr=0.05,
        beta_init=0.1,
        per_batch_params=True,
    )

    assert observed == [0.01, 0.05]


def test_nonfinite_or_nonpositive_beta_learning_rate_is_rejected() -> None:
    for value in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="beta_lr"):
            run_alpha_beta_crown_mlp(
                _module(),
                InputSpec.linf(
                    value_name="input", center=torch.tensor([[0.0]]), eps=1.0
                ),
                relu_split_state={"h1": torch.tensor([[1]], dtype=torch.int8)},
                steps=1,
                lr=0.01,
                beta_lr=value,
            )
