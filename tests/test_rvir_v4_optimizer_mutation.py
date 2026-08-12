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
    ProductionOptimizerStepTraceV4,
    ProductionOptimizerStepV4,
    capture_production_optimizer_controls_v4,
    production_optimizer_controls_from_payload_v4,
    production_optimizer_step_trace_from_payload_v4,
    production_optimizer_step_trace_to_payload_v4,
)
from boundflow.runtime.rvir_v4_production_state import (
    OwnedProductionTensorV4,
    ProductionOptimizerPolicyV4,
    ProductionTensorOwnership,
    ProductionTensorRole,
    production_tensor_sha256,
)
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


def _loss_reduction_sum(value: torch.Tensor) -> torch.Tensor:
    return value.sum(dim=-1)


def _controls_args() -> dict[str, object]:
    return {
        "optimizer": "adam",
        "lr_decay": 0.98,
        "keep_best": True,
        "loss_reduction_func": _loss_reduction_sum,
        "early_stop_patience": 10,
        "start_save_best": 0.5,
        "use_float64_in_last_iteration": False,
        "pruning_in_iteration": True,
        "pruning_in_iteration_threshold": 0.2,
        "max_time": 60.0,
        "enable_alpha_crown": True,
        "enable_beta_crown": True,
        "init_alpha": False,
        "use_shared_alpha": False,
        "apply_output_constraints_to": [],
        "directly_optimize": [],
        "tighten_input_bounds": False,
    }


def _mutation_policy() -> ProductionMutationPolicyV4:
    controls = capture_production_optimizer_controls_v4(
        _controls_args(), cuts_enabled=False
    )
    return ProductionMutationPolicyV4(_production_policy(), controls)


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


def _owned(
    *, path: str, role: ProductionTensorRole, value: torch.Tensor, ordinal: int
) -> OwnedProductionTensorV4:
    ownership = (
        ProductionTensorOwnership.MUTABLE_COPY_OUT
        if role in {ProductionTensorRole.ALPHA, ProductionTensorRole.BETA_VALUE}
        else ProductionTensorOwnership.COPY_IN
    )
    return OwnedProductionTensorV4.own(
        semantic_path=path,
        role=role,
        axes=tuple(f"axis_{axis}" for axis in range(value.ndim)),
        value=value,
        ownership=ownership,
        alias_group=f"alias:{ordinal:06d}",
    )


def _step_state(step: int) -> tuple[OwnedProductionTensorV4, ...]:
    tensors: list[OwnedProductionTensorV4] = []
    for layer in range(6):
        tensors.append(
            _owned(
                path=f"alpha/layer-{layer}/start",
                role=ProductionTensorRole.ALPHA,
                value=torch.full((2, 1, 6, 1), 0.1 * step + layer / 100.0),
                ordinal=len(tensors),
            )
        )
    for layer in range(6):
        tensors.extend(
            (
                _owned(
                    path=f"beta/layer-{layer}/value",
                    role=ProductionTensorRole.BETA_VALUE,
                    value=torch.full(
                        (6, 1), 0.01 * step if layer == 0 else layer / 100.0
                    ),
                    ordinal=len(tensors),
                ),
                _owned(
                    path=f"beta/layer-{layer}/location",
                    role=ProductionTensorRole.BETA_LOCATION,
                    value=torch.full((6, 1), layer, dtype=torch.int64),
                    ordinal=len(tensors) + 1,
                ),
                _owned(
                    path=f"beta/layer-{layer}/sign",
                    role=ProductionTensorRole.BETA_SIGN,
                    value=torch.ones((6, 1)),
                    ordinal=len(tensors) + 2,
                ),
            )
        )
    return tuple(sorted(tensors, key=lambda tensor: tensor.semantic_path))


def _step_trace() -> ProductionOptimizerStepTraceV4:
    steps: list[ProductionOptimizerStepV4] = []
    for ordinal in range(10):
        lower = torch.full((6, 1), -1.0 + ordinal / 100.0)
        steps.append(
            ProductionOptimizerStepV4(
                core_id=0,
                call_id=11 + ordinal,
                parent_call_id=10,
                evaluation_ordinal=ordinal,
                updates_before=ordinal,
                update_after=ordinal < 9,
                optimizer_step_ordinal=ordinal if ordinal < 9 else None,
                alpha_learning_rate=0.01 * 0.98**ordinal,
                beta_learning_rate=0.05 * 0.98**ordinal,
                state_tensors=_step_state(ordinal),
                lower=lower,
                lower_sha256=production_tensor_sha256(lower),
            )
        )
    return ProductionOptimizerStepTraceV4(
        mutation_policy=_mutation_policy(), steps=tuple(steps)
    )


def test_production_policy_maps_ten_evaluations_to_nine_updates() -> None:
    policy = _mutation_policy()

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


def test_full_production_optimizer_controls_are_digest_bound() -> None:
    policy = _mutation_policy()

    assert policy.controls.optimizer == "adam"
    assert policy.controls.lr_decay == 0.98
    assert policy.controls.pruning_in_iteration is True
    assert policy.controls.loss_reduction_id.endswith("_loss_reduction_sum")
    assert len(policy.controls.stable_hash()) == 64
    assert policy.to_dict()["controls"] == policy.controls.to_dict()
    assert (
        production_optimizer_controls_from_payload_v4(policy.controls.to_dict())
        == policy.controls
    )


def test_missing_or_unadmitted_optimizer_controls_fail_closed() -> None:
    missing = _controls_args()
    del missing["lr_decay"]
    with pytest.raises(ValueError, match="controls missing"):
        capture_production_optimizer_controls_v4(missing, cuts_enabled=False)

    controls = capture_production_optimizer_controls_v4(
        _controls_args(), cuts_enabled=True
    )
    with pytest.raises(ValueError, match="not admitted"):
        ProductionMutationPolicyV4(_production_policy(), controls).validate()

    with pytest.raises(ValueError, match="not admitted"):
        ProductionMutationPolicyV4(
            _production_policy(), replace(_mutation_policy().controls, init_alpha=True)
        ).validate()

    payload = _mutation_policy().controls.to_dict()
    payload["keep_best"] = "true"
    with pytest.raises(TypeError, match="boolean fields"):
        production_optimizer_controls_from_payload_v4(payload)

    live = _controls_args()
    live["keep_best"] = "true"
    with pytest.raises(TypeError, match="live boolean fields"):
        capture_production_optimizer_controls_v4(live, cuts_enabled=False)


def test_ten_step_trace_raw_payload_semantically_replays() -> None:
    trace = _step_trace()
    payload = production_optimizer_step_trace_to_payload_v4(trace)

    replayed = production_optimizer_step_trace_from_payload_v4(payload)

    assert replayed.metadata() == trace.metadata()
    assert replayed.metadata()["evaluation_count"] == 10
    assert replayed.metadata()["update_count"] == 9
    assert len(replayed.steps[0].state_tensors) == 24


def test_step_trace_tensor_tamper_is_rejected() -> None:
    payload = production_optimizer_step_trace_to_payload_v4(_step_trace())
    steps = payload["steps"]
    assert isinstance(steps, list)
    lower = steps[3]["lower"]
    assert torch.is_tensor(lower)
    lower[0, 0] += 1.0

    with pytest.raises(ValueError, match="identity/result differs"):
        production_optimizer_step_trace_from_payload_v4(payload)


def test_step_trace_copy_in_drift_rejected_with_valid_tensor_digest() -> None:
    trace = _step_trace()
    changed_steps = list(trace.steps)
    state = list(changed_steps[5].state_tensors)
    target = next(
        index
        for index, tensor in enumerate(state)
        if tensor.role == ProductionTensorRole.BETA_LOCATION
    )
    original = state[target]
    state[target] = _owned(
        path=original.semantic_path,
        role=original.role,
        value=original.value + 1,
        ordinal=target,
    )
    changed_steps[5] = replace(changed_steps[5], state_tensors=tuple(state))

    with pytest.raises(ValueError, match="schema/copy-in drift"):
        ProductionOptimizerStepTraceV4(
            mutation_policy=trace.mutation_policy, steps=tuple(changed_steps)
        ).validate()


def test_step_trace_loop_and_mutation_count_tamper_fail_closed() -> None:
    trace = _step_trace()
    wrong_ordinal = list(trace.steps)
    wrong_ordinal[4] = replace(wrong_ordinal[4], updates_before=3)
    with pytest.raises(ValueError, match="identity/result differs|loop semantics"):
        ProductionOptimizerStepTraceV4(
            mutation_policy=trace.mutation_policy, steps=tuple(wrong_ordinal)
        ).validate()

    wrong_schedule = list(trace.steps)
    wrong_schedule[2] = replace(wrong_schedule[2], alpha_learning_rate=0.01)
    with pytest.raises(ValueError, match="learning-rate schedule"):
        ProductionOptimizerStepTraceV4(
            mutation_policy=trace.mutation_policy, steps=tuple(wrong_schedule)
        ).validate()

    wrong_mutation = list(trace.steps)
    state = list(wrong_mutation[1].state_tensors)
    unchanged_alpha = next(
        index
        for index, tensor in enumerate(state)
        if tensor.role == ProductionTensorRole.ALPHA
    )
    state[unchanged_alpha] = trace.steps[0].state_tensors[unchanged_alpha]
    wrong_mutation[1] = replace(wrong_mutation[1], state_tensors=tuple(state))
    with pytest.raises(ValueError, match="mutation count"):
        ProductionOptimizerStepTraceV4(
            mutation_policy=trace.mutation_policy, steps=tuple(wrong_mutation)
        ).validate()


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
        ProductionMutationPolicyV4(policy, _mutation_policy().controls).validate()


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
