"""Native alpha/beta state, beta execution, and warm-start contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.bound import OptimizedReluRelaxationAttrs
from boundflow.ir.schedule import LaunchAction
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    build_native_alpha_beta_scope,
    classify_native_alpha_beta_warm_start,
    compile_native_alpha_beta_state_query,
    execute_native_alpha_beta_state_query,
    optimize_native_alpha_beta_state,
)
from boundflow.runtime.native_verifier_ir_integration import (
    execute_native_plain_crown_representation_query,
)
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="native-alpha-beta-toy",
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
        entry_task_id="native-alpha-beta-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0]]),
                "b1": torch.tensor([0.0]),
                "W2": torch.tensor([[1.0]]),
                "b2": torch.tensor([0.0]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.linf(value_name="input", center=torch.tensor([[0.0]]), eps=1.0)


def _policy(*, steps: int = 5) -> NativeAlphaBetaOptimizerPolicy:
    return NativeAlphaBetaOptimizerPolicy(
        steps=steps, lr=0.2, alpha_init=0.5, beta_init=0.1
    )


def test_frozen_state_executes_beta_constraint_through_all_ir_layers() -> None:
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0]])
    split = {"h1": torch.tensor([[1]], dtype=torch.int8)}
    optimized = optimize_native_alpha_beta_state(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=_policy(),
    )
    compilation = compile_native_alpha_beta_state_query(
        module,
        spec,
        linear_spec_C=objective,
        optimization=optimized,
        query_id="native-alpha-beta-toy",
    )
    result, trace = execute_native_alpha_beta_state_query(
        compilation,
        module,
        spec,
        linear_spec_C=objective,
        optimization=optimized,
    )

    assert torch.equal(result.lower, optimized.bounds.lower)
    assert torch.equal(result.upper, optimized.bounds.upper)
    assert compilation.build.module.domain.alpha_enabled is True
    assert compilation.build.module.domain.beta_enabled is True
    assert compilation.build.module.domain.split_state_present is True
    optimized_ops = [
        op
        for op in compilation.build.module.graph.ops
        if isinstance(op.attrs, OptimizedReluRelaxationAttrs)
    ]
    assert len(optimized_ops) == 1
    assert len(optimized_ops[0].inputs) == 7
    assert compilation.source_template.workload.alpha_enabled is True
    assert compilation.source_template.workload.beta_enabled is True
    assert len(trace.events) == len(compilation.task_module.tasks)
    assert len(
        [
            action
            for action in compilation.schedule.actions
            if isinstance(action, LaunchAction)
        ]
    ) == len(compilation.task_module.tasks)

    zero_beta_state = NativeAlphaBetaOptimizationState(
        scope=optimized.state.scope,
        split_by_relu_input=optimized.state.split_by_relu_input,
        alpha_by_relu_input=optimized.state.alpha_by_relu_input,
        beta_by_relu_input=(("h1", torch.zeros_like(optimized.state.betas["h1"])),),
    )
    zero_beta = replace(optimized, state=zero_beta_state)
    zero_compilation = compile_native_alpha_beta_state_query(
        module,
        spec,
        linear_spec_C=objective,
        optimization=zero_beta,
        query_id="native-alpha-beta-zero-beta",
    )
    zero_result, _trace = execute_native_alpha_beta_state_query(
        zero_compilation,
        module,
        spec,
        linear_spec_C=objective,
        optimization=zero_beta,
    )
    assert float(result.lower.item()) > float(zero_result.lower.item())
    assert compilation.hashes() != zero_compilation.hashes()


def test_warm_start_exact_refinement_and_rejection_are_distinct() -> None:
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0]])
    parent_split = {"h1": torch.tensor([[0]], dtype=torch.int8)}
    parent = optimize_native_alpha_beta_state(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=parent_split,
        policy=_policy(steps=1),
    )
    exact = classify_native_alpha_beta_warm_start(
        parent.state,
        target_scope=parent.state.scope,
        target_split_state=parent_split,
    )
    assert exact.kind == "exact"
    assert exact.exact_state_reuse_allowed is True

    child_split = {"h1": torch.tensor([[1]], dtype=torch.int8)}
    _child_env, child_pre = _forward_ibp_trace_mlp(
        module, spec, relu_split_state=child_split
    )
    child_scope = build_native_alpha_beta_scope(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=child_pre,
        relu_split_state=child_split,
        policy=_policy(steps=1),
    )
    refinement = classify_native_alpha_beta_warm_start(
        parent.state,
        target_scope=child_scope,
        target_split_state=child_split,
    )
    assert refinement.kind == "monotonic_split_refinement"
    assert refinement.alpha_initialization_allowed is True
    assert refinement.beta_initialization_allowed is True
    assert refinement.exact_state_reuse_allowed is False
    child = optimize_native_alpha_beta_state(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=child_split,
        policy=_policy(steps=1),
        warm_start=parent.state,
    )
    assert child.warm_start_decision == refinement

    reversed_split = {"h1": torch.tensor([[-1]], dtype=torch.int8)}
    _reversed_env, reversed_pre = _forward_ibp_trace_mlp(
        module, spec, relu_split_state=reversed_split
    )
    reversed_scope = build_native_alpha_beta_scope(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=reversed_pre,
        relu_split_state=reversed_split,
        policy=_policy(steps=1),
    )
    rejected = classify_native_alpha_beta_warm_start(
        child.state,
        target_scope=reversed_scope,
        target_split_state=reversed_split,
    )
    assert rejected.kind == "rejected"
    assert "split_reversal_or_removal" in rejected.reason

    objective_drift = classify_native_alpha_beta_warm_start(
        parent.state,
        target_scope=replace(parent.state.scope, objective_hash="f" * 64),
        target_split_state=parent_split,
    )
    assert objective_drift.kind == "rejected"
    assert "objective_hash" in objective_drift.reason


def test_runtime_payload_tampering_fails_closed() -> None:
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0]])
    split = {"h1": torch.tensor([[1]], dtype=torch.int8)}
    optimized = optimize_native_alpha_beta_state(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=_policy(steps=1),
    )
    compilation = compile_native_alpha_beta_state_query(
        module,
        spec,
        linear_spec_C=objective,
        optimization=optimized,
        query_id="native-alpha-beta-tamper",
    )
    bad_alpha = {"h1": optimized.state.alphas["h1"].clone()}
    bad_alpha["h1"][0, 0] = 2.0
    with pytest.raises(ValueError, match="alpha lies outside"):
        execute_native_plain_crown_representation_query(
            compilation,
            legacy_task_module=module,
            input_spec=spec,
            relu_pre=optimized.relu_pre,
            linear_spec_C=objective,
            relu_split_state=optimized.state.splits,
            relu_alpha_state=bad_alpha,
            relu_beta_state=optimized.state.betas,
        )

    bad_beta = {"h1": optimized.state.betas["h1"].clone()}
    bad_beta["h1"][0, 0] = -0.1
    with pytest.raises(ValueError, match="beta contains negative"):
        execute_native_plain_crown_representation_query(
            compilation,
            legacy_task_module=module,
            input_spec=spec,
            relu_pre=optimized.relu_pre,
            linear_spec_C=objective,
            relu_split_state=optimized.state.splits,
            relu_alpha_state=optimized.state.alphas,
            relu_beta_state=bad_beta,
        )
