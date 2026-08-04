"""Native alpha/beta optimizer Plan/Task/Schedule control contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.domains.interval import IntervalState
from boundflow.ir.bound import (
    BoundOpKind,
    IntermediateBoundSource,
    ReluRelaxationAttrs,
)
from boundflow.ir.optimizer import OptimizerTaskKind
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationResult,
    NativeAlphaBetaOptimizerPolicy,
    compile_native_alpha_beta_state_query,
    execute_native_alpha_beta_state_query,
    optimize_native_alpha_beta_state,
)
from boundflow.runtime.native_alpha_beta_optimizer_schedule import (
    NativePreparedOptimizerProgram,
    compile_native_alpha_beta_optimizer_program,
    execute_native_alpha_beta_optimizer_program,
    execute_prepared_native_alpha_beta_optimizer_program,
)
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="optimizer-schedule-toy",
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
        entry_task_id="optimizer-schedule-toy",
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


def _policy(*, steps: int = 2) -> NativeAlphaBetaOptimizerPolicy:
    return NativeAlphaBetaOptimizerPolicy(
        steps=steps,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
    )


def _compile(*, steps: int = 2):
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0]])
    split = {"h1": torch.tensor([[1]], dtype=torch.int8)}
    program = compile_native_alpha_beta_optimizer_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=_policy(steps=steps),
        program_id=f"optimizer-schedule-toy-s{steps}",
    )
    return module, spec, objective, split, program


def test_schedule_drives_optimizer_and_matches_legacy_oracle() -> None:
    module, spec, objective, split, program = _compile()
    scheduled = execute_native_alpha_beta_optimizer_program(
        program, module, spec, linear_spec_C=objective
    )
    legacy = optimize_native_alpha_beta_state(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=_policy(),
    )

    assert torch.equal(scheduled.bounds.lower, legacy.bounds.lower)
    assert torch.equal(scheduled.bounds.upper, legacy.bounds.upper)
    assert torch.equal(scheduled.state.alphas["h1"], legacy.state.alphas["h1"])
    assert torch.equal(scheduled.state.betas["h1"], legacy.state.betas["h1"])
    assert len(program.task_module.tasks) == len(program.schedule.actions) == 13
    assert len(scheduled.trace.actions) == 13
    assert len(scheduled.trace.evaluations) == 3
    assert scheduled.trace.best_iteration_by_domain == (2,)
    backward = [
        action
        for action in scheduled.trace.actions
        if action.kind == OptimizerTaskKind.BACKWARD
    ]
    projected = [
        action
        for action in scheduled.trace.actions
        if action.kind == OptimizerTaskKind.PROJECT_STATE
    ]
    assert len(backward) == len(projected) == 2
    assert all(action.beta_gradient_l1 == 1.0 for action in backward)
    assert all(action.projection_applied for action in projected)


def test_selected_state_executes_through_frozen_native_compiler_stack() -> None:
    module, spec, objective, _split, program = _compile()
    scheduled = execute_native_alpha_beta_optimizer_program(
        program, module, spec, linear_spec_C=objective
    )
    selected = NativeAlphaBetaOptimizationResult(
        bounds=scheduled.bounds,
        state=scheduled.state,
        interval_env=program.interval_env,
        relu_pre=program.relu_pre,
        warm_start_decision=program.warm_start_decision,
    )
    compilation = compile_native_alpha_beta_state_query(
        module,
        spec,
        linear_spec_C=objective,
        optimization=selected,
        query_id="optimizer-schedule-selected-state",
    )
    native_bounds, native_trace = execute_native_alpha_beta_state_query(
        compilation,
        module,
        spec,
        linear_spec_C=objective,
        optimization=selected,
    )

    assert torch.equal(native_bounds.lower, scheduled.bounds.lower)
    assert torch.equal(native_bounds.upper, scheduled.bounds.upper)
    assert len(native_trace.events) == len(compilation.task_module.tasks)
    assert compilation.hashes() != program.source_compilation.hashes()


@pytest.mark.parametrize("tamper", ["sequence", "task_id", "input", "output"])
def test_schedule_task_linkage_tampering_fails_closed(tamper: str) -> None:
    _module_value, _spec_value, _objective, _split, program = _compile()
    action = program.schedule.actions[0]
    changes: dict[str, object] = {}
    if tamper == "sequence":
        changes["sequence"] = 1
    elif tamper == "task_id":
        changes["task_id"] = "optimizer.reduce_metric.s000"
    elif tamper == "input":
        changes["input_value_ids"] = ("optimizer.state.s999",)
    else:
        changes["output_value_ids"] = ("optimizer.bounds.s999",)
    bad_action = replace(action, **changes)
    bad_schedule = replace(
        program.schedule,
        actions=(bad_action, *program.schedule.actions[1:]),
    )

    with pytest.raises(ValueError, match="Schedule/Task linkage differs"):
        bad_schedule.validate(plan=program.plan, task_module=program.task_module)


def test_plan_and_execution_hash_tampering_fail_closed() -> None:
    module, spec, objective, _split, program = _compile()
    scheduled = execute_native_alpha_beta_optimizer_program(
        program, module, spec, linear_spec_C=objective
    )
    bad_plan = replace(program.plan, initial_state_hash="f" * 64)
    with pytest.raises(ValueError, match="Task IR module is invalid"):
        program.schedule.validate(plan=bad_plan, task_module=program.task_module)

    first = scheduled.trace.actions[0]
    output_id = first.output_hashes[0][0]
    bad_first = replace(first, output_hashes=((output_id, "f" * 64),))
    bad_trace = replace(
        scheduled.trace,
        actions=(bad_first, *scheduled.trace.actions[1:]),
    )
    with pytest.raises(ValueError, match="evaluation bound hash differs"):
        bad_trace.validate(program=program)


def test_warm_start_refinement_is_bound_into_optimizer_plan() -> None:
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
    child_split = {"h1": torch.tensor([[1]], dtype=torch.int8)}
    child = compile_native_alpha_beta_optimizer_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=child_split,
        policy=_policy(steps=1),
        program_id="optimizer-schedule-warm-child",
        warm_start=parent.state,
    )

    assert child.warm_start_decision is not None
    assert child.warm_start_decision.kind == "monotonic_split_refinement"
    assert child.plan.warm_start_kind == "monotonic_split_refinement"
    assert torch.equal(child.initial_state.alphas["h1"], parent.state.alphas["h1"])
    assert torch.equal(child.initial_state.betas["h1"], parent.state.betas["h1"])

    with pytest.raises(ValueError, match="semantic_scope_drift:objective_hash"):
        compile_native_alpha_beta_optimizer_program(
            module,
            spec,
            linear_spec_C=torch.tensor([[2.0]]),
            relu_split_state=child_split,
            policy=_policy(steps=1),
            program_id="optimizer-schedule-warm-rejected",
            warm_start=parent.state,
        )


def test_zero_step_plan_has_only_evaluate_reduce_select() -> None:
    module, spec, objective, _split, program = _compile(steps=0)
    result = execute_native_alpha_beta_optimizer_program(
        program, module, spec, linear_spec_C=objective
    )

    assert tuple(task.kind for task in program.task_module.tasks) == (
        OptimizerTaskKind.EVALUATE_BOUND,
        OptimizerTaskKind.REDUCE_METRIC,
        OptimizerTaskKind.SELECT_BEST,
    )
    assert result.state.stable_hash() == program.initial_state.stable_hash()
    assert result.trace.best_iteration_by_domain == (0,)


def test_external_intermediate_semantics_and_adaptive_alpha_are_bound() -> None:
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0]])
    split = {"h1": torch.tensor([[0]], dtype=torch.int8)}
    external = {
        "h1": IntervalState(
            lower=torch.tensor([[-0.25]]),
            upper=torch.tensor([[0.75]]),
        )
    }
    policy = NativeAlphaBetaOptimizerPolicy(
        steps=0,
        lr=0.2,
        alpha_initialization_mode="adaptive",
    )
    program = compile_native_alpha_beta_optimizer_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=policy,
        program_id="optimizer-external-adaptive",
        relu_pre_override=external,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
    )

    assert torch.equal(program.relu_pre["h1"].lower, torch.tensor([[-0.25]]))
    assert torch.equal(program.initial_state.alphas["h1"], torch.tensor([[1.0]]))
    relu_ops = tuple(
        op
        for op in program.source_compilation.build.module.graph.ops
        if op.kind == BoundOpKind.RELU_RELAXATION
    )
    assert relu_ops
    assert all(
        isinstance(op.attrs, ReluRelaxationAttrs)
        and op.attrs.intermediate_bound_source
        == IntermediateBoundSource.EXTERNAL_VERIFIER
        for op in relu_ops
    )
    assert policy.to_dict()["alpha_initialization_mode"] == "adaptive"
    assert "alpha_initialization_mode" not in _policy().to_dict()


def test_external_intermediate_semantics_mismatch_fails_closed() -> None:
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0]])
    split = {"h1": torch.tensor([[0]], dtype=torch.int8)}
    policy = _policy(steps=0)
    with pytest.raises(ValueError, match="require ReLU bounds"):
        compile_native_alpha_beta_optimizer_program(
            module,
            spec,
            linear_spec_C=objective,
            relu_split_state=split,
            policy=policy,
            program_id="optimizer-external-missing",
            intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        )
    with pytest.raises(ValueError, match="requires external provenance"):
        compile_native_alpha_beta_optimizer_program(
            module,
            spec,
            linear_spec_C=objective,
            relu_split_state=split,
            policy=policy,
            program_id="optimizer-external-provenance",
            relu_pre_override={
                "h1": IntervalState(
                    lower=torch.tensor([[-0.25]]),
                    upper=torch.tensor([[0.75]]),
                )
            },
        )
    with pytest.raises(ValueError, match="schema differs"):
        compile_native_alpha_beta_optimizer_program(
            module,
            spec,
            linear_spec_C=objective,
            relu_split_state=split,
            policy=policy,
            program_id="optimizer-external-shape",
            relu_pre_override={
                "h1": IntervalState(
                    lower=torch.tensor([[-0.25, -0.25]]),
                    upper=torch.tensor([[0.75, 0.75]]),
                )
            },
            intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        )


def test_prepared_production_optimizer_matches_audit_without_hash_chain() -> None:
    module, spec, objective, _split, program = _compile()
    audit = execute_native_alpha_beta_optimizer_program(
        program, module, spec, linear_spec_C=objective
    )
    prepared = NativePreparedOptimizerProgram.prepare(
        program,
        module,
        spec,
        linear_spec_C=objective,
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
    )
    production = execute_prepared_native_alpha_beta_optimizer_program(
        prepared,
        module,
        spec,
        linear_spec_C=objective,
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
    )

    production.validate(prepared=prepared)
    assert torch.equal(production.bounds.lower, audit.bounds.lower)
    assert torch.equal(production.bounds.upper, audit.bounds.upper)
    assert production.state.stable_hash() == audit.state.stable_hash()
    assert production.best_iteration_by_domain == audit.trace.best_iteration_by_domain


def test_prepared_production_optimizer_rejects_semantic_drift() -> None:
    module, spec, objective, _split, program = _compile()
    prepared = NativePreparedOptimizerProgram.prepare(
        program,
        module,
        spec,
        linear_spec_C=objective,
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
    )
    with pytest.raises(ValueError, match="exact identity differs"):
        execute_prepared_native_alpha_beta_optimizer_program(
            prepared,
            module,
            spec,
            linear_spec_C=-objective,
            intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
        )
    with pytest.raises(ValueError, match="exact identity differs"):
        execute_prepared_native_alpha_beta_optimizer_program(
            prepared,
            module,
            spec,
            linear_spec_C=objective,
            intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        )
