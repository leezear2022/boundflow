"""Contracts for the NRIR-39 objective-branch shared evaluator."""

# pylint: disable=missing-function-docstring,duplicate-code,redefined-outer-name

from dataclasses import replace

import pytest
import torch

from boundflow.ir.objective_branch_shared_evaluator import (
    NativeObjectiveBranchSharedScheduleIR,
    NativeObjectiveBranchSharedTaskKind,
)
from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
)
from boundflow.runtime.native_intermediate_refinement import (
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
)
from boundflow.runtime.native_objective_branch_score import (
    NativeObjectiveBranchPolicy,
)
from boundflow.runtime.native_objective_branch_shared_evaluator import (
    compile_native_objective_branch_shared_plan,
    execute_native_objective_branch_shared_queue,
)
from boundflow.runtime.native_objective_branch_shared_production_queue import (
    execute_native_objective_branch_shared_production_queue,
)
from boundflow.runtime.native_objective_branch_shared_multi_clause_anytime import (
    execute_native_objective_branch_shared_multi_clause_anytime_program,
)
from boundflow.runtime.native_multi_clause_anytime import (
    compile_native_multi_clause_anytime_program,
)
from boundflow.runtime.native_parametric_optimizer import (
    NativeParametricOptimizerTemplateCache,
)
from boundflow.runtime.native_shared_parametric_ancestral import (
    execute_native_shared_parametric_ancestral_queue,
)
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="objective-branch-shared-toy",
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
        entry_task_id="objective-branch-shared-toy",
        bindings={
            "params": {
                "W1": torch.tensor(
                    [
                        [1.0, -0.5],
                        [-0.25, 0.75],
                        [0.6, 0.8],
                        [-0.7, -0.4],
                        [0.9, 0.3],
                        [-0.4, 0.9],
                        [0.5, -0.8],
                        [-0.9, 0.2],
                    ]
                ),
                "b1": torch.zeros(8),
                "W2": torch.tensor(
                    [
                        [0.75, -1.0, 0.4, -0.3, 0.5, -0.6, 0.8, -0.2],
                        [-0.5, 0.25, -0.6, 0.9, -0.7, 0.4, -0.3, 0.65],
                    ]
                ),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )


def _root_refinement(module: BFTaskModule, spec: InputSpec, objective: torch.Tensor):
    shared_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1, max_neurons_per_relu=4, backward_chunk_size=4
        ),
        plan_id="objective-branch-shared:shared",
    )
    shared = execute_native_intermediate_refinement_program(
        shared_program, module, spec
    )
    root_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=4,
            backward_chunk_size=4,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
        plan_id="objective-branch-shared:root",
        linear_spec_C=objective,
        source_refinement_execution=shared,
    )
    return execute_native_intermediate_refinement_program(root_program, module, spec)


@pytest.fixture(scope="module")
def execution_bundle():
    module = _module()
    spec = _spec()
    objective = torch.tensor([[[1.0, -1.0]]])
    threshold = torch.tensor([1e6])
    root = _root_refinement(module, spec, objective)
    optimizer_policy = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    branch_policy = NativeObjectiveBranchPolicy()
    plan = compile_native_objective_branch_shared_plan(
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        plan_id="objective-branch-shared-toy",
    )
    control = execute_native_shared_parametric_ancestral_queue(
        plan.shared_plan,
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        compiler_cache=NativeParametricOptimizerTemplateCache(),
        query_id="objective-branch-shared-control",
        clock_ns=lambda: 0,
    )
    candidate = execute_native_objective_branch_shared_queue(
        plan,
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        compiler_cache=NativeParametricOptimizerTemplateCache(),
        control_execution=control,
        query_id="objective-branch-shared-candidate",
        clock_ns=lambda: 0,
    )
    return (
        module,
        spec,
        objective,
        threshold,
        root,
        optimizer_policy,
        branch_policy,
        control,
        candidate,
    )


def test_objective_branch_shared_owns_six_stage_composite_ir(
    execution_bundle,
) -> None:
    *_unused, candidate = execution_bundle

    assert [task.kind for task in candidate.task_ir.tasks] == list(
        NativeObjectiveBranchSharedTaskKind
    )
    assert len(candidate.task_ir.tasks) == 6
    assert len(candidate.shared_execution.queue.trace.evaluations) == 31
    assert len(candidate.branch_bindings) == len(
        candidate.shared_execution.queue.objective_branch_executions
    )
    assert len(candidate.branch_bindings) == 31
    assert candidate.decision.structure_passed is True
    candidate.schedule.validate_against(candidate.task_ir)


def test_objective_branch_shared_runtime_binds_selected_candidate(
    execution_bundle,
) -> None:
    (
        module,
        spec,
        objective,
        threshold,
        root,
        optimizer_policy,
        branch_policy,
        control,
        candidate,
    ) = execution_bundle

    evaluations = {
        item.node.node_id: item
        for item in candidate.shared_execution.queue.trace.evaluations
    }
    branches = dict(candidate.shared_execution.queue.objective_branch_executions)
    for binding in candidate.branch_bindings:
        assert (
            evaluations[binding.node_id].branch_candidate
            == branches[binding.node_id].branch
        )
        assert (
            binding.selected_candidate_ordinal
            == branches[binding.node_id].trace.selected_candidate_ordinal
        )
    candidate.validate_against(
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        control_execution=control,
    )


def test_objective_branch_shared_rejects_policy_and_binding_tamper(
    execution_bundle,
) -> None:
    (
        module,
        spec,
        objective,
        threshold,
        root,
        optimizer_policy,
        branch_policy,
        control,
        candidate,
    ) = execution_bundle

    with pytest.raises(ValueError, match="Plan IR differs"):
        replace(candidate.plan, candidates_per_relu=7).validate()
    tampered = replace(candidate.branch_bindings[0], selected_neuron_index=999)
    with pytest.raises(ValueError, match="evidence differs"):
        replace(
            candidate,
            branch_bindings=(tampered, *candidate.branch_bindings[1:]),
        ).validate_against(
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            control_execution=control,
        )


def test_objective_branch_shared_rejects_schedule_task_drift(
    execution_bundle,
) -> None:
    *_unused, candidate = execution_bundle
    action = replace(candidate.schedule.actions[2], task_hash="0" * 64)
    schedule = NativeObjectiveBranchSharedScheduleIR(
        plan_hash=candidate.schedule.plan_hash,
        task_ir_hash=candidate.schedule.task_ir_hash,
        actions=(
            *candidate.schedule.actions[:2],
            action,
            *candidate.schedule.actions[3:],
        ),
    )
    with pytest.raises(ValueError, match="Schedule/Task binding differs"):
        schedule.validate_against(candidate.task_ir)


def test_objective_branch_shared_production_needs_no_widest_control(
    execution_bundle,
) -> None:
    (
        module,
        spec,
        objective,
        threshold,
        root,
        optimizer_policy,
        branch_policy,
        _control,
        candidate,
    ) = execution_bundle

    production = execute_native_objective_branch_shared_production_queue(
        candidate.plan,
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        compiler_cache=NativeParametricOptimizerTemplateCache(),
        query_id="objective-branch-shared-production",
        clock_ns=lambda: 0,
    )

    assert len(production.queue.trace.evaluations) == 31
    assert len(production.queue.objective_branch_executions) == 31
    assert production.queue.objective_branch_policy == branch_policy
    assert all(
        decision.reason != "widest_unsplit_ambiguous_relu"
        for decision in production.queue.trace.decisions
    )


def test_objective_branch_multi_clause_preserves_floor_only_result() -> None:
    module = _module()
    spec = _spec()
    objectives = torch.tensor([[[1.0, -1.0]]]).repeat(1, 9, 1)
    thresholds = torch.full((9,), -1e6)
    search = NativeProjectedGradientSearchPolicy(steps=1, step_size=0.01)
    optimizer = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    branch = NativeObjectiveBranchPolicy()
    program = compile_native_multi_clause_anytime_program(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id="objective-branch-multi-clause-floor-only",
        search_policy=search,
        optimizer_policy=optimizer,
    )

    execution = execute_native_objective_branch_shared_multi_clause_anytime_program(
        program,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="objective-branch-multi-clause-floor-only",
        search_policy=search,
        optimizer_policy=optimizer,
        branch_policy=branch,
    )

    assert execution.floor.trace.final_status == "verified"
    assert not execution.packed_executions
    assert not execution.cache_events
    assert execution.aggregate.final_status == "verified"
