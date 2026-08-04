"""Objective-aware ReLU branch Plan/Task/Schedule and runtime tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.domains.interval import IntervalState
from boundflow.ir.bound import IntermediateBoundSource
from boundflow.ir.branch import ObjectiveBranchTaskKind
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_alpha_beta_optimizer_schedule import (
    compile_native_alpha_beta_optimizer_program,
    execute_native_alpha_beta_optimizer_program,
)
from boundflow.runtime.native_objective_branch_score import (
    NativeObjectiveBranchPolicy,
    compile_native_objective_branch_program,
    execute_native_objective_branch_program,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    execute_native_optimized_relu_split_bab,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="objective-branch-toy",
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
        entry_task_id="objective-branch-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [-0.25, 0.75], [0.5, 0.5]]),
                "b1": torch.tensor([0.0, -0.1, 0.05]),
                "W2": torch.tensor([[0.75, -1.0, 0.5]]),
                "b2": torch.tensor([0.1]),
            }
        },
    )


def _compile():
    module = _module()
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.5, -0.4]]),
        upper=torch.tensor([[0.6, 0.7]]),
    )
    objective = torch.tensor([[1.0]])
    optimizer_policy = NativeAlphaBetaOptimizerPolicy(
        steps=1,
        lr=0.1,
        alpha_initialization_mode="adaptive",
    )
    split = {"h1": torch.zeros((1, 3), dtype=torch.int8)}
    optimizer = compile_native_alpha_beta_optimizer_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=optimizer_policy,
        program_id="objective-branch-toy:optimizer",
    )
    selected = execute_native_alpha_beta_optimizer_program(
        optimizer, module, spec, linear_spec_C=objective
    )
    branch_policy = NativeObjectiveBranchPolicy(
        candidates_per_relu=3,
        candidate_batch_size=2,
    )
    branch = compile_native_objective_branch_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=optimizer.relu_pre,
        selected_state=selected.state,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
        plan_id="objective-branch-toy:score",
    )
    return module, spec, objective, optimizer_policy, branch_policy, branch


def test_plan_task_schedule_drive_exact_objective_branch_score() -> None:
    _module_value, _spec, _objective, _optimizer_policy, _branch_policy, program = (
        _compile()
    )
    execution = execute_native_objective_branch_program(program, node_id="toy:n0")

    assert tuple(task.kind for task in program.task_module.tasks) == tuple(
        ObjectiveBranchTaskKind
    )
    assert len(program.schedule.actions) == 5
    assert len(execution.trace.scores) == len(program.plan.candidates) == 3
    selected = execution.trace.scores[execution.trace.selected_candidate_ordinal]
    assert selected.worst_child_lower == max(
        score.worst_child_lower for score in execution.trace.scores
    )
    assert execution.branch.relu_input == "h1"
    execution.validate()


def test_objective_branch_score_is_deterministic_and_hash_bound() -> None:
    _module_value, _spec, _objective, _optimizer_policy, _branch_policy, program = (
        _compile()
    )
    left = execute_native_objective_branch_program(program, node_id="toy:n0")
    right = execute_native_objective_branch_program(program, node_id="toy:n0")

    assert left.branch == right.branch
    assert left.trace.stable_hash(program=program) == right.trace.stable_hash(
        program=program
    )
    assert left.trace.child_lower_hash == right.trace.child_lower_hash


def test_program_and_trace_tamper_fail_closed() -> None:
    _module_value, _spec, _objective, _optimizer_policy, _branch_policy, program = (
        _compile()
    )
    execution = execute_native_objective_branch_program(program, node_id="toy:n0")

    with pytest.raises(ValueError, match="program identity"):
        replace(
            program,
            branch_policy=replace(program.branch_policy, candidates_per_relu=2),
        ).validate()
    with pytest.raises(ValueError, match="trace"):
        replace(
            execution.trace,
            selected_candidate_ordinal=(execution.trace.selected_candidate_ordinal + 1)
            % len(execution.trace.scores),
        ).validate(program=program)
    scores = list(execution.trace.scores)
    scores[0] = replace(scores[0], inactive_lower=scores[0].inactive_lower + 0.1)
    with pytest.raises(ValueError, match="score"):
        replace(execution.trace, scores=tuple(scores)).validate(program=program)


def test_candidate_cap_and_semantic_scope_fail_closed() -> None:
    module, spec, objective, optimizer_policy, _branch_policy, program = _compile()
    with pytest.raises(ValueError, match="safety cap"):
        compile_native_objective_branch_program(
            module,
            spec,
            linear_spec_C=objective,
            relu_pre=program.relu_pre,
            selected_state=program.selected_state,
            optimizer_policy=optimizer_policy,
            branch_policy=NativeObjectiveBranchPolicy(
                candidates_per_relu=3,
                max_candidates=2,
            ),
            intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
            plan_id="objective-branch-toy:capped",
        )
    with pytest.raises(ValueError, match="program identity"):
        replace(program, objective=objective + 1.0).validate()


def test_objective_branch_policy_drives_queue_and_is_not_erasable() -> None:
    module = _module()
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.5, -0.4]]),
        upper=torch.tensor([[0.6, 0.7]]),
    )
    branch_policy = NativeObjectiveBranchPolicy(
        candidates_per_relu=3,
        candidate_batch_size=2,
    )
    execution = execute_native_optimized_relu_split_bab(
        module,
        spec,
        linear_spec_C=torch.tensor([[1.0]]),
        run_id="objective-branch-queue-toy",
        config=NativeReluSplitBabConfig(
            max_nodes=3,
            max_depth=1,
            expansion_batch_size=1,
            max_eval_batch_size=2,
            threshold=1e6,
        ),
        optimizer_policy=NativeAlphaBetaOptimizerPolicy(
            steps=1,
            lr=0.1,
            alpha_initialization_mode="adaptive",
        ),
        objective_branch_policy=branch_policy,
    )

    assert execution.objective_branch_policy == branch_policy
    assert execution.trace.decisions[0].reason == "objective_bound_impact"
    assert len(execution.objective_branch_executions) == 3
    execution.validate()
    with pytest.raises(ValueError, match="coverage"):
        replace(execution, objective_branch_executions=()).validate()


def test_external_constraint_refinement_is_explicit_and_propagates_forward() -> None:
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="refined-external-toy",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["h2"]),
                    TaskOp("relu", "relu2", ["h2"], ["r2"]),
                    TaskOp("linear", "linear3", ["r2", "W3", "b3"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="refined-external-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0]]),
                "b1": torch.tensor([0.0]),
                "W2": torch.tensor([[1.0]]),
                "b2": torch.tensor([-0.5]),
                "W3": torch.tensor([[1.0]]),
                "b3": torch.tensor([0.0]),
            }
        },
    )
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[0.0]]), eps=1.0)
    objective = torch.tensor([[1.0]])
    split = {
        "h1": torch.zeros((1, 1), dtype=torch.int8),
        "h2": torch.zeros((1, 1), dtype=torch.int8),
    }
    external = {
        "h1": IntervalState(lower=torch.tensor([[-0.2]]), upper=torch.tensor([[0.2]])),
        "h2": IntervalState(lower=torch.tensor([[-0.5]]), upper=torch.tensor([[0.5]])),
    }
    policy = NativeAlphaBetaOptimizerPolicy(steps=0, lr=0.1)
    default = compile_native_alpha_beta_optimizer_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=policy,
        program_id="external-default-toy",
        relu_pre_override=external,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
    )
    refined = compile_native_alpha_beta_optimizer_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=policy,
        program_id="external-refined-toy",
        relu_pre_override=external,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        refine_external_constraints=True,
    )

    assert torch.equal(default.relu_pre["h2"].upper, torch.tensor([[0.5]]))
    assert torch.allclose(refined.relu_pre["h2"].upper, torch.tensor([[-0.3]]))
    assert default.initial_state.scope != refined.initial_state.scope
    with pytest.raises(ValueError, match="requires external provenance"):
        compile_native_alpha_beta_optimizer_program(
            module,
            spec,
            linear_spec_C=objective,
            relu_split_state=split,
            policy=policy,
            program_id="external-refined-owner-mismatch",
            refine_external_constraints=True,
        )
