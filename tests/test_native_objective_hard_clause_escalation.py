"""Runtime tests for shared-source objective-directed escalation."""

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
)
from boundflow.runtime.native_intermediate_refinement import (
    intermediate_refinement_semantic_trace_hash,
)
from boundflow.runtime.native_objective_hard_clause_escalation import (
    compile_native_objective_hard_clause_escalation_program,
    execute_native_objective_hard_clause_escalation_program,
)
from boundflow.runtime.task_executor import InputSpec


def _fixture() -> tuple[BFTaskModule, InputSpec]:
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="objective-hard-clause-toy",
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
        entry_task_id="objective-hard-clause-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [-0.25, 0.75]]),
                "b1": torch.tensor([0.1, -0.2]),
                "W2": torch.tensor([[0.75, -1.0], [-0.5, 0.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )
    return module, spec


def _execute(thresholds: torch.Tensor):
    module, spec = _fixture()
    search = NativeProjectedGradientSearchPolicy(steps=1, step_size=0.01)
    optimizer = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    objectives = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    program = compile_native_objective_hard_clause_escalation_program(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id="objective-hard-clause-toy",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    execution = execute_native_objective_hard_clause_escalation_program(
        program,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="objective-hard-clause-toy",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    execution.validate_against(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        search_policy=search,
        optimizer_policy=optimizer,
    )
    return module, spec, objectives, execution


def test_objective_escalation_binds_each_child_to_shared_execution() -> None:
    _module, _spec, objectives, execution = _execute(torch.tensor([-0.1, -1e6]))

    assert execution.shared_refinement is not None
    admitted = execution.trace.decision.escalated_clause_indices
    assert admitted
    assert (
        tuple(child.original_clause_index for child in execution.clause_executions)
        == admitted
    )
    source_hash = execution.shared_refinement.program.plan.stable_hash()
    source_trace_hash = intermediate_refinement_semantic_trace_hash(
        execution.shared_refinement
    )
    for child in execution.clause_executions:
        ordinal = child.original_clause_index
        assert child.refinement_program.plan.objective_hash is not None
        assert child.refinement_program.plan.source_refinement_plan_hash == source_hash
        assert (
            child.refinement_program.plan.source_refinement_semantic_trace_hash
            == source_trace_hash
        )
        assert (
            child.query.trace.objective_matrix_hash
            == child.refinement_program.plan.objective_hash
        )
        assert child.refinement_program.objective is not None
        assert torch.equal(
            child.refinement_program.objective,
            objectives[:, ordinal : ordinal + 1, :],
        )
    assert set(execution.trace.decision.baseline_verified_clause_indices) <= set(
        execution.trace.final_verified_clause_indices
    )


def test_objective_escalation_skips_all_clause_tasks_after_baseline_closure() -> None:
    _module, _spec, _objectives, execution = _execute(torch.tensor([-1e6, -1e6]))

    assert execution.trace.final_status == "verified"
    assert execution.trace.decision.escalated_clause_indices == ()
    assert execution.shared_refinement is None
    assert execution.clause_executions == ()
    assert all(not action.executed for action in execution.trace.actions[2:-2])


def test_objective_escalation_aggregate_tamper_fails_closed() -> None:
    _module, _spec, _objectives, execution = _execute(torch.tensor([-0.1, -1e6]))
    tampered = replace(
        execution.trace,
        final_verified_clause_indices=tuple(range(execution.program.plan.clause_count)),
    )

    with pytest.raises(ValueError, match="aggregate trace|signature"):
        tampered.validate_against(execution.program)
