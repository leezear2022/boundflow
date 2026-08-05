"""Runtime tests for the NRIR44 ranking-floor root projection."""

# pylint: disable=missing-function-docstring

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
from boundflow.runtime.native_objective_hard_clause_escalation import (
    execute_native_objective_hard_clause_escalation_program,
)
from boundflow.runtime.native_root_projection_floor import (
    compile_native_root_projection_floor_program,
    execute_native_root_projection_floor_program,
)
from boundflow.runtime.task_executor import InputSpec


def _fixture() -> tuple[BFTaskModule, InputSpec]:
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="root-projection-toy",
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
        entry_task_id="root-projection-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [-0.25, 0.75]]),
                "b1": torch.tensor([0.1, -0.2]),
                "W2": torch.tensor([[0.75, -1.0], [-0.5, 0.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )
    return module, InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )


def _inputs():
    module, spec = _fixture()
    objectives = torch.tensor([[[1.0, -1.0]]]).repeat(1, 9, 1)
    thresholds = torch.full((9,), -0.1)
    search = NativeProjectedGradientSearchPolicy(steps=1, step_size=0.01)
    optimizer = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    return module, spec, objectives, thresholds, search, optimizer


def test_root_projection_matches_full_floor_roots_and_executes_nine_evals() -> None:
    module, spec, objectives, thresholds, search, optimizer = _inputs()
    program = compile_native_root_projection_floor_program(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id="root-projection-toy",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    projected = execute_native_root_projection_floor_program(
        program,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="root-projection-toy:projected",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    full = execute_native_objective_hard_clause_escalation_program(
        program.source_program,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="root-projection-toy:full",
        search_policy=search,
        optimizer_policy=optimizer,
    )

    assert projected.program.schedule.full_evaluation_budget == 279
    assert projected.program.schedule.projected_evaluation_budget == 9
    assert len(projected.projection_trace.clause_traces) == 9
    for projected_child, full_child in zip(
        projected.source_execution.clause_executions,
        full.clause_executions,
    ):
        projected_queue = projected_child.query.clauses[0].queue.trace
        full_queue = full_child.query.clauses[0].queue.trace
        projected_root = projected_queue.evaluations[0]
        full_root = full_queue.evaluations[0]
        assert len(projected_queue.evaluations) == 1
        assert projected_root.lower == full_root.lower
        assert projected_root.upper == full_root.upper
        assert projected_root.branch_candidate == full_root.branch_candidate


def test_root_projection_trace_tamper_fails_closed() -> None:
    module, spec, objectives, thresholds, search, optimizer = _inputs()
    program = compile_native_root_projection_floor_program(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id="root-projection-tamper",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    execution = execute_native_root_projection_floor_program(
        program,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="root-projection-tamper",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    tampered = replace(execution.projection_trace, final_status="verified")

    with pytest.raises(ValueError, match="execution/Trace differs"):
        replace(execution, projection_trace=tampered).validate_against(
            module,
            spec,
            linear_spec_C=objectives,
            thresholds=thresholds,
            search_policy=search,
            optimizer_policy=optimizer,
        )
