"""Runtime integration tests for typed unresolved-clause escalation."""

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
)
from boundflow.runtime.native_hard_clause_escalation import (
    compile_native_hard_clause_escalation_program,
    execute_native_hard_clause_escalation_program,
)
from boundflow.runtime.task_executor import InputSpec


def _fixture() -> tuple[BFTaskModule, InputSpec]:
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="hard-clause-toy",
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
        entry_task_id="hard-clause-toy",
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


def _policies():
    return (
        NativeProjectedGradientSearchPolicy(steps=1, step_size=0.01),
        NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1),
    )


def _execute(threshold: float):
    module, spec = _fixture()
    search_policy, optimizer_policy = _policies()
    objective = torch.tensor([[[1.0, -1.0]]])
    thresholds = torch.tensor([threshold])
    program = compile_native_hard_clause_escalation_program(
        module,
        spec,
        linear_spec_C=objective,
        thresholds=thresholds,
        plan_id=f"hard-clause-toy:{threshold}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    execution = execute_native_hard_clause_escalation_program(
        program,
        module,
        spec,
        linear_spec_C=objective,
        thresholds=thresholds,
        query_id=f"hard-clause-toy:{threshold}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    execution.validate_against(
        module,
        spec,
        linear_spec_C=objective,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    return execution


def test_hard_clause_escalation_skips_guarded_stage_when_baseline_closes() -> None:
    execution = _execute(-1e6)

    assert execution.trace.final_status == "verified"
    assert execution.trace.decision.escalated_clause_indices == ()
    assert [item.executed for item in execution.trace.actions[2:6]] == [False] * 4
    assert execution.trace.fallback_reason == "no_escalation_needed"


def test_hard_clause_escalation_executes_refined_projected_query() -> None:
    execution = _execute(-0.1)

    assert execution.trace.decision.escalated_clause_indices == (0,)
    assert execution.refinement is not None
    assert execution.escalation is not None
    assert all(item.executed for item in execution.trace.actions[2:6])
    assert execution.trace.escalation_original_clause_indices == (0,)
