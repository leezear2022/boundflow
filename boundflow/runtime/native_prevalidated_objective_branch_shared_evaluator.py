"""Bind compile-owned objective-branch scorer programs to shared nodes."""

# pylint: disable=too-many-arguments,protected-access,missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import torch

from ..ir.bound import IntermediateBoundSource
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_objective_branch_score import (
    NativeObjectiveBranchExecution,
    NativeObjectiveBranchPolicy,
    _repeat_box_input_spec,
)
from .native_prevalidated_objective_branch_score import (
    compile_native_prevalidated_objective_branch_program,
    execute_native_prevalidated_objective_branch_program,
)
from .native_shared_parametric_ancestral import _SharedEvaluatedNode
from .task_executor import InputSpec


def bind_prevalidated_objective_branches(
    evaluated: tuple[_SharedEvaluatedNode, ...],
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    batch_id: str,
) -> tuple[
    tuple[_SharedEvaluatedNode, ...],
    tuple[tuple[str, NativeObjectiveBranchExecution], ...],
]:
    """Compile one capsule per accepted node and bind its exact selection."""

    rebound: list[_SharedEvaluatedNode] = []
    executions: list[tuple[str, NativeObjectiveBranchExecution]] = []
    scalar_input = _repeat_box_input_spec(input_spec, count=1)
    for position, item in enumerate(evaluated):
        if item.evaluation.branch_candidate is None:
            rebound.append(item)
            continue
        program = compile_native_prevalidated_objective_branch_program(
            module,
            scalar_input,
            linear_spec_C=objective,
            relu_pre=item.relu_pre,
            selected_state=item.selected_state,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            refine_external_constraints=False,
            plan_id=f"{batch_id}:node:{position}:objective-branch-scorer-ownership",
        )
        execution = execute_native_prevalidated_objective_branch_program(
            program, node_id=item.runtime_node.node.node_id
        )
        execution.validate()
        rebound.append(
            replace(
                item,
                evaluation=replace(item.evaluation, branch_candidate=execution.branch),
            )
        )
        executions.append((item.runtime_node.node.node_id, execution))
    return tuple(rebound), tuple(executions)


__all__ = ["bind_prevalidated_objective_branches"]
