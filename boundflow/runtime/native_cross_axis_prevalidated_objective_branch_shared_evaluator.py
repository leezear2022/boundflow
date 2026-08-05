"""Bind shared evaluated nodes through one cross-axis scorer launch."""

# pylint: disable=too-many-arguments,too-many-locals,protected-access
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import torch

from ..ir.bound import IntermediateBoundSource
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_cross_axis_prevalidated_objective_branch import (
    NativeCrossAxisObjectiveBranchBinding,
    NativeCrossAxisPrevalidatedObjectiveBranchBatchExecution,
    compile_native_cross_axis_prevalidated_objective_branch_batch,
    execute_native_cross_axis_prevalidated_objective_branch_batch,
)
from .native_objective_branch_score import (
    NativeObjectiveBranchExecution,
    NativeObjectiveBranchPolicy,
    _repeat_box_input_spec,
)
from .native_prevalidated_objective_branch_score import (
    compile_native_prevalidated_objective_branch_program,
)
from .native_shared_parametric_ancestral import _SharedEvaluatedNode
from .task_executor import InputSpec


@dataclass(frozen=True)
class NativeCrossAxisSharedObjectiveBranchInput:
    """One shared node plus its exact clause objective owner."""

    clause_ordinal: int
    batch_position: int
    objective: torch.Tensor
    evaluated: _SharedEvaluatedNode

    def validate(self) -> None:
        if (
            self.clause_ordinal < 0
            or self.batch_position < 0
            or not torch.is_tensor(self.objective)
            or not torch.is_floating_point(self.objective)
            or self.evaluated.evaluation.branch_candidate is None
        ):
            raise ValueError("cross-axis shared objective branch input is invalid")


def bind_cross_axis_prevalidated_objective_branches(
    inputs: Tuple[NativeCrossAxisSharedObjectiveBranchInput, ...],
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    batch_id: str,
    max_child_domains: int = 512,
) -> tuple[
    Tuple[_SharedEvaluatedNode, ...],
    Tuple[tuple[str, NativeObjectiveBranchExecution], ...],
    NativeCrossAxisPrevalidatedObjectiveBranchBatchExecution,
]:
    """Compile node capsules, launch once, then restore the input order."""

    if not inputs or not batch_id:
        raise ValueError("cross-axis shared objective branch batch is empty")
    scalar_input = _repeat_box_input_spec(input_spec, count=1)
    bindings: list[NativeCrossAxisObjectiveBranchBinding] = []
    for item in inputs:
        item.validate()
        evaluated = item.evaluated
        program = compile_native_prevalidated_objective_branch_program(
            module,
            scalar_input,
            linear_spec_C=item.objective,
            relu_pre=evaluated.relu_pre,
            selected_state=evaluated.selected_state,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            refine_external_constraints=False,
            plan_id=(
                f"{batch_id}:clause:{item.clause_ordinal}:"
                f"position:{item.batch_position}:objective-branch-cross-axis"
            ),
        )
        bindings.append(
            NativeCrossAxisObjectiveBranchBinding(
                clause_ordinal=item.clause_ordinal,
                node_id=evaluated.runtime_node.node.node_id,
                program=program,
            )
        )
    batch_program = compile_native_cross_axis_prevalidated_objective_branch_batch(
        tuple(bindings),
        batch_id=f"{batch_id}:cross-axis-scorer",
        max_child_domains=max_child_domains,
    )
    batch_execution = execute_native_cross_axis_prevalidated_objective_branch_batch(
        batch_program
    )
    rebound = tuple(
        replace(
            item.evaluated,
            evaluation=replace(
                item.evaluated.evaluation,
                branch_candidate=execution.branch,
            ),
        )
        for item, execution in zip(inputs, batch_execution.executions)
    )
    executions = tuple(
        (item.evaluated.runtime_node.node.node_id, execution)
        for item, execution in zip(inputs, batch_execution.executions)
    )
    return rebound, executions, batch_execution


__all__ = [
    "NativeCrossAxisSharedObjectiveBranchInput",
    "bind_cross_axis_prevalidated_objective_branches",
]
