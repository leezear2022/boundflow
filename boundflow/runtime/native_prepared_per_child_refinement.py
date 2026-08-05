"""Prepared per-child refinement for the frozen optimized queue semantics."""

# pylint: disable=too-many-arguments,too-many-locals,protected-access
# pylint: disable=duplicate-code,missing-function-docstring

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..ir.refinement import (
    NativeIntermediateRefinementBudgetPolicyIR,
    NativeIntermediateRefinementMultiPassPolicyIR,
    NativeIntermediateRefinementPolicyIR,
)
from ..ir.task import BFTaskModule
from .native_intermediate_refinement import (
    NativeExternalIntermediateConstraintSeed,
    NativeIntermediateRefinementExecution,
)
from .native_optimized_relu_split_bab_runtime import (
    NativePerChildRefinementTrace,
    PerChildRefinementStrategy,
    _OptimizedEvaluatedNode,
    _batch_intermediate_bounds,
    _node_split_mapping,
    _refinement_budget_decisions,
    _refinement_semantic_trace_hash,
)
from .native_prepared_intermediate_refinement import (
    compile_native_prepared_intermediate_refinement_program,
    execute_native_prepared_intermediate_refinement_program,
)
from .native_relu_split_bab_runtime import _RuntimeNode, _repeat_box_input_spec
from .task_executor import InputSpec


def _execute_prepared_per_child_refinements(
    legacy_task_module: BFTaskModule,
    root_input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    policy: NativeIntermediateRefinementPolicyIR,
    budget_policy: Optional[NativeIntermediateRefinementBudgetPolicyIR],
    multi_pass_policy: Optional[NativeIntermediateRefinementMultiPassPolicyIR],
    budget_group_id: str,
    parent_by_id: Mapping[str, _OptimizedEvaluatedNode],
    strategy: PerChildRefinementStrategy,
    external_constraint_seed: Optional[NativeExternalIntermediateConstraintSeed],
) -> tuple[
    dict[str, IntervalState],
    tuple[tuple[str, NativeIntermediateRefinementExecution], ...],
    tuple[NativePerChildRefinementTrace, ...],
]:
    single_input = _repeat_box_input_spec(root_input_spec, count=1)
    executions: list[tuple[str, NativeIntermediateRefinementExecution]] = []
    records: list[NativePerChildRefinementTrace] = []
    budget_decisions = _refinement_budget_decisions(
        nodes,
        base_policy=policy,
        budget_policy=budget_policy,
        parent_by_id=parent_by_id,
        group_id=budget_group_id,
    )
    for node, budget_decision in zip(nodes, budget_decisions):
        split = _node_split_mapping(node)
        parent_id = node.node.parent_node_id
        source_execution: Optional[NativeIntermediateRefinementExecution] = None
        node_external_seed: Optional[NativeExternalIntermediateConstraintSeed] = None
        if (
            strategy
            in {
                "ancestral_constraint_carry_v1",
                "external_seeded_ancestral_carry_v1",
            }
            and parent_id is not None
        ):
            parent = parent_by_id.get(parent_id)
            if parent is None or parent.refinement_execution is None:
                raise ValueError("ancestral refinement child lacks a parent execution")
            source_execution = parent.refinement_execution
        elif strategy == "external_seeded_ancestral_carry_v1":
            if external_constraint_seed is None:
                raise ValueError("external-seeded refinement root lacks a seed")
            node_external_seed = external_constraint_seed
        program = compile_native_prepared_intermediate_refinement_program(
            legacy_task_module,
            single_input,
            policy=(
                policy
                if budget_decision is None
                else replace(
                    policy,
                    max_neurons_per_relu=(
                        budget_decision.assigned_max_neurons_per_relu
                    ),
                )
            ),
            plan_id=f"per-child-refinement:{node.node.split_state_hash}",
            multi_pass_policy=multi_pass_policy,
            relu_split_state=split,
            linear_spec_C=objective,
            source_refinement_execution=source_execution,
            external_constraint_seed=node_external_seed,
        )
        execution = execute_native_prepared_intermediate_refinement_program(
            program, legacy_task_module, single_input
        )
        hashes = program.hashes()
        record = NativePerChildRefinementTrace(
            node_id=node.node.node_id,
            node_split_state_hash=node.node.split_state_hash,
            refinement_plan_hash=hashes["refinement_plan_hash"],
            refinement_task_module_hash=hashes["refinement_task_module_hash"],
            refinement_schedule_hash=hashes["refinement_schedule_hash"],
            refinement_semantic_trace_hash=_refinement_semantic_trace_hash(execution),
            initial_intermediate_bounds_hash=(
                program.plan.initial_intermediate_bounds_hash
            ),
            final_intermediate_bounds_hash=(
                execution.trace.final_intermediate_bounds_hash
            ),
            selected_target_count=len(program.plan.targets),
            source_parent_node_id=(None if source_execution is None else parent_id),
            source_intermediate_constraints_hash=(
                program.plan.source_intermediate_constraints_hash
            ),
            source_refinement_plan_hash=program.plan.source_refinement_plan_hash,
            source_refinement_semantic_trace_hash=(
                program.plan.source_refinement_semantic_trace_hash
            ),
            source_consumption=(
                None if source_execution is None else "sound_constraint_only"
            ),
            external_constraint_seed_hash=(
                None
                if program.plan.external_constraint_seed is None
                else program.plan.external_constraint_seed.stable_hash()
            ),
            external_constraint_seed_constraints_hash=(
                None
                if program.plan.external_constraint_seed is None
                else program.plan.external_constraint_seed.bound_intermediate_constraints_hash
            ),
            external_semantics_owner=(
                None
                if program.plan.external_constraint_seed is None
                else program.plan.external_constraint_seed.semantics_owner
            ),
            external_seed_consumption=(
                None
                if program.plan.external_constraint_seed is None
                else program.plan.external_constraint_seed.consumption
            ),
            budget_decision=budget_decision,
            budget_policy=budget_policy,
            multi_pass_policy=multi_pass_policy,
            multi_pass_decisions=tuple(
                item.selection_decision
                for item in execution.trace.pass_traces
                if item.selection_decision is not None
            ),
        )
        record.validate()
        executions.append((node.node.node_id, execution))
        records.append(record)
    return (
        _batch_intermediate_bounds(
            tuple(execution.relu_pre for _node_id, execution in executions)
        ),
        tuple(executions),
        tuple(records),
    )


__all__ = ["_execute_prepared_per_child_refinements"]
