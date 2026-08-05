"""Prepared-refinement evaluator for shared-parametric sibling batches."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=protected-access,duplicate-code,missing-function-docstring
# pylint: disable=too-many-branches

from __future__ import annotations

from dataclasses import replace
import time
from typing import Mapping, Optional

import torch

from ..ir.bound import IntermediateBoundSource
from ..ir.production_verifier import (
    NativeProductionVerifierTaskKind,
    lower_native_production_verifier_ir,
)
from ..ir.refinement import NativeIntermediateRefinementPolicyIR
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_intermediate_refinement import NativeIntermediateRefinementExecution
from .native_optimized_relu_split_bab_runtime import (
    _batched_split_state,
    _build_batched_parent_warm_state,
    _repeat_relu_pre_override,
)
from .native_parametric_optimizer import (
    NativeParametricOptimizerTemplateCache,
    execute_native_parametric_optimizer,
    instantiate_native_parametric_optimizer,
)
from .native_parametric_production_verifier import (
    NativeParametricCompilerBatchTrace,
    _parametric_production_plan,
    _slice_parametric_state,
)
from .native_prepared_per_child_refinement import (
    _execute_prepared_per_child_refinements,
)
from .native_production_verifier import (
    NativeProductionBabEvaluation,
    NativeProductionVerifierActionTrace,
    NativeProductionVerifierBatchTrace,
)
from .native_relu_split_bab_runtime import (
    _RuntimeNode,
    _priority,
    _repeat_box_input_spec,
    _select_branch,
    _slice_interval,
)
from .native_shared_parametric_ancestral import _SharedEvaluatedNode
from .task_executor import InputSpec


def _evaluate_prepared_shared_parametric_batch(  # pylint: disable=too-many-statements
    module: BFTaskModule,
    root_input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    batch_id: str,
    policy: NativeAlphaBetaOptimizerPolicy,
    parent_by_id: Mapping[str, _SharedEvaluatedNode],
    root_refinement: Optional[NativeIntermediateRefinementExecution],
    child_refinement_policy: NativeIntermediateRefinementPolicyIR,
    compiler_cache: NativeParametricOptimizerTemplateCache,
) -> tuple[
    tuple[_SharedEvaluatedNode, ...],
    NativeProductionVerifierBatchTrace,
    NativeParametricCompilerBatchTrace,
    tuple[tuple[str, NativeIntermediateRefinementExecution], ...],
]:
    if not nodes:
        raise ValueError("shared-parametric batch cannot be empty")
    is_root = nodes[0].node.depth == 0
    if any((node.node.depth == 0) != is_root for node in nodes):
        raise ValueError("shared-parametric batch mixes root and child nodes")
    batch_input = _repeat_box_input_spec(root_input_spec, count=len(nodes))
    split_batch = _batched_split_state(nodes)
    refinement_executions: tuple[tuple[str, NativeIntermediateRefinementExecution], ...]
    if is_root:
        if root_refinement is None or len(nodes) != 1:
            raise ValueError("shared-parametric root refinement differs")
        refinement_executions = ((nodes[0].node.node_id, root_refinement),)
        batch_relu_pre = _repeat_relu_pre_override(
            root_refinement.relu_pre, count=len(nodes)
        )
        warm_state = None
    else:
        if root_refinement is not None:
            raise ValueError("shared-parametric child received root refinement")
        batch_relu_pre, refinement_executions, _records = (
            _execute_prepared_per_child_refinements(
                module,
                root_input_spec,
                objective=objective,
                nodes=nodes,
                policy=child_refinement_policy,
                budget_policy=None,
                multi_pass_policy=None,
                budget_group_id=batch_id,
                parent_by_id=parent_by_id,  # type: ignore[arg-type]
                strategy="ancestral_constraint_carry_v1",
                external_constraint_seed=None,
            )
        )
        warm_state = _build_batched_parent_warm_state(
            module,
            batch_input,
            objective=objective,
            nodes=nodes,
            policy=policy,
            parent_by_id=parent_by_id,  # type: ignore[arg-type]
            relu_pre_override=batch_relu_pre,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            refine_external_constraints=False,
            use_parent_runtime_bounds=True,
        )
    acquire_started_ns = time.perf_counter_ns()
    template, cache_event = compiler_cache.acquire(
        module,
        batch_input,
        linear_spec_C=objective,
        relu_pre=batch_relu_pre,
        policy=policy,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        refine_external_constraints=False,
        template_id=f"{batch_id}:optimizer-template",
        batch_id=batch_id,
    )
    acquire_elapsed_ns = time.perf_counter_ns() - acquire_started_ns
    instantiate_started_ns = time.perf_counter_ns()
    instance = instantiate_native_parametric_optimizer(
        template,
        module,
        batch_input,
        linear_spec_C=objective,
        relu_split_state=split_batch,
        instance_id=f"{batch_id}:optimizer-instance",
        warm_start=warm_state,
        relu_pre_override=batch_relu_pre,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        refine_external_constraints=False,
    )
    instantiate_elapsed_ns = time.perf_counter_ns() - instantiate_started_ns
    production_plan = _parametric_production_plan(
        batch_id=batch_id,
        nodes=nodes,
        parent_by_id=parent_by_id,  # type: ignore[arg-type]
        instance=instance,
    )
    task_ir, schedule = lower_native_production_verifier_ir(production_plan)
    selected = None
    materialized: Optional[tuple[_SharedEvaluatedNode, ...]] = None
    action_traces: list[NativeProductionVerifierActionTrace] = []
    execute_elapsed_ns = 0
    refinement_by_id = dict(refinement_executions)
    for action in schedule.actions:
        started_ns = time.perf_counter_ns()
        if action.kind == NativeProductionVerifierTaskKind.VALIDATE_PROGRAM:
            instance.require_exact_runtime(
                module,
                batch_input,
                linear_spec_C=objective,
                intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            )
        elif action.kind == NativeProductionVerifierTaskKind.EXECUTE_OPTIMIZER:
            selected = execute_native_parametric_optimizer(
                instance,
                module,
                batch_input,
                linear_spec_C=objective,
                intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            )
        elif action.kind == NativeProductionVerifierTaskKind.MATERIALIZE_NODE_RESULTS:
            if selected is None:
                raise ValueError("shared-parametric materializes before execution")
            if tuple(selected.bounds.lower.shape) != (len(nodes), 1):
                raise ValueError("shared-parametric batch must return scalar bounds")
            values: list[_SharedEvaluatedNode] = []
            for index, runtime_node in enumerate(nodes):
                node_pre = {
                    name: _slice_interval(value, index=index)
                    for name, value in instance.relu_pre.items()
                }
                node_split = {
                    name: value[index : index + 1].contiguous()
                    for name, value in split_batch.items()
                }
                state = _slice_parametric_state(
                    instance,
                    root_input_spec,
                    module=module,
                    objective=objective,
                    selected_state=selected.state,
                    index=index,
                )
                parent = (
                    None
                    if runtime_node.node.parent_node_id is None
                    else parent_by_id.get(runtime_node.node.parent_node_id)
                )
                if runtime_node.node.depth > 0 and parent is None:
                    raise ValueError("shared-parametric node lacks a parent")
                branch = _select_branch(node_pre, relu_split_state=node_split)
                lower = float(selected.bounds.lower[index, 0].item())
                upper = float(selected.bounds.upper[index, 0].item())
                values.append(
                    _SharedEvaluatedNode(
                        runtime_node=runtime_node,
                        evaluation=NativeProductionBabEvaluation(
                            node=runtime_node.node,
                            lower=lower,
                            upper=upper,
                            priority=_priority(lower),
                            selected_state_hash=state.stable_hash(),
                            parent_selected_state_hash=(
                                None
                                if parent is None
                                else parent.selected_state.stable_hash()
                            ),
                            warm_start_kind=instance.ir.warm_start_kind,
                            eval_batch_id=batch_id,
                            eval_batch_position=index,
                            batch_trace_hash="0" * 64,
                            branch_candidate=branch,
                        ),
                        selected_state=state,
                        relu_pre=node_pre,
                        refinement_execution=refinement_by_id[
                            runtime_node.node.node_id
                        ],
                    )
                )
            materialized = tuple(values)
        elif action.kind == NativeProductionVerifierTaskKind.COMMIT_QUEUE_RESULTS:
            if materialized is None:
                raise ValueError("shared-parametric commits before materialization")
        else:
            raise AssertionError("unreachable production verifier action")
        elapsed_ns = time.perf_counter_ns() - started_ns
        if action.kind == NativeProductionVerifierTaskKind.EXECUTE_OPTIMIZER:
            execute_elapsed_ns = elapsed_ns
        action_traces.append(
            NativeProductionVerifierActionTrace(
                sequence=action.sequence,
                action_id=action.action_id,
                task_id=action.task_id,
                kind=action.kind,
                elapsed_ns=elapsed_ns,
            )
        )
    if selected is None or materialized is None:
        raise ValueError("shared-parametric Schedule produced no result")
    batch_trace = NativeProductionVerifierBatchTrace(
        plan=production_plan,
        task_ir=task_ir,
        schedule=schedule,
        actions=tuple(action_traces),
        selected_batch_state_hash=selected.state.stable_hash(),
    )
    batch_trace.validate()
    trace_hash = batch_trace.stable_hash()
    rebound = tuple(
        replace(
            item,
            evaluation=replace(item.evaluation, batch_trace_hash=trace_hash),
        )
        for item in materialized
    )
    for item in rebound:
        item.evaluation.validate()
    compiler = NativeParametricCompilerBatchTrace(
        batch_id=batch_id,
        cache_event=cache_event,
        instance_ir=instance.ir,
        template_hash=template.template_hash,
        task_hash=template.task_hash,
        schedule_hash=template.schedule_hash,
        acquire_elapsed_ns=acquire_elapsed_ns,
        instantiate_elapsed_ns=instantiate_elapsed_ns,
        execute_elapsed_ns=execute_elapsed_ns,
    )
    compiler.validate()
    return rebound, batch_trace, compiler, refinement_executions


__all__ = ["_evaluate_prepared_shared_parametric_batch"]
