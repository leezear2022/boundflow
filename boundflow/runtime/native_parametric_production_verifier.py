"""Parametric-template production ReLU-split verifier queue."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes,protected-access
# pylint: disable=too-many-lines,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import time
from typing import Mapping, Optional, Sequence

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.parametric_optimizer import (
    NativeParametricOptimizerCacheEventIR,
    NativeParametricOptimizerInstanceIR,
)
from ..ir.production_verifier import (
    NativeProductionVerifierPlanIR,
    NativeProductionVerifierTaskKind,
    lower_native_production_verifier_ir,
)
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    build_native_alpha_beta_scope,
)
from .native_objective_branch_score import (
    NativeObjectiveBranchExecution,
    NativeObjectiveBranchPolicy,
    compile_native_objective_branch_program,
    execute_native_objective_branch_program,
)
from .native_optimized_relu_split_bab_runtime import (
    _batched_split_state,
    _repeat_relu_pre_override,
)
from .native_parametric_optimizer import (
    NativeParametricOptimizerInstance,
    NativeParametricOptimizerTemplateCache,
    execute_native_parametric_optimizer,
    instantiate_native_parametric_optimizer,
)
from .native_production_verifier import (
    NativeProductionBabEvaluation,
    NativeProductionReluSplitBabExecution,
    NativeProductionReluSplitBabTrace,
    NativeProductionVerifierActionTrace,
    NativeProductionVerifierBatchTrace,
    _build_parent_warm_state,
    _ProductionEvaluatedNode,
)
from .native_relu_split_bab_runtime import (
    BabQueueStatus,
    NativeReluSplitBabConfig,
    NativeReluSplitBabDecision,
    NativeReluSplitBabNode,
    _make_child_runtime_node,
    _normalize_scalar_objective,
    _priority,
    _QueueEntry,
    _repeat_box_input_spec,
    _root_box_bounds,
    _RuntimeNode,
    _select_branch,
    _slice_interval,
)
from .task_executor import InputSpec

NATIVE_PARAMETRIC_COMPILER_BATCH_TRACE_SCHEMA_VERSION = (
    "boundflow.native-parametric-compiler-batch-trace/v1"
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeParametricCompilerBatchTrace:
    """Template decision, exact instance, and compiler/runtime phase timing."""

    batch_id: str
    cache_event: NativeParametricOptimizerCacheEventIR
    instance_ir: NativeParametricOptimizerInstanceIR
    template_hash: str
    task_hash: str
    schedule_hash: str
    acquire_elapsed_ns: int
    instantiate_elapsed_ns: int
    execute_elapsed_ns: int
    schema_version: str = NATIVE_PARAMETRIC_COMPILER_BATCH_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        self.cache_event.validate()
        self.instance_ir.validate()
        if (
            self.schema_version != NATIVE_PARAMETRIC_COMPILER_BATCH_TRACE_SCHEMA_VERSION
            or not self.batch_id
            or self.cache_event.batch_id != self.batch_id
            or self.instance_ir.instance_id != f"{self.batch_id}:optimizer-instance"
            or self.cache_event.template_hash != self.template_hash
            or self.instance_ir.template_hash != self.template_hash
            or self.instance_ir.cache_key != self.cache_event.cache_key
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (self.template_hash, self.task_hash, self.schedule_hash)
            )
            or any(
                elapsed < 0
                for elapsed in (
                    self.acquire_elapsed_ns,
                    self.instantiate_elapsed_ns,
                    self.execute_elapsed_ns,
                )
            )
        ):
            raise ValueError("parametric compiler batch trace differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "cache_event": self.cache_event.to_dict(),
            "cache_event_hash": self.cache_event.stable_hash(),
            "instance": self.instance_ir.to_dict(),
            "instance_hash": self.instance_ir.stable_hash(),
            "template_hash": self.template_hash,
            "task_hash": self.task_hash,
            "schedule_hash": self.schedule_hash,
            "acquire_elapsed_ns": self.acquire_elapsed_ns,
            "instantiate_elapsed_ns": self.instantiate_elapsed_ns,
            "execute_elapsed_ns": self.execute_elapsed_ns,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeParametricProductionReluSplitBabExecution:
    """Production queue semantics plus parametric compiler evidence."""

    queue: NativeProductionReluSplitBabExecution
    compiler_batches: tuple[NativeParametricCompilerBatchTrace, ...]

    @property
    def trace(self) -> NativeProductionReluSplitBabTrace:
        return self.queue.trace

    def state_map(self) -> dict[str, NativeAlphaBetaOptimizationState]:
        return self.queue.state_map()

    def validate(self) -> None:
        self.queue.validate()
        if len(self.compiler_batches) != len(self.queue.trace.batches):
            raise ValueError("parametric compiler/queue batch coverage differs")
        for compiler, batch in zip(self.compiler_batches, self.queue.trace.batches):
            compiler.validate()
            hashes = dict(batch.plan.optimizer_ir_hashes)
            if (
                compiler.batch_id != batch.plan.plan_id
                or hashes["optimizer_plan_hash"] != compiler.instance_ir.stable_hash()
                or hashes["optimizer_task_module_hash"] != compiler.task_hash
                or hashes["optimizer_schedule_hash"] != compiler.schedule_hash
                or batch.plan.state_scope_hash != compiler.instance_ir.state_scope_hash
            ):
                raise ValueError("parametric compiler/production Plan binding differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "execution_mode": "production_parametric_template_instance",
            "queue": self.queue.trace.to_dict(),
            "queue_trace_hash": self.queue.trace.stable_hash(),
            "compiler_batches": [item.to_dict() for item in self.compiler_batches],
            "audit_hash_chain_constructed": False,
            "selected_native_reexecution": False,
            "performance_claimed": False,
        }


def _slice_parametric_state(
    instance: NativeParametricOptimizerInstance,
    root_input_spec: InputSpec,
    *,
    module: BFTaskModule,
    objective: torch.Tensor,
    selected_state: NativeAlphaBetaOptimizationState,
    index: int,
) -> NativeAlphaBetaOptimizationState:
    single_input = _repeat_box_input_spec(root_input_spec, count=1)
    node_pre = {
        name: _slice_interval(value, index=index)
        for name, value in instance.relu_pre.items()
    }
    node_split = {
        name: value[index : index + 1].contiguous()
        for name, value in selected_state.splits.items()
    }
    scope = build_native_alpha_beta_scope(
        module,
        single_input,
        linear_spec_C=objective,
        relu_pre=node_pre,
        relu_split_state=node_split,
        policy=instance.policy,
    )
    state = NativeAlphaBetaOptimizationState(
        scope=scope,
        split_by_relu_input=tuple(sorted(node_split.items())),
        alpha_by_relu_input=tuple(
            (name, value[index : index + 1].detach().contiguous().clone())
            for name, value in sorted(selected_state.alphas.items())
        ),
        beta_by_relu_input=tuple(
            (name, value[index : index + 1].detach().contiguous().clone())
            for name, value in sorted(selected_state.betas.items())
        ),
    )
    state.validate()
    return state


def _parametric_production_plan(
    *,
    batch_id: str,
    nodes: tuple[_RuntimeNode, ...],
    parent_by_id: Mapping[str, _ProductionEvaluatedNode],
    instance: NativeParametricOptimizerInstance,
) -> NativeProductionVerifierPlanIR:
    scope = instance.initial_state.scope
    template = instance.template
    hashes = {
        "optimizer_plan_hash": instance.ir.stable_hash(),
        "optimizer_task_module_hash": template.task_hash,
        "optimizer_schedule_hash": template.schedule_hash,
    }
    plan = NativeProductionVerifierPlanIR(
        plan_id=batch_id,
        node_ids=tuple(node.node.node_id for node in nodes),
        node_split_state_hashes=tuple(node.node.split_state_hash for node in nodes),
        parent_selected_state_hashes=tuple(
            (
                None
                if node.node.parent_node_id is None
                else parent_by_id[node.node.parent_node_id].selected_state.stable_hash()
            )
            for node in nodes
        ),
        state_scope_hash=scope.stable_hash(),
        primal_graph_hash=scope.primal_graph_hash,
        input_region_hash=scope.input_region_hash,
        objective_hash=scope.objective_hash,
        optimizer_policy_hash=scope.optimizer_policy_hash,
        intermediate_bounds_hash=scope.intermediate_bounds_hash,
        intermediate_bound_source=instance.intermediate_bound_source.value,
        optimizer_ir_hashes=tuple(sorted(hashes.items())),
    )
    plan.validate()
    return plan


def _evaluate_parametric_node_batch(  # pylint: disable=too-many-statements
    module: BFTaskModule,
    root_input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    batch_id: str,
    policy: NativeAlphaBetaOptimizerPolicy,
    parent_by_id: Mapping[str, _ProductionEvaluatedNode],
    relu_pre_override: Optional[Mapping[str, IntervalState]],
    template_relu_pre: Mapping[str, IntervalState],
    intermediate_bound_source: IntermediateBoundSource,
    objective_branch_policy: Optional[NativeObjectiveBranchPolicy],
    refine_external_constraints: bool,
    compiler_cache: NativeParametricOptimizerTemplateCache,
) -> tuple[
    tuple[_ProductionEvaluatedNode, ...],
    NativeProductionVerifierBatchTrace,
    NativeParametricCompilerBatchTrace,
    tuple[tuple[str, NativeObjectiveBranchExecution], ...],
]:
    if not nodes:
        raise ValueError("parametric production node batch cannot be empty")
    batch_input = _repeat_box_input_spec(root_input_spec, count=len(nodes))
    split_batch = _batched_split_state(nodes)
    batch_relu_pre_override = (
        None
        if relu_pre_override is None
        else _repeat_relu_pre_override(relu_pre_override, count=len(nodes))
    )
    warm_state = (
        None
        if nodes[0].node.depth == 0
        else _build_parent_warm_state(
            module,
            batch_input,
            objective=objective,
            nodes=nodes,
            policy=policy,
            parent_by_id=parent_by_id,
            relu_pre_override=batch_relu_pre_override,
            intermediate_bound_source=intermediate_bound_source,
            refine_external_constraints=refine_external_constraints,
        )
    )
    if any((node.node.depth == 0) != (warm_state is None) for node in nodes):
        raise ValueError("parametric production batch mixes root and child nodes")

    acquire_started_ns = time.perf_counter_ns()
    template, cache_event = compiler_cache.acquire(
        module,
        batch_input,
        linear_spec_C=objective,
        relu_pre=template_relu_pre,
        policy=policy,
        intermediate_bound_source=intermediate_bound_source,
        refine_external_constraints=refine_external_constraints,
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
        relu_pre_override=batch_relu_pre_override,
        intermediate_bound_source=intermediate_bound_source,
        refine_external_constraints=refine_external_constraints,
    )
    instantiate_elapsed_ns = time.perf_counter_ns() - instantiate_started_ns
    plan = _parametric_production_plan(
        batch_id=batch_id,
        nodes=nodes,
        parent_by_id=parent_by_id,
        instance=instance,
    )
    task_ir, schedule = lower_native_production_verifier_ir(plan)

    selected = None
    materialized: Optional[tuple[_ProductionEvaluatedNode, ...]] = None
    branch_executions: list[tuple[str, NativeObjectiveBranchExecution]] = []
    action_traces: list[NativeProductionVerifierActionTrace] = []
    execute_elapsed_ns = 0
    for action in schedule.actions:
        started_ns = time.perf_counter_ns()
        if action.kind == NativeProductionVerifierTaskKind.VALIDATE_PROGRAM:
            instance.require_exact_runtime(
                module,
                batch_input,
                linear_spec_C=objective,
                intermediate_bound_source=intermediate_bound_source,
            )
        elif action.kind == NativeProductionVerifierTaskKind.EXECUTE_OPTIMIZER:
            selected = execute_native_parametric_optimizer(
                instance,
                module,
                batch_input,
                linear_spec_C=objective,
                intermediate_bound_source=intermediate_bound_source,
            )
        elif action.kind == NativeProductionVerifierTaskKind.MATERIALIZE_NODE_RESULTS:
            if selected is None:
                raise ValueError("parametric production materializes before execution")
            if tuple(selected.bounds.lower.shape) != (len(nodes), 1):
                raise ValueError("parametric node batch must return scalar objectives")
            items: list[_ProductionEvaluatedNode] = []
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
                    raise ValueError("parametric production node lacks a parent")
                branch = _select_branch(node_pre, relu_split_state=node_split)
                if objective_branch_policy is not None and branch is not None:
                    branch_program = compile_native_objective_branch_program(
                        module,
                        _repeat_box_input_spec(root_input_spec, count=1),
                        linear_spec_C=objective,
                        relu_pre=node_pre,
                        selected_state=state,
                        optimizer_policy=policy,
                        branch_policy=objective_branch_policy,
                        intermediate_bound_source=intermediate_bound_source,
                        refine_external_constraints=refine_external_constraints,
                        plan_id=f"{batch_id}:node:{index}:objective-branch",
                    )
                    branch_execution = execute_native_objective_branch_program(
                        branch_program, node_id=runtime_node.node.node_id
                    )
                    branch = branch_execution.branch
                    branch_executions.append(
                        (runtime_node.node.node_id, branch_execution)
                    )
                lower = float(selected.bounds.lower[index, 0].item())
                upper = float(selected.bounds.upper[index, 0].item())
                items.append(
                    _ProductionEvaluatedNode(
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
                    )
                )
            materialized = tuple(items)
        elif action.kind == NativeProductionVerifierTaskKind.COMMIT_QUEUE_RESULTS:
            if materialized is None:
                raise ValueError("parametric production commits before materialization")
        else:
            raise AssertionError("unreachable parametric production task kind")
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
        raise ValueError("parametric production Schedule did not produce results")
    batch_trace = NativeProductionVerifierBatchTrace(
        plan=plan,
        task_ir=task_ir,
        schedule=schedule,
        actions=tuple(action_traces),
        selected_batch_state_hash=selected.state.stable_hash(),
    )
    batch_trace.validate()
    trace_hash = batch_trace.stable_hash()
    rebound = tuple(
        _ProductionEvaluatedNode(
            runtime_node=item.runtime_node,
            evaluation=NativeProductionBabEvaluation(
                **{**item.evaluation.__dict__, "batch_trace_hash": trace_hash}
            ),
            selected_state=item.selected_state,
            relu_pre=item.relu_pre,
        )
        for item in materialized
    )
    for item in rebound:
        item.evaluation.validate()
    compiler_trace = NativeParametricCompilerBatchTrace(
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
    compiler_trace.validate()
    return rebound, batch_trace, compiler_trace, tuple(branch_executions)


def execute_native_parametric_production_relu_split_bab(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    run_id: str,
    config: NativeReluSplitBabConfig,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    compiler_cache: NativeParametricOptimizerTemplateCache,
    relu_pre_override: Optional[Mapping[str, IntervalState]] = None,
    intermediate_bound_source: IntermediateBoundSource = (
        IntermediateBoundSource.LOCAL_FORWARD
    ),
    objective_branch_policy: Optional[NativeObjectiveBranchPolicy] = None,
    refine_external_constraints: bool = False,
) -> NativeParametricProductionReluSplitBabExecution:
    """Run the production queue using a shared parametric optimizer template."""

    if not run_id:
        raise ValueError("parametric production run ID must be non-empty")
    if not isinstance(compiler_cache, NativeParametricOptimizerTemplateCache):
        raise TypeError("parametric production compiler cache is invalid")
    config.validate()
    optimizer_policy.validate()
    module.validate()
    if objective_branch_policy is not None:
        objective_branch_policy.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("parametric production intermediate-bound source is invalid")
    if not isinstance(refine_external_constraints, bool):
        raise TypeError("parametric production external refinement flag is invalid")
    if refine_external_constraints and intermediate_bound_source != (
        IntermediateBoundSource.EXTERNAL_VERIFIER
    ):
        raise ValueError("external constraint refinement requires external provenance")
    if (relu_pre_override is None) != (
        intermediate_bound_source == IntermediateBoundSource.LOCAL_FORWARD
    ):
        raise ValueError("parametric production semantics/provenance differ")

    lower, upper = _root_box_bounds(input_spec)
    objective = _normalize_scalar_objective(linear_spec_C)
    _root_interval, root_pre = _forward_ibp_trace_mlp(module, input_spec)
    template_relu_pre = root_pre if relu_pre_override is None else relu_pre_override
    root_splits = tuple(
        (
            name,
            torch.zeros(
                tuple(int(dimension) for dimension in pre.lower.shape[1:]),
                dtype=torch.int8,
                device=pre.lower.device,
            ),
        )
        for name, pre in sorted(root_pre.items())
    )
    if not root_splits:
        raise ValueError("parametric production verifier requires at least one ReLU")
    root_mapping = {name: value.unsqueeze(0) for name, value in root_splits}
    root = _RuntimeNode(
        node=NativeReluSplitBabNode(
            node_id=f"{run_id}:n000000",
            parent_node_id=None,
            depth=0,
            branch_relu_input=None,
            branch_neuron_index=None,
            branch_value=0,
            split_state_hash=relu_split_state_hash(root_mapping),
        ),
        split_state=root_splits,
    )

    evaluations: list[NativeProductionBabEvaluation] = []
    decisions: list[NativeReluSplitBabDecision] = []
    batches: list[NativeProductionVerifierBatchTrace] = []
    compiler_batches: list[NativeParametricCompilerBatchTrace] = []
    runtime_by_id: dict[str, _ProductionEvaluatedNode] = {}
    objective_branch_executions: list[tuple[str, NativeObjectiveBranchExecution]] = []
    batch_serial = 0
    next_node_serial = 1

    def evaluate(nodes: Sequence[_RuntimeNode]) -> None:
        nonlocal batch_serial
        for start in range(0, len(nodes), config.max_eval_batch_size):
            chunk = tuple(nodes[start : start + config.max_eval_batch_size])
            batch_id = f"{run_id}:eval:{batch_serial:04d}"
            batch_serial += 1
            evaluated, batch, compiler, branches = _evaluate_parametric_node_batch(
                module,
                input_spec,
                objective=objective,
                nodes=chunk,
                batch_id=batch_id,
                policy=optimizer_policy,
                parent_by_id=runtime_by_id,
                relu_pre_override=relu_pre_override,
                template_relu_pre=template_relu_pre,
                intermediate_bound_source=intermediate_bound_source,
                objective_branch_policy=objective_branch_policy,
                refine_external_constraints=refine_external_constraints,
                compiler_cache=compiler_cache,
            )
            batches.append(batch)
            compiler_batches.append(compiler)
            objective_branch_executions.extend(branches)
            evaluations.extend(item.evaluation for item in evaluated)
            runtime_by_id.update(
                {item.runtime_node.node.node_id: item for item in evaluated}
            )

    evaluate((root,))
    heap: list[_QueueEntry] = []
    root_evaluation = runtime_by_id[root.node.node_id].evaluation
    heapq.heappush(heap, _QueueEntry(root_evaluation.priority, 0, root.node.node_id))
    max_queue_size = 1
    budget_exhausted = False

    while heap and not budget_exhausted:
        selected_entries = [
            heapq.heappop(heap)
            for _unused in range(min(config.expansion_batch_size, len(heap)))
        ]
        generated: list[_RuntimeNode] = []
        for selected_index, entry in enumerate(selected_entries):
            evaluated = runtime_by_id[entry.node_id]
            node = evaluated.runtime_node.node
            result = evaluated.evaluation
            if result.lower >= config.threshold:
                decisions.append(
                    NativeReluSplitBabDecision(
                        decision_index=len(decisions),
                        node_id=node.node_id,
                        kind="prune",
                        reason="lower_bound_meets_threshold",
                    )
                )
                continue
            if node.depth >= config.max_depth:
                decisions.append(
                    NativeReluSplitBabDecision(
                        decision_index=len(decisions),
                        node_id=node.node_id,
                        kind="terminal",
                        reason="configured_depth_limit",
                    )
                )
                continue
            branch = result.branch_candidate
            if branch is None:
                decisions.append(
                    NativeReluSplitBabDecision(
                        decision_index=len(decisions),
                        node_id=node.node_id,
                        kind="terminal",
                        reason="no_unsplit_ambiguous_relu",
                    )
                )
                continue
            if len(evaluations) + len(generated) + 2 > config.max_nodes:
                for pending in selected_entries[selected_index:]:
                    heapq.heappush(heap, pending)
                budget_exhausted = True
                break
            children: list[_RuntimeNode] = []
            for branch_value in (-1, 1):
                child_id = f"{run_id}:n{next_node_serial:06d}"
                next_node_serial += 1
                children.append(
                    _make_child_runtime_node(
                        evaluated.runtime_node,
                        child_id=child_id,
                        branch=branch,
                        branch_value=branch_value,
                    )
                )
            generated.extend(children)
            decisions.append(
                NativeReluSplitBabDecision(
                    decision_index=len(decisions),
                    node_id=node.node_id,
                    kind="expand",
                    reason=(
                        "objective_bound_impact"
                        if objective_branch_policy is not None
                        else "widest_unsplit_ambiguous_relu"
                    ),
                    child_node_ids=tuple(item.node.node_id for item in children),
                    branch_candidate=branch,
                )
            )
        if generated:
            evaluate(generated)
            for child in generated:
                evaluation = runtime_by_id[child.node.node_id].evaluation
                heapq.heappush(
                    heap,
                    _QueueEntry(
                        evaluation.priority,
                        next_node_serial,
                        child.node.node_id,
                    ),
                )
            max_queue_size = max(max_queue_size, len(heap))

    frontier = tuple(
        entry.node_id
        for entry in sorted(heap, key=lambda item: (item.priority, item.serial))
    )
    status: BabQueueStatus = "budget_exhausted" if budget_exhausted else "complete"
    trace = NativeProductionReluSplitBabTrace(
        run_id=run_id,
        status=status,
        termination_reason=(
            "node_budget_exhausted"
            if budget_exhausted
            else "configured_bounded_tree_exhausted"
        ),
        config=config,
        optimizer_policy=optimizer_policy,
        intermediate_bound_source=intermediate_bound_source,
        root_input_lower_hash=tensor_content_hash(lower),
        root_input_upper_hash=tensor_content_hash(upper),
        objective_hash=tensor_content_hash(objective),
        evaluations=tuple(evaluations),
        decisions=tuple(decisions),
        final_frontier_node_ids=frontier,
        batches=tuple(batches),
        max_queue_size=max_queue_size,
    )
    trace.validate()
    queue = NativeProductionReluSplitBabExecution(
        trace=trace,
        selected_states=tuple(
            (
                evaluation.node.node_id,
                runtime_by_id[evaluation.node.node_id].selected_state,
            )
            for evaluation in trace.evaluations
        ),
        objective_branch_executions=tuple(objective_branch_executions),
        objective_branch_policy=objective_branch_policy,
    )
    execution = NativeParametricProductionReluSplitBabExecution(
        queue=queue,
        compiler_batches=tuple(compiler_batches),
    )
    execution.validate()
    return execution


__all__ = [
    "NATIVE_PARAMETRIC_COMPILER_BATCH_TRACE_SCHEMA_VERSION",
    "NativeParametricCompilerBatchTrace",
    "NativeParametricProductionReluSplitBabExecution",
    "execute_native_parametric_production_relu_split_bab",
]
