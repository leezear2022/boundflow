"""Cross-axis execution of compile-owned objective-branch scorer programs."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes,protected-access
# pylint: disable=too-many-boolean-expressions,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Optional, Tuple, cast

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.bound import IntermediateBoundSource
from ..ir.cross_axis_verification_batch import (
    NativeCrossAxisVerificationBatchInstanceIR,
    NativeCrossAxisVerificationBatchPlanIR,
    NativeCrossAxisVerificationBatchScheduleIR,
    NativeCrossAxisVerificationBatchSegmentIR,
    NativeCrossAxisVerificationBatchTaskIRModule,
    NativeCrossAxisVerificationBatchTraceIR,
    lower_native_cross_axis_verification_batch_ir,
)
from .native_alpha_beta_optimizer_schedule import (
    _evaluate_state,
    _optimizer_intermediate_semantics,
)
from .native_objective_branch_score import (
    NativeObjectiveBranchExecution,
    NativeObjectiveBranchProgram,
    NativeObjectiveBranchScore,
    NativeObjectiveBranchTrace,
    _materialize_child_splits,
    _normalize_scalar_objective,
    _repeat_box_input_spec,
    _repeat_intervals,
)
from .native_prevalidated_objective_branch_score import (
    NativePrevalidatedObjectiveBranchProgram,
)
from .native_relu_split_bab_runtime import ReluSplitBranch


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeCrossAxisObjectiveBranchBinding:
    """One node/program owner admitted into a cross-axis launch."""

    clause_ordinal: int
    node_id: str
    program: NativePrevalidatedObjectiveBranchProgram

    def validate(self) -> None:
        if self.clause_ordinal < 0 or not self.node_id:
            raise ValueError("cross-axis objective branch binding is invalid")
        self.program.validate()
        if (
            len(self.program.plan.candidates)
            > self.program.branch_policy.candidate_batch_size
        ):
            raise ValueError("cross-axis objective branch requires one candidate chunk")


def _input_box_hash(program: NativePrevalidatedObjectiveBranchProgram) -> str:
    lower, upper = program.input_spec.perturbation.bounding_box(
        program.input_spec.center
    )
    return _canonical_hash(
        {
            "value_name": program.input_spec.value_name,
            "lower": tensor_content_hash(lower),
            "upper": tensor_content_hash(upper),
        }
    )


@dataclass(frozen=True)
class NativeCrossAxisPrevalidatedObjectiveBranchBatchProgram:
    """Typed Plan/Instance/Task/Schedule plus admitted node programs."""

    plan: NativeCrossAxisVerificationBatchPlanIR
    instance: NativeCrossAxisVerificationBatchInstanceIR
    task_module: NativeCrossAxisVerificationBatchTaskIRModule
    schedule: NativeCrossAxisVerificationBatchScheduleIR
    bindings: Tuple[NativeCrossAxisObjectiveBranchBinding, ...]

    def validate(self) -> None:
        self.schedule.validate(
            plan=self.plan,
            instance=self.instance,
            task_module=self.task_module,
        )
        if len(self.bindings) != self.plan.node_count:
            raise ValueError("cross-axis objective branch binding count differs")
        first = self.bindings[0]
        first.validate()
        module = first.program.module
        input_hash = _input_box_hash(first.program)
        source = first.program.intermediate_bound_source
        refine_external = first.program.refine_external_constraints
        for segment, binding in zip(self.plan.segments, self.bindings):
            binding.validate()
            program = binding.program
            capsule_hash = program.capsule.stable_hash(
                plan=program.plan,
                task_module=program.task_module,
                schedule=program.schedule,
            )
            if (
                program.module is not module
                or _input_box_hash(program) != input_hash
                or program.intermediate_bound_source != source
                or program.refine_external_constraints != refine_external
                or program.optimizer_policy.stable_hash()
                != self.plan.optimizer_policy_hash
                or program.branch_policy.stable_hash() != self.plan.branch_policy_hash
                or segment.clause_ordinal != binding.clause_ordinal
                or segment.node_id != binding.node_id
                or segment.branch_plan_hash != program.plan.stable_hash()
                or segment.capsule_hash != capsule_hash
                or segment.objective_hash != program.plan.objective_hash
                or segment.selected_state_hash != program.selected_state.stable_hash()
                or segment.candidate_count != len(program.plan.candidates)
            ):
                raise ValueError("cross-axis objective branch owner differs")


@dataclass(frozen=True)
class NativeCrossAxisPrevalidatedObjectiveBranchBatchExecution:
    """Per-node legacy-compatible executions plus one cross-axis trace."""

    program: NativeCrossAxisPrevalidatedObjectiveBranchBatchProgram
    executions: Tuple[NativeObjectiveBranchExecution, ...]
    trace: NativeCrossAxisVerificationBatchTraceIR

    def validate(self) -> None:
        self.program.validate()
        plan = self.program.plan
        if len(self.executions) != plan.node_count:
            raise ValueError("cross-axis objective branch execution count differs")
        for binding, execution in zip(self.program.bindings, self.executions):
            execution.validate()
            if (
                execution.trace.node_id != binding.node_id
                or execution.program.plan != binding.program.plan
            ):
                raise ValueError("cross-axis objective branch execution owner differs")
        self.trace.validate(
            plan=plan,
            instance=self.program.instance,
            task_module=self.program.task_module,
            schedule=self.program.schedule,
        )
        if (
            self.trace.segment_child_lower_hashes
            != tuple(item.trace.child_lower_hash for item in self.executions)
            or self.trace.segment_score_hashes
            != tuple(item.trace.score_hash for item in self.executions)
            or self.trace.selected_candidate_ordinals
            != tuple(item.trace.selected_candidate_ordinal for item in self.executions)
        ):
            raise ValueError("cross-axis objective branch Trace projection differs")


def compile_native_cross_axis_prevalidated_objective_branch_batch(
    bindings: Tuple[NativeCrossAxisObjectiveBranchBinding, ...],
    *,
    batch_id: str,
    max_child_domains: int = 512,
) -> NativeCrossAxisPrevalidatedObjectiveBranchBatchProgram:
    """Compile an already-admitted ragged ready set without re-enumeration."""

    if not batch_id or not bindings:
        raise ValueError("cross-axis objective branch batch identity is invalid")
    if len({binding.node_id for binding in bindings}) != len(bindings):
        raise ValueError("cross-axis objective branch node identity repeats")
    candidate_cursor = 0
    child_cursor = 0
    segments: list[NativeCrossAxisVerificationBatchSegmentIR] = []
    for binding in bindings:
        binding.validate()
        program = binding.program
        candidate_count = len(program.plan.candidates)
        child_count = 2 * candidate_count
        segments.append(
            NativeCrossAxisVerificationBatchSegmentIR(
                clause_ordinal=binding.clause_ordinal,
                node_id=binding.node_id,
                branch_plan_hash=program.plan.stable_hash(),
                capsule_hash=program.capsule.stable_hash(
                    plan=program.plan,
                    task_module=program.task_module,
                    schedule=program.schedule,
                ),
                objective_hash=program.plan.objective_hash,
                selected_state_hash=program.selected_state.stable_hash(),
                candidate_offset=candidate_cursor,
                candidate_count=candidate_count,
                child_domain_offset=child_cursor,
                child_domain_count=child_count,
            )
        )
        candidate_cursor += candidate_count
        child_cursor += child_count
    first = bindings[0].program
    plan = NativeCrossAxisVerificationBatchPlanIR(
        plan_id=batch_id,
        optimizer_policy_hash=first.optimizer_policy.stable_hash(),
        branch_policy_hash=first.branch_policy.stable_hash(),
        segments=tuple(segments),
        clause_count=len({binding.clause_ordinal for binding in bindings}),
        node_count=len(bindings),
        candidate_count=candidate_cursor,
        child_domain_count=child_cursor,
        max_child_domains=max_child_domains,
    )
    plan.validate()
    instance = NativeCrossAxisVerificationBatchInstanceIR.from_plan(plan)
    task_module, schedule = lower_native_cross_axis_verification_batch_ir(
        plan, instance
    )
    result = NativeCrossAxisPrevalidatedObjectiveBranchBatchProgram(
        plan=plan,
        instance=instance,
        task_module=task_module,
        schedule=schedule,
        bindings=bindings,
    )
    result.validate()
    return result


def _concatenate_tensor_state(
    programs: Tuple[NativePrevalidatedObjectiveBranchProgram, ...],
    counts: Tuple[int, ...],
    *,
    field: str,
) -> dict[str, torch.Tensor]:
    states = [getattr(program.selected_state, field) for program in programs]
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states):
        raise ValueError("cross-axis objective branch state identities differ")
    return {
        name: torch.cat(
            [
                state[name].repeat((count, *(1 for _unused in state[name].shape[1:])))
                for state, count in zip(states, counts)
            ],
            dim=0,
        ).contiguous()
        for name in keys
    }


def _concatenate_relu_pre(
    programs: Tuple[NativePrevalidatedObjectiveBranchProgram, ...],
    counts: Tuple[int, ...],
) -> dict[str, IntervalState]:
    repeated = [
        _repeat_intervals(program.relu_pre, count=count)
        for program, count in zip(programs, counts)
    ]
    keys = tuple(repeated[0])
    if any(tuple(value) != keys for value in repeated):
        raise ValueError("cross-axis objective branch ReLU identities differ")
    return {
        name: IntervalState(
            lower=torch.cat([value[name].lower for value in repeated], dim=0)
            .detach()
            .contiguous(),
            upper=torch.cat([value[name].upper for value in repeated], dim=0)
            .detach()
            .contiguous(),
        )
        for name in keys
    }


def _reduce_execution(
    binding: NativeCrossAxisObjectiveBranchBinding,
    child_lowers: torch.Tensor,
) -> NativeObjectiveBranchExecution:
    prevalidated = binding.program
    candidates = prevalidated.plan.candidates
    scores = tuple(
        NativeObjectiveBranchScore(
            candidate_ordinal=candidate.ordinal,
            inactive_lower=float(child_lowers[2 * candidate.ordinal, 0]),
            active_lower=float(child_lowers[2 * candidate.ordinal + 1, 0]),
            worst_child_lower=float(
                torch.minimum(
                    child_lowers[2 * candidate.ordinal, 0],
                    child_lowers[2 * candidate.ordinal + 1, 0],
                )
            ),
            mean_child_lower=float(
                (
                    child_lowers[2 * candidate.ordinal, 0]
                    + child_lowers[2 * candidate.ordinal + 1, 0]
                )
                / 2.0
            ),
        )
        for candidate in candidates
    )
    for score in scores:
        score.validate()
    selected_ordinal = min(
        scores,
        key=lambda score: (
            -score.worst_child_lower,
            -score.mean_child_lower,
            candidates[score.candidate_ordinal].relu_input,
            candidates[score.candidate_ordinal].neuron_index,
        ),
    ).candidate_ordinal
    selected = candidates[selected_ordinal]
    legacy = cast(NativeObjectiveBranchProgram, prevalidated)
    trace = NativeObjectiveBranchTrace(
        node_id=binding.node_id,
        program_hashes=tuple(sorted(prevalidated.hashes().items())),
        action_ids=tuple(action.action_id for action in prevalidated.schedule.actions),
        child_lower_hash=tensor_content_hash(child_lowers),
        score_hash=_canonical_hash([score.to_dict() for score in scores]),
        scores=scores,
        selected_candidate_ordinal=selected_ordinal,
    )
    execution = NativeObjectiveBranchExecution(
        branch=ReluSplitBranch(
            relu_input=selected.relu_input,
            neuron_index=selected.neuron_index,
            lower=selected.lower,
            upper=selected.upper,
            width=selected.width,
        ),
        program=legacy,
        trace=trace,
    )
    execution.validate()
    return execution


def execute_native_cross_axis_prevalidated_objective_branch_batch(
    program: NativeCrossAxisPrevalidatedObjectiveBranchBatchProgram,
) -> NativeCrossAxisPrevalidatedObjectiveBranchBatchExecution:
    """Execute one physical lower launch and project legacy-compatible traces."""

    program.validate()
    programs = tuple(binding.program for binding in program.bindings)
    child_counts = tuple(2 * len(item.plan.candidates) for item in programs)
    child_splits_per_program = tuple(
        _materialize_child_splits(
            cast(NativeObjectiveBranchProgram, item), item.plan.candidates
        )
        for item in programs
    )
    split_keys = tuple(child_splits_per_program[0])
    if any(tuple(value) != split_keys for value in child_splits_per_program):
        raise ValueError("cross-axis objective branch split identities differ")
    child_splits = {
        name: torch.cat(
            [value[name] for value in child_splits_per_program], dim=0
        ).contiguous()
        for name in split_keys
    }
    first = programs[0]
    batch_input = _repeat_box_input_spec(
        first.input_spec, count=program.plan.child_domain_count
    )
    override: Optional[Mapping[str, IntervalState]] = None
    if first.intermediate_bound_source != IntermediateBoundSource.LOCAL_FORWARD:
        override = _concatenate_relu_pre(programs, child_counts)
    interval_env, child_pre = _optimizer_intermediate_semantics(
        first.module,
        batch_input,
        relu_split_state=child_splits,
        relu_pre_override=override,
        intermediate_bound_source=first.intermediate_bound_source,
        refine_external_constraints=first.refine_external_constraints,
    )
    alpha = _concatenate_tensor_state(programs, child_counts, field="alphas")
    beta = _concatenate_tensor_state(programs, child_counts, field="betas")
    objectives = []
    for item, count in zip(programs, child_counts):
        objective = _normalize_scalar_objective(item.objective)
        if objective.dim() == 2:
            objective = objective.unsqueeze(0)
        objectives.append(objective.repeat(count, 1, 1))
    objective_batch = torch.cat(objectives, dim=0).contiguous()
    bounds = _evaluate_state(
        first.module,
        batch_input,
        objective_batch,
        alpha,
        beta,
        interval_env=interval_env,
        relu_pre=child_pre,
        relu_split_state=child_splits,
        objective="lower",
    )
    if tuple(bounds.lower.shape) != (program.plan.child_domain_count, 1):
        raise ValueError("cross-axis objective branch lower shape differs")
    batch_lowers = bounds.lower.detach().contiguous()
    executions = tuple(
        _reduce_execution(
            binding,
            batch_lowers[
                segment.child_domain_offset : segment.child_domain_offset
                + segment.child_domain_count
            ].contiguous(),
        )
        for binding, segment in zip(program.bindings, program.plan.segments)
    )
    trace = NativeCrossAxisVerificationBatchTraceIR(
        plan_hash=program.plan.stable_hash(),
        instance_hash=program.instance.stable_hash(plan=program.plan),
        task_module_hash=program.task_module.stable_hash(
            plan=program.plan, instance=program.instance
        ),
        schedule_hash=program.schedule.stable_hash(
            plan=program.plan,
            instance=program.instance,
            task_module=program.task_module,
        ),
        batch_child_lower_hash=tensor_content_hash(batch_lowers),
        segment_child_lower_hashes=tuple(
            item.trace.child_lower_hash for item in executions
        ),
        segment_score_hashes=tuple(item.trace.score_hash for item in executions),
        selected_candidate_ordinals=tuple(
            item.trace.selected_candidate_ordinal for item in executions
        ),
        lower_launch_count=1,
    )
    result = NativeCrossAxisPrevalidatedObjectiveBranchBatchExecution(
        program=program,
        executions=executions,
        trace=trace,
    )
    result.validate()
    return result


__all__ = [
    "NativeCrossAxisObjectiveBranchBinding",
    "NativeCrossAxisPrevalidatedObjectiveBranchBatchExecution",
    "NativeCrossAxisPrevalidatedObjectiveBranchBatchProgram",
    "compile_native_cross_axis_prevalidated_objective_branch_batch",
    "execute_native_cross_axis_prevalidated_objective_branch_batch",
]
