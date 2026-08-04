"""Schedule-driven objective-bound-impact scoring for native ReLU branches."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,invalid-name
# pylint: disable=not-an-iterable,unsubscriptable-object

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.branch import (
    NativeObjectiveBranchCandidateIR,
    NativeObjectiveBranchPlanIR,
    NativeObjectiveBranchScheduleIR,
    NativeObjectiveBranchTaskIRModule,
    ObjectiveBranchTaskKind,
    lower_native_objective_branch_ir,
)
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    build_native_alpha_beta_scope,
)
from .native_alpha_beta_optimizer_schedule import (
    _evaluate_state,
    _optimizer_intermediate_semantics,
)
from .native_relu_split_bab_runtime import ReluSplitBranch
from .task_executor import InputSpec

NATIVE_OBJECTIVE_BRANCH_PROGRAM_SCHEMA_VERSION = (
    "boundflow.native-objective-branch-program/v1"
)
NATIVE_OBJECTIVE_BRANCH_TRACE_SCHEMA_VERSION = (
    "boundflow.native-objective-branch-trace/v1"
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NativeObjectiveBranchPolicy:
    """Deterministic shortlist, batch, and reduction policy."""

    candidates_per_relu: int = 8
    candidate_batch_size: int = 64
    max_candidates: int = 256
    candidate_policy_id: str = "top_width_per_relu_v1"
    reduce_policy: str = "maximize_worst_child_then_mean"

    def validate(self) -> None:
        if (
            self.candidates_per_relu < 1
            or self.candidate_batch_size < 1
            or self.max_candidates < 1
            or self.candidate_policy_id != "top_width_per_relu_v1"
            or self.reduce_policy != "maximize_worst_child_then_mean"
        ):
            raise ValueError("native objective branch policy is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "candidates_per_relu": self.candidates_per_relu,
            "candidate_batch_size": self.candidate_batch_size,
            "max_candidates": self.max_candidates,
            "candidate_policy_id": self.candidate_policy_id,
            "reduce_policy": self.reduce_policy,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchProgram:
    """Exact score program with first-class Plan/Task/Schedule IR."""

    plan: NativeObjectiveBranchPlanIR
    task_module: NativeObjectiveBranchTaskIRModule
    schedule: NativeObjectiveBranchScheduleIR
    module: BFTaskModule
    input_spec: InputSpec
    objective: torch.Tensor
    relu_pre: Mapping[str, IntervalState]
    selected_state: NativeAlphaBetaOptimizationState
    optimizer_policy: NativeAlphaBetaOptimizerPolicy
    branch_policy: NativeObjectiveBranchPolicy
    intermediate_bound_source: IntermediateBoundSource
    refine_external_constraints: bool = False
    schema_version: str = NATIVE_OBJECTIVE_BRANCH_PROGRAM_SCHEMA_VERSION

    def validate(self) -> None:
        self.plan.validate()
        self.task_module.validate(plan=self.plan)
        self.schedule.validate(plan=self.plan, task_module=self.task_module)
        self.selected_state.validate()
        self.optimizer_policy.validate()
        self.branch_policy.validate()
        objective = _normalize_scalar_objective(self.objective)
        splits = self.selected_state.splits
        scope = build_native_alpha_beta_scope(
            self.module,
            self.input_spec,
            linear_spec_C=objective,
            relu_pre=self.relu_pre,
            relu_split_state=splits,
            policy=self.optimizer_policy,
        )
        expected_intermediate = (
            "external_verifier_refined"
            if self.refine_external_constraints
            else self.intermediate_bound_source.value
        )
        if (
            self.schema_version != NATIVE_OBJECTIVE_BRANCH_PROGRAM_SCHEMA_VERSION
            or self.plan.objective_hash != tensor_content_hash(objective)
            or self.plan.split_state_hash != relu_split_state_hash(splits)
            or self.plan.selected_state_hash != self.selected_state.stable_hash()
            or self.plan.state_scope_hash != scope.stable_hash()
            or self.plan.policy_hash != self.branch_policy.stable_hash()
            or self.plan.candidate_policy_id != self.branch_policy.candidate_policy_id
            or self.plan.candidates_per_relu != self.branch_policy.candidates_per_relu
            or self.plan.candidate_batch_size != self.branch_policy.candidate_batch_size
            or self.plan.max_candidates != self.branch_policy.max_candidates
            or self.plan.intermediate_bound_source != expected_intermediate
            or self.plan.reduce_policy != self.branch_policy.reduce_policy
            or scope != self.selected_state.scope
            or tuple(self.relu_pre) != tuple(splits)
        ):
            raise ValueError("native objective branch program identity differs")
        candidates = _enumerate_candidates(
            self.relu_pre,
            splits,
            policy=self.branch_policy,
        )
        if candidates != self.plan.candidates:
            raise ValueError("native objective branch candidate enumeration differs")

    def hashes(self) -> dict[str, str]:
        self.validate()
        return {
            "branch_plan_hash": self.plan.stable_hash(),
            "branch_task_module_hash": self.task_module.stable_hash(plan=self.plan),
            "branch_schedule_hash": self.schedule.stable_hash(
                plan=self.plan, task_module=self.task_module
            ),
        }


@dataclass(frozen=True)
class NativeObjectiveBranchScore:
    """Fixed-state lower estimates for both children of one candidate."""

    candidate_ordinal: int
    inactive_lower: float
    active_lower: float
    worst_child_lower: float
    mean_child_lower: float

    def validate(self) -> None:
        values = (
            self.inactive_lower,
            self.active_lower,
            self.worst_child_lower,
            self.mean_child_lower,
        )
        if (
            self.candidate_ordinal < 0
            or not all(math.isfinite(value) for value in values)
            or not math.isclose(
                self.worst_child_lower,
                min(self.inactive_lower, self.active_lower),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or not math.isclose(
                self.mean_child_lower,
                (self.inactive_lower + self.active_lower) / 2.0,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("native objective branch score is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "candidate_ordinal": self.candidate_ordinal,
            "inactive_lower": self.inactive_lower,
            "active_lower": self.active_lower,
            "worst_child_lower": self.worst_child_lower,
            "mean_child_lower": self.mean_child_lower,
        }


@dataclass(frozen=True)
class NativeObjectiveBranchTrace:
    """Replayable score outputs bound to exact branch Plan/Task/Schedule hashes."""

    node_id: str
    program_hashes: tuple[tuple[str, str], ...]
    action_ids: tuple[str, ...]
    child_lower_hash: str
    score_hash: str
    scores: tuple[NativeObjectiveBranchScore, ...]
    selected_candidate_ordinal: int
    performance_claimed: bool = False
    schema_version: str = NATIVE_OBJECTIVE_BRANCH_TRACE_SCHEMA_VERSION

    def validate(self, *, program: NativeObjectiveBranchProgram) -> None:
        program.validate()
        score_ordinals = tuple(score.candidate_ordinal for score in self.scores)
        for score in self.scores:
            score.validate()
        expected_selected = min(
            self.scores,
            key=lambda score: (
                -score.worst_child_lower,
                -score.mean_child_lower,
                program.plan.candidates[score.candidate_ordinal].relu_input,
                program.plan.candidates[score.candidate_ordinal].neuron_index,
            ),
        ).candidate_ordinal
        if (
            self.schema_version != NATIVE_OBJECTIVE_BRANCH_TRACE_SCHEMA_VERSION
            or not self.node_id
            or dict(self.program_hashes) != program.hashes()
            or len(self.program_hashes) != len(program.hashes())
            or self.action_ids
            != tuple(action.action_id for action in program.schedule.actions)
            or not _is_sha256(self.child_lower_hash)
            or not _is_sha256(self.score_hash)
            or score_ordinals != tuple(range(len(program.plan.candidates)))
            or self.selected_candidate_ordinal != expected_selected
            or self.performance_claimed is not False
        ):
            raise ValueError("native objective branch trace is invalid")
        if self.score_hash != _canonical_hash(
            [score.to_dict() for score in self.scores]
        ):
            raise ValueError("native objective branch score hash differs")

    def to_dict(self, *, program: NativeObjectiveBranchProgram) -> dict[str, object]:
        self.validate(program=program)
        return {
            "schema_version": self.schema_version,
            "node_id": self.node_id,
            "performance_claimed": self.performance_claimed,
            "program_hashes": dict(self.program_hashes),
            "action_ids": list(self.action_ids),
            "child_lower_hash": self.child_lower_hash,
            "score_hash": self.score_hash,
            "scores": [score.to_dict() for score in self.scores],
            "selected_candidate_ordinal": self.selected_candidate_ordinal,
            "selected_candidate": program.plan.candidates[
                self.selected_candidate_ordinal
            ].to_dict(),
        }

    def stable_hash(self, *, program: NativeObjectiveBranchProgram) -> str:
        return _canonical_hash(self.to_dict(program=program))


@dataclass(frozen=True)
class NativeObjectiveBranchExecution:
    """Selected branch plus its exact program and score trace."""

    branch: ReluSplitBranch
    program: NativeObjectiveBranchProgram
    trace: NativeObjectiveBranchTrace

    def validate(self) -> None:
        self.trace.validate(program=self.program)
        selected = self.program.plan.candidates[self.trace.selected_candidate_ordinal]
        self.branch.validate()
        if (
            self.branch.relu_input != selected.relu_input
            or self.branch.neuron_index != selected.neuron_index
            or self.branch.lower != selected.lower
            or self.branch.upper != selected.upper
            or self.branch.width != selected.width
        ):
            raise ValueError("native objective branch execution differs")


def _normalize_scalar_objective(linear_spec_C: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(linear_spec_C) or not torch.is_floating_point(linear_spec_C):
        raise TypeError("objective branch objective must be floating point")
    objective = linear_spec_C.detach().contiguous()
    if objective.dim() == 2 and int(objective.shape[0]) == 1:
        return objective
    if objective.dim() == 3 and tuple(objective.shape[:2]) == (1, 1):
        return objective
    raise ValueError("objective branch requires one domain and scalar objective")


def _enumerate_candidates(
    relu_pre: Mapping[str, IntervalState],
    relu_split_state: Mapping[str, torch.Tensor],
    *,
    policy: NativeObjectiveBranchPolicy,
) -> tuple[NativeObjectiveBranchCandidateIR, ...]:
    policy.validate()
    if tuple(relu_pre) != tuple(relu_split_state):
        raise ValueError("objective branch ReLU/split identities differ")
    shortlisted: list[tuple[str, int, float, float, float]] = []
    for name in sorted(relu_pre):
        pre = relu_pre[name]
        split = relu_split_state[name]
        if (
            int(pre.lower.shape[0]) != 1
            or pre.lower.shape != pre.upper.shape
            or split.shape != pre.lower.shape
        ):
            raise ValueError("objective branch requires one exact ReLU domain")
        lower = pre.lower.reshape(-1)
        upper = pre.upper.reshape(-1)
        split_flat = split.reshape(-1)
        candidates = [
            (
                float((upper[index] - lower[index]).item()),
                index,
                float(lower[index].item()),
                float(upper[index].item()),
            )
            for index in range(int(split_flat.numel()))
            if int(split_flat[index].item()) == 0
            and float(lower[index].item()) < 0.0 < float(upper[index].item())
        ]
        for width, index, low, high in sorted(
            candidates, key=lambda item: (-item[0], item[1])
        )[: policy.candidates_per_relu]:
            shortlisted.append((name, index, low, high, width))
    shortlisted.sort(key=lambda item: (item[0], item[1]))
    if not shortlisted:
        raise ValueError("objective branch has no unsplit ambiguous candidate")
    if len(shortlisted) > policy.max_candidates:
        raise ValueError("objective branch candidate safety cap exceeded")
    result = tuple(
        NativeObjectiveBranchCandidateIR(
            ordinal=ordinal,
            relu_input=name,
            neuron_index=index,
            lower=low,
            upper=high,
            width=width,
        )
        for ordinal, (name, index, low, high, width) in enumerate(shortlisted)
    )
    for candidate in result:
        candidate.validate()
    return result


def compile_native_objective_branch_program(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_pre: Mapping[str, IntervalState],
    selected_state: NativeAlphaBetaOptimizationState,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    intermediate_bound_source: IntermediateBoundSource,
    refine_external_constraints: bool = False,
    plan_id: str,
) -> NativeObjectiveBranchProgram:
    """Compile one exact selected optimizer state into branch score IR."""

    if not plan_id:
        raise ValueError("objective branch plan ID must be non-empty")
    module.validate()
    selected_state.validate()
    optimizer_policy.validate()
    branch_policy.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("objective branch intermediate source is invalid")
    if not isinstance(refine_external_constraints, bool):
        raise TypeError("objective branch external refinement flag is invalid")
    if refine_external_constraints and intermediate_bound_source != (
        IntermediateBoundSource.EXTERNAL_VERIFIER
    ):
        raise ValueError("objective branch refinement requires external provenance")
    objective = _normalize_scalar_objective(linear_spec_C)
    candidates = _enumerate_candidates(
        relu_pre, selected_state.splits, policy=branch_policy
    )
    plan = NativeObjectiveBranchPlanIR(
        plan_id=plan_id,
        objective_hash=tensor_content_hash(objective),
        split_state_hash=relu_split_state_hash(selected_state.splits),
        selected_state_hash=selected_state.stable_hash(),
        state_scope_hash=selected_state.scope.stable_hash(),
        policy_hash=branch_policy.stable_hash(),
        candidate_policy_id=branch_policy.candidate_policy_id,
        candidates_per_relu=branch_policy.candidates_per_relu,
        candidate_batch_size=branch_policy.candidate_batch_size,
        max_candidates=branch_policy.max_candidates,
        intermediate_bound_source=(
            "external_verifier_refined"
            if refine_external_constraints
            else intermediate_bound_source.value
        ),
        candidates=candidates,
        reduce_policy=branch_policy.reduce_policy,
    )
    task_module, schedule = lower_native_objective_branch_ir(plan)
    program = NativeObjectiveBranchProgram(
        plan=plan,
        task_module=task_module,
        schedule=schedule,
        module=module,
        input_spec=input_spec,
        objective=objective,
        relu_pre=dict(relu_pre),
        selected_state=selected_state,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        intermediate_bound_source=intermediate_bound_source,
        refine_external_constraints=refine_external_constraints,
    )
    program.validate()
    return program


def _repeat_box_input_spec(input_spec: InputSpec, *, count: int) -> InputSpec:
    if count < 1:
        raise ValueError("objective branch domain count must be positive")
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    repeats = (count, *(1 for _unused in lower.shape[1:]))
    return InputSpec.box(
        value_name=input_spec.value_name,
        lower=lower.repeat(repeats).contiguous(),
        upper=upper.repeat(repeats).contiguous(),
    )


def _repeat_intervals(
    relu_pre: Mapping[str, IntervalState], *, count: int
) -> dict[str, IntervalState]:
    return {
        name: IntervalState(
            lower=value.lower.repeat(
                (count, *(1 for _unused in value.lower.shape[1:]))
            ).contiguous(),
            upper=value.upper.repeat(
                (count, *(1 for _unused in value.upper.shape[1:]))
            ).contiguous(),
        )
        for name, value in relu_pre.items()
    }


def _materialize_child_splits(
    program: NativeObjectiveBranchProgram,
    candidates: tuple[NativeObjectiveBranchCandidateIR, ...],
) -> dict[str, torch.Tensor]:
    count = 2 * len(candidates)
    splits = {
        name: value.repeat((count, *(1 for _unused in value.shape[1:])))
        .detach()
        .contiguous()
        .clone()
        for name, value in program.selected_state.splits.items()
    }
    for candidate in candidates:
        flat = splits[candidate.relu_input].reshape(count, -1)
        if not bool(
            (
                flat[
                    2 * candidate.ordinal : 2 * candidate.ordinal + 2,
                    candidate.neuron_index,
                ]
                == 0
            ).all()
        ):
            raise ValueError("objective branch candidate was already split")
        flat[2 * candidate.ordinal, candidate.neuron_index] = -1
        flat[2 * candidate.ordinal + 1, candidate.neuron_index] = 1
    return splits


def _evaluate_child_lowers(
    program: NativeObjectiveBranchProgram,
    child_splits: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    count = int(next(iter(child_splits.values())).shape[0])
    all_lowers: list[torch.Tensor] = []
    for start in range(0, count, program.branch_policy.candidate_batch_size * 2):
        stop = min(count, start + program.branch_policy.candidate_batch_size * 2)
        size = stop - start
        split_chunk = {
            name: value[start:stop].contiguous() for name, value in child_splits.items()
        }
        batch_input = _repeat_box_input_spec(program.input_spec, count=size)
        override: Optional[Mapping[str, IntervalState]] = None
        if program.intermediate_bound_source != IntermediateBoundSource.LOCAL_FORWARD:
            override = _repeat_intervals(program.relu_pre, count=size)
        interval_env, child_pre = _optimizer_intermediate_semantics(
            program.module,
            batch_input,
            relu_split_state=split_chunk,
            relu_pre_override=override,
            intermediate_bound_source=program.intermediate_bound_source,
            refine_external_constraints=program.refine_external_constraints,
        )
        alpha = {
            name: value.repeat((size, *(1 for _unused in value.shape[1:])))
            for name, value in program.selected_state.alphas.items()
        }
        beta = {
            name: value.repeat((size, *(1 for _unused in value.shape[1:])))
            for name, value in program.selected_state.betas.items()
        }
        objective = (
            program.objective.unsqueeze(0)
            if program.objective.dim() == 2
            else program.objective
        )
        bounds = _evaluate_state(
            program.module,
            batch_input,
            objective.repeat(size, 1, 1),
            alpha,
            beta,
            interval_env=interval_env,
            relu_pre=child_pre,
            relu_split_state=split_chunk,
            objective="lower",
        )
        if tuple(bounds.lower.shape) != (size, 1):
            raise ValueError("objective branch child lower shape differs")
        all_lowers.append(bounds.lower.detach().contiguous())
    return torch.cat(all_lowers, dim=0)


def execute_native_objective_branch_program(
    program: NativeObjectiveBranchProgram,
    *,
    node_id: str,
) -> NativeObjectiveBranchExecution:
    """Execute the exact five-stage branch score Schedule."""

    if not node_id:
        raise ValueError("objective branch node ID must be non-empty")
    program.validate()
    candidates: Optional[tuple[NativeObjectiveBranchCandidateIR, ...]] = None
    child_splits: Optional[dict[str, torch.Tensor]] = None
    child_lowers: Optional[torch.Tensor] = None
    scores: Optional[tuple[NativeObjectiveBranchScore, ...]] = None
    selected_ordinal: Optional[int] = None
    for action, task in zip(program.schedule.actions, program.task_module.tasks):
        if task.kind == ObjectiveBranchTaskKind.ENUMERATE_CANDIDATES:
            candidates = _enumerate_candidates(
                program.relu_pre,
                program.selected_state.splits,
                policy=program.branch_policy,
            )
        elif task.kind == ObjectiveBranchTaskKind.MATERIALIZE_CHILDREN:
            assert candidates is not None
            child_splits = _materialize_child_splits(program, candidates)
        elif task.kind == ObjectiveBranchTaskKind.EVALUATE_CHILD_BOUNDS:
            assert child_splits is not None
            child_lowers = _evaluate_child_lowers(program, child_splits)
        elif task.kind == ObjectiveBranchTaskKind.REDUCE_WORST_CHILD:
            assert candidates is not None and child_lowers is not None
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
        elif task.kind == ObjectiveBranchTaskKind.SELECT_CANDIDATE:
            assert scores is not None and candidates is not None
            selected_ordinal = min(
                scores,
                key=lambda score: (
                    -score.worst_child_lower,
                    -score.mean_child_lower,
                    candidates[score.candidate_ordinal].relu_input,
                    candidates[score.candidate_ordinal].neuron_index,
                ),
            ).candidate_ordinal
        else:
            raise AssertionError("unreachable objective branch task kind")
        if action.task_id != task.task_id:
            raise ValueError("objective branch runtime Schedule/Task differs")
    assert candidates is not None
    assert child_lowers is not None
    assert scores is not None
    assert selected_ordinal is not None
    selected = candidates[selected_ordinal]
    trace = NativeObjectiveBranchTrace(
        node_id=node_id,
        program_hashes=tuple(sorted(program.hashes().items())),
        action_ids=tuple(action.action_id for action in program.schedule.actions),
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
        program=program,
        trace=trace,
    )
    execution.validate()
    return execution
