"""Objective-branch scorer with compile-owned immutable candidates."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=protected-access,missing-function-docstring
# pylint: disable=not-an-iterable,unsubscriptable-object

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping, Optional, cast

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.branch import NativeObjectiveBranchPlanIR, ObjectiveBranchTaskKind
from ..ir.objective_branch_scorer_ownership import (
    NativeObjectiveBranchScorerScheduleIR,
    NativeObjectiveBranchScorerTaskIRModule,
    NativeValidatedBranchProgramCapsuleIR,
    lower_native_objective_branch_scorer_ir,
)
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    build_native_alpha_beta_scope,
)
from .native_objective_branch_score import (
    NATIVE_OBJECTIVE_BRANCH_PROGRAM_SCHEMA_VERSION,
    NativeObjectiveBranchExecution,
    NativeObjectiveBranchPolicy,
    NativeObjectiveBranchProgram,
    NativeObjectiveBranchScore,
    NativeObjectiveBranchTrace,
    _enumerate_candidates,
    _evaluate_child_lowers,
    _materialize_child_splits,
    _normalize_scalar_objective,
)
from .native_relu_split_bab_runtime import ReluSplitBranch
from .task_executor import InputSpec


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _relu_pre_hash(relu_pre: Mapping[str, IntervalState]) -> str:
    if not relu_pre:
        raise ValueError("prevalidated objective branch ReLU mapping is empty")
    return _canonical_hash(
        {
            name: {
                "lower": tensor_content_hash(value.lower),
                "upper": tensor_content_hash(value.upper),
            }
            for name, value in sorted(relu_pre.items())
        }
    )


@dataclass(frozen=True)
class NativePrevalidatedObjectiveBranchProgram:
    """Exact scorer program admitted once by its immutable capsule."""

    plan: NativeObjectiveBranchPlanIR
    task_module: NativeObjectiveBranchScorerTaskIRModule
    schedule: NativeObjectiveBranchScorerScheduleIR
    capsule: NativeValidatedBranchProgramCapsuleIR
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
        self.capsule.validate(
            plan=self.plan,
            task_module=self.task_module,
            schedule=self.schedule,
        )
        self.selected_state.validate()
        self.optimizer_policy.validate()
        self.branch_policy.validate()
        objective = _normalize_scalar_objective(self.objective)
        splits = self.selected_state.splits
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
            or self.plan.state_scope_hash != self.selected_state.scope.stable_hash()
            or self.plan.policy_hash != self.branch_policy.stable_hash()
            or self.plan.candidate_policy_id != self.branch_policy.candidate_policy_id
            or self.plan.candidates_per_relu != self.branch_policy.candidates_per_relu
            or self.plan.candidate_batch_size != self.branch_policy.candidate_batch_size
            or self.plan.max_candidates != self.branch_policy.max_candidates
            or self.plan.intermediate_bound_source != expected_intermediate
            or self.plan.reduce_policy != self.branch_policy.reduce_policy
            or self.selected_state.scope.optimizer_policy_hash
            != self.optimizer_policy.stable_hash()
            or self.selected_state.scope.objective_hash
            != tensor_content_hash(objective)
            or self.selected_state.scope.intermediate_bounds_hash
            != _relu_pre_hash(self.relu_pre)
            or tuple(self.relu_pre) != tuple(splits)
            or self.capsule.objective_hash != tensor_content_hash(objective)
            or self.capsule.relu_pre_hash != _relu_pre_hash(self.relu_pre)
            or self.capsule.split_state_hash != relu_split_state_hash(splits)
            or self.capsule.selected_state_hash != self.selected_state.stable_hash()
            or self.capsule.state_scope_hash != self.selected_state.scope.stable_hash()
            or self.capsule.optimizer_policy_hash != self.optimizer_policy.stable_hash()
            or self.capsule.branch_policy_hash != self.branch_policy.stable_hash()
        ):
            raise ValueError("prevalidated objective branch program identity differs")

    def hashes(self) -> dict[str, str]:
        self.validate()
        return {
            "branch_plan_hash": self.capsule.branch_plan_hash,
            "branch_task_module_hash": self.capsule.branch_task_module_hash,
            "branch_schedule_hash": self.capsule.branch_schedule_hash,
        }


def compile_native_prevalidated_objective_branch_program(
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
) -> NativePrevalidatedObjectiveBranchProgram:
    """Enumerate once, then seal Plan/Task/Schedule and tensor identities."""

    if not plan_id:
        raise ValueError("prevalidated objective branch plan ID must be non-empty")
    module.validate()
    selected_state.validate()
    optimizer_policy.validate()
    branch_policy.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("prevalidated objective branch intermediate source is invalid")
    if not isinstance(refine_external_constraints, bool):
        raise TypeError("prevalidated objective branch refinement flag is invalid")
    if refine_external_constraints and intermediate_bound_source != (
        IntermediateBoundSource.EXTERNAL_VERIFIER
    ):
        raise ValueError("prevalidated branch refinement requires external provenance")
    objective = _normalize_scalar_objective(linear_spec_C)
    scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_pre=relu_pre,
        relu_split_state=selected_state.splits,
        policy=optimizer_policy,
    )
    if scope != selected_state.scope:
        raise ValueError("prevalidated objective branch selected scope differs")

    # This is the sole candidate enumeration in the new scorer lifecycle.
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
    task_module, schedule = lower_native_objective_branch_scorer_ir(plan)
    capsule = NativeValidatedBranchProgramCapsuleIR(
        capsule_id=f"{plan_id}:validated-capsule",
        branch_plan_hash=plan.stable_hash(),
        branch_task_module_hash=task_module.stable_hash(plan=plan),
        branch_schedule_hash=schedule.stable_hash(plan=plan, task_module=task_module),
        objective_hash=tensor_content_hash(objective),
        relu_pre_hash=_relu_pre_hash(relu_pre),
        split_state_hash=plan.split_state_hash,
        selected_state_hash=plan.selected_state_hash,
        state_scope_hash=plan.state_scope_hash,
        optimizer_policy_hash=optimizer_policy.stable_hash(),
        branch_policy_hash=branch_policy.stable_hash(),
        candidate_table_hash=_canonical_hash(
            [candidate.to_dict() for candidate in candidates]
        ),
        candidate_count=len(candidates),
        intermediate_bound_source=plan.intermediate_bound_source,
        refine_external_constraints=refine_external_constraints,
        compile_enumeration_count=1,
        execute_enumeration_count=0,
        semantic_token="0" * 64,
    )
    capsule = replace(capsule, semantic_token=_canonical_hash(capsule.semantic_dict()))
    program = NativePrevalidatedObjectiveBranchProgram(
        plan=plan,
        task_module=task_module,
        schedule=schedule,
        capsule=capsule,
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


def execute_native_prevalidated_objective_branch_program(
    program: NativePrevalidatedObjectiveBranchProgram,
    *,
    node_id: str,
) -> NativeObjectiveBranchExecution:
    """Execute five stages while consuming, never regenerating, candidates."""

    if not node_id:
        raise ValueError("prevalidated objective branch node ID must be non-empty")
    program.validate()
    candidates = None
    child_splits: Optional[dict[str, torch.Tensor]] = None
    child_lowers: Optional[torch.Tensor] = None
    scores: Optional[tuple[NativeObjectiveBranchScore, ...]] = None
    selected_ordinal: Optional[int] = None
    legacy_program = cast(NativeObjectiveBranchProgram, program)
    for action, task in zip(program.schedule.actions, program.task_module.tasks):
        if task.kind == ObjectiveBranchTaskKind.ENUMERATE_CANDIDATES:
            candidates = program.plan.candidates
        elif task.kind == ObjectiveBranchTaskKind.MATERIALIZE_CHILDREN:
            assert candidates is not None
            child_splits = _materialize_child_splits(legacy_program, candidates)
        elif task.kind == ObjectiveBranchTaskKind.EVALUATE_CHILD_BOUNDS:
            assert child_splits is not None
            child_lowers = _evaluate_child_lowers(legacy_program, child_splits)
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
            raise AssertionError("unreachable prevalidated objective branch task kind")
        if action.task_id != task.task_id:
            raise ValueError("prevalidated objective branch Schedule/Task differs")
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
        program=legacy_program,
        trace=trace,
    )
    execution.validate()
    return execution


__all__ = [
    "NativePrevalidatedObjectiveBranchProgram",
    "compile_native_prevalidated_objective_branch_program",
    "execute_native_prevalidated_objective_branch_program",
]
