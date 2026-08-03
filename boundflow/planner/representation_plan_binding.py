"""Bind native representation Plan decisions to an executable Bound program."""

# The binder deliberately verifies every cross-layer identity in one place.
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-instance-attributes,too-few-public-methods
# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Protocol, TypeVar

from ..ir.bound import (
    BFBoundModule,
    BoundOp,
    BoundOpKind,
    BoundRepresentation,
    BoundValue,
    BoundValueRole,
    RepresentationChangeAttrs,
)
from ..ir.bound_rewrite import rewrite_plain_crown_structured_regions
from ..ir.plan import (
    BackendCandidate,
    MaterializationCandidate,
    PlanCost,
    PlanInstance,
    PlanProvenance,
    PlanTemplate,
    RepresentationCandidate,
    TransitionKind,
)
from ..ir.schedule import MaterializeAction, ScheduleModule
from .storage_plan_variants import build_native_storage_plan_variants

REPRESENTATION_BINDING_SCHEMA_VERSION = "boundflow.native-representation-binding/v1"
NATIVE_REPRESENTATION_POLICY_COMPILER_VERSION = (
    "boundflow.native-representation-policy/v1"
)
DENSE_POLICY_ID = "native-dense-v1"
STRUCTURED_AFFINE_POLICY_ID = "native-structured-affine-v1"


@dataclass(frozen=True)
class RepresentationBindingEvent:
    """One selected Plan transition bound to Schedule and execution Bound IR."""

    transition_candidate_id: str
    schedule_action_id: str
    execution_op_id: str
    execution_output_value_id: str
    source_value_id: str
    before_op_id: str
    transition_kind: TransitionKind
    source_representation: BoundRepresentation
    target_representation: BoundRepresentation

    def validate(self) -> None:
        for name in (
            "transition_candidate_id",
            "schedule_action_id",
            "execution_op_id",
            "execution_output_value_id",
            "source_value_id",
            "before_op_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"representation binding {name} is empty")
        if self.source_representation == self.target_representation:
            raise ValueError("representation binding event does not change layout")
        if (
            self.transition_kind == TransitionKind.MATERIALIZE
            and self.target_representation != BoundRepresentation.DENSE
        ):
            raise ValueError("representation binding materialization is not dense")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "transition_candidate_id": self.transition_candidate_id,
            "schedule_action_id": self.schedule_action_id,
            "execution_op_id": self.execution_op_id,
            "execution_output_value_id": self.execution_output_value_id,
            "source_value_id": self.source_value_id,
            "before_op_id": self.before_op_id,
            "transition_kind": self.transition_kind.value,
            "source_representation": self.source_representation.value,
            "target_representation": self.target_representation.value,
        }


@dataclass(frozen=True)
class RepresentationBindingTrace:
    """Replay-grade identity joining source planning to execution Bound IR."""

    source_bound_module_hash: str
    source_plan_template_hash: str
    source_plan_instance_hash: str
    source_schedule_hash: str
    execution_bound_module_hash: str
    policy_id: str
    selected_representation_candidate_ids: tuple[str, ...]
    selected_transition_candidate_ids: tuple[str, ...]
    events: tuple[RepresentationBindingEvent, ...]
    schema_version: str = REPRESENTATION_BINDING_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != REPRESENTATION_BINDING_SCHEMA_VERSION:
            raise ValueError("unsupported representation binding schema")
        for name in (
            "source_bound_module_hash",
            "source_plan_template_hash",
            "source_plan_instance_hash",
            "source_schedule_hash",
            "execution_bound_module_hash",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"representation binding {name} is not SHA-256")
        if self.policy_id not in {DENSE_POLICY_ID, STRUCTURED_AFFINE_POLICY_ID}:
            raise ValueError("representation binding selected an unknown policy")
        if len(self.selected_representation_candidate_ids) != len(
            set(self.selected_representation_candidate_ids)
        ):
            raise ValueError("representation binding repeats a representation")
        if len(self.selected_transition_candidate_ids) != len(
            set(self.selected_transition_candidate_ids)
        ):
            raise ValueError("representation binding repeats a transition")
        for event in self.events:
            event.validate()
        event_transition_ids = tuple(
            event.transition_candidate_id for event in self.events
        )
        if event_transition_ids != self.selected_transition_candidate_ids:
            raise ValueError("representation binding event/transition order differs")
        if self.policy_id == DENSE_POLICY_ID:
            if self.events or self.selected_transition_candidate_ids:
                raise ValueError("dense representation binding contains transitions")
            if self.source_bound_module_hash != self.execution_bound_module_hash:
                raise ValueError("dense representation binding rewrote Bound IR")
        elif not self.events:
            raise ValueError("structured representation binding has no transitions")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "source_bound_module_hash": self.source_bound_module_hash,
            "source_plan_template_hash": self.source_plan_template_hash,
            "source_plan_instance_hash": self.source_plan_instance_hash,
            "source_schedule_hash": self.source_schedule_hash,
            "execution_bound_module_hash": self.execution_bound_module_hash,
            "policy_id": self.policy_id,
            "selected_representation_candidate_ids": list(
                self.selected_representation_candidate_ids
            ),
            "selected_transition_candidate_ids": list(
                self.selected_transition_candidate_ids
            ),
            "events": [event.to_dict() for event in self.events],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BoundRepresentationBinding:
    """Executable Bound module plus its exact source-plan binding trace."""

    execution_bound_module: BFBoundModule
    trace: RepresentationBindingTrace

    def validate(self) -> None:
        self.execution_bound_module.validate()
        self.trace.validate()
        if (
            self.execution_bound_module.stable_hash()
            != self.trace.execution_bound_module_hash
        ):
            raise ValueError("representation binding execution Bound hash differs")


def build_native_representation_plan_variants(
    template: PlanTemplate,
    *,
    bound_module: BFBoundModule,
) -> PlanTemplate:
    """Add globally coherent dense and structured-affine source Plan choices."""

    template.validate(bound_module=bound_module)
    if len(template.storage_candidates) != 1:
        raise ValueError("native representation policies require one baseline storage")
    if len(template.region_candidates) != len(bound_module.graph.ops):
        raise ValueError("native representation policies require singleton regions")
    dense_by_region = _one_by_region(
        template.representation_candidates,
        label="dense representation",
    )
    backend_by_region = _one_by_region(
        template.backend_candidates,
        label="dense backend",
    )
    if any(
        candidate.representation != BoundRepresentation.DENSE
        for candidate in dense_by_region.values()
    ):
        raise ValueError("native representation source template is not all-dense")

    rewritten = rewrite_plain_crown_structured_regions(bound_module)
    transition_ops = _execution_transition_ops(rewritten)
    transition_candidates = tuple(
        MaterializationCandidate(
            candidate_id=(f"transition:{STRUCTURED_AFFINE_POLICY_ID}:{op.op_id}"),
            source_value_id=signature[1],
            before_op_id=signature[2],
            kind=signature[0],
            source_representation=signature[3],
            target_representation=signature[4],
            static_legal=True,
            rejection_reasons=(),
            cost=_correctness_cost(),
        )
        for op, signature in transition_ops
    )
    transition_ids_by_before: dict[str, list[str]] = {}
    for candidate in transition_candidates:
        transition_ids_by_before.setdefault(candidate.before_op_id, []).append(
            candidate.candidate_id
        )

    rewritten_ops = {op.op_id: op for op in rewritten.graph.ops}
    rewritten_values = {value.value_id: value for value in rewritten.graph.values}
    structured_representations: list[RepresentationCandidate] = []
    structured_backends: list[BackendCandidate] = []
    source_ops = {op.op_id: op for op in bound_module.graph.ops}
    for region in template.region_candidates:
        if len(region.op_ids) != 1:
            raise ValueError("native representation policy found a fused region")
        op_id = region.op_ids[0]
        execution_op = rewritten_ops[op_id]
        representation = _execution_op_representation(
            execution_op, values=rewritten_values
        )
        representation_candidate = RepresentationCandidate(
            candidate_id=(
                f"representation:{STRUCTURED_AFFINE_POLICY_ID}:{region.region_id}"
            ),
            region_id=region.region_id,
            representation=representation,
            required_transition_candidate_ids=tuple(
                sorted(transition_ids_by_before.get(op_id, ()))
            ),
            static_legal=True,
            rejection_reasons=(),
            cost=_correctness_cost(),
        )
        structured_representations.append(representation_candidate)
        dense_backend = backend_by_region[region.region_id]
        structured_backends.append(
            replace(
                dense_backend,
                candidate_id=(
                    f"backend:{STRUCTURED_AFFINE_POLICY_ID}:{region.region_id}"
                ),
                compatible_representation_candidate_ids=(
                    representation_candidate.candidate_id,
                ),
            )
        )
        if op_id not in source_ops:
            raise ValueError("native representation region references unknown op")

    supported_representations = tuple(
        dict.fromkeys(
            (
                BoundRepresentation.DENSE,
                *(item.representation for item in structured_representations),
            )
        )
    )
    capabilities = tuple(
        replace(
            capability,
            supported_representations=supported_representations,
        )
        for capability in template.capabilities
    )
    all_representations = (
        *template.representation_candidates,
        *structured_representations,
    )
    baseline_storage = replace(
        template.storage_candidates[0],
        compatible_representation_candidate_ids=tuple(
            item.candidate_id for item in all_representations
        ),
    )
    expanded = replace(
        template,
        capabilities=capabilities,
        representation_candidates=all_representations,
        materialization_candidates=transition_candidates,
        backend_candidates=(*template.backend_candidates, *structured_backends),
        storage_candidates=(baseline_storage,),
        provenance=(
            *template.provenance,
            PlanProvenance(
                "representation_policy_compiler",
                NATIVE_REPRESENTATION_POLICY_COMPILER_VERSION,
            ),
            PlanProvenance("representation_policy_scope", "global_two_policy_v1"),
            PlanProvenance("representation_performance_claim", "forbidden"),
        ),
    )
    expanded = _rehash_template(expanded, parent_hash=template.planner_config_hash)
    expanded.validate(bound_module=bound_module)
    expanded = build_native_storage_plan_variants(expanded, bound_module=bound_module)

    dense_ids = tuple(
        dense_by_region[region.region_id].candidate_id
        for region in template.region_candidates
    )
    structured_ids = tuple(item.candidate_id for item in structured_representations)
    storage = tuple(
        replace(
            candidate,
            compatible_representation_candidate_ids=(
                dense_ids
                if candidate.candidate_id == "storage:native-retain-all-v1"
                else structured_ids
            ),
        )
        for candidate in expanded.storage_candidates
    )
    result = replace(expanded, storage_candidates=storage)
    result = _rehash_template(result, parent_hash=expanded.planner_config_hash)
    result.validate(bound_module=bound_module)
    return result


def bind_native_representation_plan(
    bound_module: BFBoundModule,
    *,
    template: PlanTemplate,
    instance: PlanInstance,
    schedule: ScheduleModule,
) -> BoundRepresentationBinding:
    """Turn one exact selected representation policy into execution Bound IR."""

    instance.validate(template=template, bound_module=bound_module)
    schedule.validate(bound_module=bound_module, template=template, instance=instance)
    selected_representation_ids = tuple(
        decision.candidate_id for decision in instance.representation_decisions
    )
    selected_transition_ids = tuple(
        decision.candidate_id for decision in instance.materialization_decisions
    )
    regions = {item.region_id: item for item in template.region_candidates}
    representations = {
        item.candidate_id: item for item in template.representation_candidates
    }
    dense_ids = tuple(
        next(
            candidate.candidate_id
            for candidate in template.representation_candidates
            if candidate.region_id == region.region_id
            and candidate.candidate_id.startswith("representation:dense:")
        )
        for region in template.region_candidates
    )
    structured_ids = tuple(
        next(
            candidate.candidate_id
            for candidate in template.representation_candidates
            if candidate.region_id == region.region_id
            and candidate.candidate_id.startswith(
                f"representation:{STRUCTURED_AFFINE_POLICY_ID}:"
            )
        )
        for region in template.region_candidates
    )
    selected_set = set(selected_representation_ids)
    if selected_set == set(dense_ids):
        policy_id = DENSE_POLICY_ID
        execution_module = bound_module
    elif selected_set == set(structured_ids):
        policy_id = STRUCTURED_AFFINE_POLICY_ID
        execution_module = rewrite_plain_crown_structured_regions(bound_module)
    else:
        raise ValueError("native representation Plan selected a mixed policy")
    if len(selected_set) != len(regions):
        raise ValueError("native representation policy does not cover every region")
    if any(candidate_id not in representations for candidate_id in selected_set):
        raise ValueError("native representation policy selects an unknown candidate")

    transitions = {
        item.candidate_id: item for item in template.materialization_candidates
    }
    actions = {
        action.transition_candidate_id: action
        for action in schedule.actions
        if isinstance(action, MaterializeAction)
    }
    if set(actions) != set(selected_transition_ids):
        raise ValueError("source Schedule materialization set differs from Plan")
    events: list[RepresentationBindingEvent] = []
    if policy_id == DENSE_POLICY_ID:
        if selected_transition_ids:
            raise ValueError("dense representation Plan selected transitions")
    else:
        execution_by_signature = {
            signature: op
            for op, signature in _execution_transition_ops(execution_module)
        }
        if len(execution_by_signature) != len(
            _execution_transition_ops(execution_module)
        ):
            raise ValueError("execution Bound rewrite repeats a transition signature")
        selected_candidates = tuple(
            transitions[item] for item in selected_transition_ids
        )
        candidate_signatures = {
            _candidate_signature(candidate): candidate
            for candidate in selected_candidates
        }
        if len(candidate_signatures) != len(selected_candidates):
            raise ValueError("selected Plan repeats a transition signature")
        if set(candidate_signatures) != set(execution_by_signature):
            raise ValueError("selected Plan transitions differ from Bound rewrite")
        for candidate_id in selected_transition_ids:
            candidate = transitions[candidate_id]
            signature = _candidate_signature(candidate)
            execution_op = execution_by_signature[signature]
            action = actions[candidate_id]
            if not _action_matches_candidate(action, candidate):
                raise ValueError("source Schedule action differs from Plan transition")
            events.append(
                RepresentationBindingEvent(
                    transition_candidate_id=candidate_id,
                    schedule_action_id=action.action_id,
                    execution_op_id=execution_op.op_id,
                    execution_output_value_id=execution_op.outputs[0],
                    source_value_id=candidate.source_value_id,
                    before_op_id=candidate.before_op_id,
                    transition_kind=candidate.kind,
                    source_representation=candidate.source_representation,
                    target_representation=candidate.target_representation,
                )
            )

    trace = RepresentationBindingTrace(
        source_bound_module_hash=bound_module.stable_hash(),
        source_plan_template_hash=template.stable_hash(bound_module=bound_module),
        source_plan_instance_hash=instance.stable_hash(
            template=template, bound_module=bound_module
        ),
        source_schedule_hash=schedule.stable_hash(
            bound_module=bound_module, template=template, instance=instance
        ),
        execution_bound_module_hash=execution_module.stable_hash(),
        policy_id=policy_id,
        selected_representation_candidate_ids=selected_representation_ids,
        selected_transition_candidate_ids=selected_transition_ids,
        events=tuple(events),
    )
    result = BoundRepresentationBinding(execution_module, trace)
    result.validate()
    return result


class _RegionCandidateLike(Protocol):
    @property
    def region_id(self) -> str:
        """Return the logical region identity."""


T = TypeVar("T", bound=_RegionCandidateLike)


def _one_by_region(
    candidates: tuple[T, ...], *, label: str
) -> dict[str, T]:
    result: dict[str, T] = {}
    for candidate in candidates:
        region_id = getattr(candidate, "region_id")
        if region_id in result:
            raise ValueError(f"native source template repeats a {label} region")
        result[region_id] = candidate
    return result


def _correctness_cost() -> PlanCost:
    return PlanCost(
        predicted_latency_ms=0.0,
        predicted_peak_bytes=0,
        compile_cost_ms=0.0,
        setup_cost_ms=0.0,
        confidence=1.0,
        risk_tags=("correctness_only", "no_performance_claim"),
    )


def _execution_op_representation(
    op: BoundOp, *, values: dict[str, BoundValue]
) -> BoundRepresentation:
    coefficient_representations = {
        values[value_id].representation
        for value_id in (*op.inputs, *op.outputs)
        if values[value_id].role == BoundValueRole.COEFFICIENT
    }
    if BoundRepresentation.STRUCTURED in coefficient_representations:
        return BoundRepresentation.STRUCTURED
    return BoundRepresentation.DENSE


TransitionSignature = tuple[
    TransitionKind,
    str,
    str,
    BoundRepresentation,
    BoundRepresentation,
]


def _execution_transition_ops(
    module: BFBoundModule,
) -> tuple[tuple[BoundOp, TransitionSignature], ...]:
    users: dict[str, list[str]] = {}
    for op in module.graph.ops:
        for value_id in op.inputs:
            users.setdefault(value_id, []).append(op.op_id)
    result: list[tuple[BoundOp, TransitionSignature]] = []
    for op in module.graph.ops:
        if op.kind not in {
            BoundOpKind.REPRESENTATION_CAST,
            BoundOpKind.MATERIALIZE,
        }:
            continue
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise ValueError("execution representation transition is not unary")
        consumers = users.get(op.outputs[0], ())
        if len(consumers) != 1:
            raise ValueError("execution representation transition has ambiguous use")
        attrs = op.attrs
        if not isinstance(attrs, RepresentationChangeAttrs):
            raise TypeError("execution transition lacks representation attrs")
        kind = (
            TransitionKind.CAST
            if op.kind == BoundOpKind.REPRESENTATION_CAST
            else TransitionKind.MATERIALIZE
        )
        result.append(
            (
                op,
                (
                    kind,
                    op.inputs[0],
                    consumers[0],
                    attrs.source,
                    attrs.target,
                ),
            )
        )
    return tuple(result)


def _candidate_signature(candidate: MaterializationCandidate) -> TransitionSignature:
    return (
        candidate.kind,
        candidate.source_value_id,
        candidate.before_op_id,
        candidate.source_representation,
        candidate.target_representation,
    )


def _action_matches_candidate(
    action: MaterializeAction, candidate: MaterializationCandidate
) -> bool:
    return (
        action.source_value_id == candidate.source_value_id
        and action.before_op_id == candidate.before_op_id
        and action.source_representation == candidate.source_representation
        and action.target_representation == candidate.target_representation
    )


def _rehash_template(template: PlanTemplate, *, parent_hash: str) -> PlanTemplate:
    payload = json.dumps(
        {
            "parent_planner_config_hash": parent_hash,
            "compiler": NATIVE_REPRESENTATION_POLICY_COMPILER_VERSION,
            "representations": [
                item.to_dict() for item in template.representation_candidates
            ],
            "transitions": [
                item.to_dict() for item in template.materialization_candidates
            ],
            "backends": [item.to_dict() for item in template.backend_candidates],
            "storage": [item.to_dict() for item in template.storage_candidates],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    planner_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    template_seed = (
        f"{template.bound_module_hash}:{planner_hash}:"
        f"{template.hardware.profile_id}:{template.workload.profile_id}"
    )
    return replace(
        template,
        template_id=(
            "plan-template:"
            + hashlib.sha256(template_seed.encode("utf-8")).hexdigest()[:20]
        ),
        planner_config_hash=planner_hash,
    )


__all__ = [
    "DENSE_POLICY_ID",
    "NATIVE_REPRESENTATION_POLICY_COMPILER_VERSION",
    "REPRESENTATION_BINDING_SCHEMA_VERSION",
    "STRUCTURED_AFFINE_POLICY_ID",
    "BoundRepresentationBinding",
    "RepresentationBindingEvent",
    "RepresentationBindingTrace",
    "bind_native_representation_plan",
    "build_native_representation_plan_variants",
]
