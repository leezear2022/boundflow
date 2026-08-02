"""Typed, deterministic reference builder for Plan IR v1 templates."""

# Candidate construction performs deliberate cross-axis normalization in one place.
# pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals,too-many-statements

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Optional, Tuple

from ..ir.bound import (
    BFBoundModule,
    BoundOp,
    BoundOpKind,
    BoundRepresentation,
)
from ..ir.plan import (
    BackendCandidate,
    BackendCapabilitySpec,
    BatchCandidate,
    HardwareProfile,
    MaterializationCandidate,
    PlanCost,
    PlanProvenance,
    PlanTemplate,
    RegionCandidate,
    RegionKind,
    RepresentationCandidate,
    StateAction,
    StateCandidate,
    StorageBinding,
    StorageCandidate,
    TransitionKind,
    WorkloadProfile,
)


@dataclass(frozen=True)
class RegionEvidence:
    """Measured or modeled evidence for one contiguous Bound IR region."""

    evidence_id: str
    op_ids: Tuple[str, ...]
    cost: PlanCost


@dataclass(frozen=True)
class TransitionEvidence:
    """Evidence for one explicit representation boundary."""

    evidence_id: str
    source_value_id: str
    before_op_id: str
    kind: TransitionKind
    source_representation: BoundRepresentation
    target_representation: BoundRepresentation
    cost: PlanCost
    rejection_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RepresentationEvidence:
    """Evidence for one representation of one region candidate."""

    evidence_id: str
    region_evidence_id: str
    representation: BoundRepresentation
    required_transition_evidence_ids: Tuple[str, ...]
    cost: PlanCost
    rejection_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendEvidence:
    """Evidence for one backend/representation pairing."""

    evidence_id: str
    region_evidence_id: str
    representation_evidence_id: str
    capability_id: str
    cost: PlanCost
    compiled_artifact_key: Optional[str] = None
    rejection_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchEvidence:
    """Evidence for independent domain/spec/sample batching."""

    evidence_id: str
    domain_batch_size: int
    spec_batch_size: int
    sample_batch_size: int
    estimated_payload_bytes: int
    cost: PlanCost
    rejection_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ValueLayoutEvidence:
    """Non-dense physical layout override for one Bound IR value."""

    value_id: str
    representation: BoundRepresentation
    physical_size_bytes: int


@dataclass(frozen=True)
class StorageEvidence:
    """Evidence for a whole-plan, statically allocated storage policy."""

    evidence_id: str
    compatible_batch_evidence_ids: Tuple[str, ...]
    compatible_representation_evidence_ids: Tuple[str, ...]
    value_layout_overrides: Tuple[ValueLayoutEvidence, ...]
    arena_id: str
    cost: PlanCost
    rejection_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StateEvidence:
    """Evidence for one cache/recompute/evict alternative."""

    evidence_id: str
    state_id: str
    source_value_id: str
    action: StateAction
    size_bytes: int
    cost: PlanCost
    rejection_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferencePlanEvidence:
    """Complete typed evidence set consumed by the reference builder."""

    evidence_set_id: str
    regions: Tuple[RegionEvidence, ...]
    transitions: Tuple[TransitionEvidence, ...]
    representations: Tuple[RepresentationEvidence, ...]
    backends: Tuple[BackendEvidence, ...]
    batches: Tuple[BatchEvidence, ...]
    storage: Tuple[StorageEvidence, ...]
    states: Tuple[StateEvidence, ...] = ()
    provenance: Tuple[PlanProvenance, ...] = ()


def build_reference_plan_template(
    bound_module: BFBoundModule,
    *,
    hardware: HardwareProfile,
    workload: WorkloadProfile,
    capabilities: Tuple[BackendCapabilitySpec, ...],
    evidence: ReferencePlanEvidence,
) -> PlanTemplate:
    """Build and verify a deterministic PlanTemplate from typed evidence."""

    bound_module.validate()
    hardware.validate()
    workload.validate()
    _validate_evidence_identity(evidence)

    graph = bound_module.graph
    ops = {op.op_id: op for op in graph.ops}
    op_index = {op.op_id: index for index, op in enumerate(graph.ops)}
    producer = {value_id: op.op_id for op in graph.ops for value_id in op.outputs}
    users: dict[str, set[str]] = {}
    for op in graph.ops:
        for value_id in op.inputs:
            users.setdefault(value_id, set()).add(op.op_id)

    regions = tuple(
        _build_region_candidate(
            item,
            ops=ops,
            op_index=op_index,
            producer=producer,
            users=users,
            graph_outputs=set(graph.outputs),
        )
        for item in evidence.regions
    )
    region_by_evidence = {
        item.evidence_id: region for item, region in zip(evidence.regions, regions)
    }

    transitions = tuple(
        MaterializationCandidate(
            candidate_id=_candidate_id("transition", item.evidence_id),
            source_value_id=item.source_value_id,
            before_op_id=item.before_op_id,
            kind=item.kind,
            source_representation=item.source_representation,
            target_representation=item.target_representation,
            static_legal=not item.rejection_reasons,
            rejection_reasons=item.rejection_reasons,
            cost=item.cost,
        )
        for item in evidence.transitions
    )
    transition_id_by_evidence = {
        item.evidence_id: candidate.candidate_id
        for item, candidate in zip(evidence.transitions, transitions)
    }

    representations = tuple(
        RepresentationCandidate(
            candidate_id=_candidate_id("representation", item.evidence_id),
            region_id=_lookup_region(
                region_by_evidence, item.region_evidence_id
            ).region_id,
            representation=item.representation,
            required_transition_candidate_ids=tuple(
                _lookup_id(
                    transition_id_by_evidence,
                    transition_id,
                    label="transition evidence",
                )
                for transition_id in item.required_transition_evidence_ids
            ),
            static_legal=not item.rejection_reasons,
            rejection_reasons=item.rejection_reasons,
            cost=item.cost,
        )
        for item in evidence.representations
    )
    representation_by_evidence = {
        item.evidence_id: candidate
        for item, candidate in zip(evidence.representations, representations)
    }

    capability_by_id = {
        capability.capability_id: capability for capability in capabilities
    }
    backends = tuple(
        _build_backend_candidate(
            item,
            region_by_evidence=region_by_evidence,
            representation_by_evidence=representation_by_evidence,
            capability_by_id=capability_by_id,
            workload=workload,
            ops=ops,
        )
        for item in evidence.backends
    )

    batches = tuple(
        BatchCandidate(
            candidate_id=_candidate_id("batch", item.evidence_id),
            domain_batch_size=item.domain_batch_size,
            spec_batch_size=item.spec_batch_size,
            sample_batch_size=item.sample_batch_size,
            estimated_payload_bytes=item.estimated_payload_bytes,
            static_legal=not item.rejection_reasons,
            rejection_reasons=item.rejection_reasons,
            cost=item.cost,
        )
        for item in evidence.batches
    )
    batch_id_by_evidence = {
        item.evidence_id: candidate.candidate_id
        for item, candidate in zip(evidence.batches, batches)
    }
    representation_id_by_evidence = {
        item.evidence_id: candidate.candidate_id
        for item, candidate in zip(evidence.representations, representations)
    }

    storage = tuple(
        _build_storage_candidate(
            item,
            bound_module=bound_module,
            hardware=hardware,
            batch_id_by_evidence=batch_id_by_evidence,
            representation_id_by_evidence=representation_id_by_evidence,
        )
        for item in evidence.storage
    )
    states = tuple(
        _build_state_candidate(item, bound_module=bound_module)
        for item in evidence.states
    )

    planner_config_hash = _evidence_hash(evidence)
    bound_hash = bound_module.stable_hash()
    template_seed = (
        f"{bound_hash}:{planner_config_hash}:{hardware.profile_id}:"
        f"{workload.profile_id}"
    )
    template = PlanTemplate(
        template_id="plan-template:"
        + hashlib.sha256(template_seed.encode("utf-8")).hexdigest()[:20],
        bound_module_hash=bound_hash,
        planner_config_hash=planner_config_hash,
        hardware=hardware,
        workload=workload,
        capabilities=capabilities,
        region_candidates=regions,
        representation_candidates=representations,
        materialization_candidates=transitions,
        backend_candidates=backends,
        batch_candidates=batches,
        storage_candidates=storage,
        state_candidates=states,
        provenance=(
            PlanProvenance("builder", "boundflow.reference-plan-builder/v1"),
            PlanProvenance("evidence_set_id", evidence.evidence_set_id),
            *evidence.provenance,
        ),
    )
    template.validate(bound_module=bound_module)
    return template


def _validate_evidence_identity(evidence: ReferencePlanEvidence) -> None:
    if not evidence.evidence_set_id:
        raise ValueError("evidence_set_id must be non-empty")
    required = (
        evidence.regions,
        evidence.representations,
        evidence.backends,
        evidence.batches,
        evidence.storage,
    )
    if any(not group for group in required):
        raise ValueError("reference plan evidence has an empty required group")
    for group in (
        evidence.regions,
        evidence.transitions,
        evidence.representations,
        evidence.backends,
        evidence.batches,
        evidence.storage,
        evidence.states,
    ):
        ids = tuple(item.evidence_id for item in group)
        if any(not evidence_id for evidence_id in ids):
            raise ValueError("plan evidence IDs must be non-empty")
        if len(ids) != len(set(ids)):
            raise ValueError("plan evidence IDs must be unique within each category")
    provenance_keys = ("builder", "evidence_set_id")
    if any(item.key in provenance_keys for item in evidence.provenance):
        raise ValueError("evidence provenance uses a reserved builder key")


def _build_region_candidate(
    item: RegionEvidence,
    *,
    ops: dict[str, BoundOp],
    op_index: dict[str, int],
    producer: dict[str, str],
    users: dict[str, set[str]],
    graph_outputs: set[str],
) -> RegionCandidate:
    if not item.op_ids:
        raise ValueError("region evidence requires op_ids")
    if any(op_id not in ops for op_id in item.op_ids):
        raise ValueError("region evidence references an unknown Bound IR op")
    indices = tuple(op_index[op_id] for op_id in item.op_ids)
    if indices != tuple(range(indices[0], indices[0] + len(indices))):
        raise ValueError("region evidence must identify a contiguous topological span")
    region_ops = set(item.op_ids)
    input_value_ids = tuple(
        value_id
        for op_id in item.op_ids
        for value_id in getattr(ops[op_id], "inputs")
        if producer.get(value_id) not in region_ops
    )
    output_value_ids = tuple(
        value_id
        for op_id in item.op_ids
        for value_id in getattr(ops[op_id], "outputs")
        if value_id in graph_outputs
        or any(user not in region_ops for user in users.get(value_id, set()))
    )
    op_kinds = tuple(getattr(ops[op_id], "kind") for op_id in item.op_ids)
    return RegionCandidate(
        candidate_id=_candidate_id("region", item.evidence_id),
        region_id=_candidate_id("logical-region", item.evidence_id),
        kind=_region_kind(op_kinds),
        op_ids=item.op_ids,
        input_value_ids=_deduplicate(input_value_ids),
        output_value_ids=_deduplicate(output_value_ids),
        fused=len(item.op_ids) > 1,
        cost=item.cost,
    )


def _build_backend_candidate(
    item: BackendEvidence,
    *,
    region_by_evidence: dict[str, RegionCandidate],
    representation_by_evidence: dict[str, RepresentationCandidate],
    capability_by_id: dict[str, BackendCapabilitySpec],
    workload: WorkloadProfile,
    ops: dict[str, BoundOp],
) -> BackendCandidate:
    region = _lookup_region(region_by_evidence, item.region_evidence_id)
    representation = representation_by_evidence.get(item.representation_evidence_id)
    if representation is None:
        raise ValueError("backend evidence references unknown representation evidence")
    if representation.region_id != region.region_id:
        raise ValueError("backend evidence region/representation mismatch")
    capability = capability_by_id.get(item.capability_id)
    if capability is None:
        raise ValueError("backend evidence references unknown capability")
    capability_rejections = capability.rejection_reasons(
        workload=workload,
        op_kinds=tuple(getattr(ops[op_id], "kind") for op_id in region.op_ids),
        representation=representation.representation,
    )
    rejection_reasons = _deduplicate((*item.rejection_reasons, *capability_rejections))
    return BackendCandidate(
        candidate_id=_candidate_id("backend", item.evidence_id),
        region_id=region.region_id,
        backend=capability.backend,
        capability_id=capability.capability_id,
        compatible_representation_candidate_ids=(representation.candidate_id,),
        compiled_artifact_key=item.compiled_artifact_key,
        static_legal=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        cost=item.cost,
    )


def _build_storage_candidate(
    item: StorageEvidence,
    *,
    bound_module: BFBoundModule,
    hardware: HardwareProfile,
    batch_id_by_evidence: dict[str, str],
    representation_id_by_evidence: dict[str, str],
) -> StorageCandidate:
    if not item.arena_id:
        raise ValueError("storage evidence arena_id must be non-empty")
    overrides = {
        override.value_id: override for override in item.value_layout_overrides
    }
    if len(overrides) != len(item.value_layout_overrides):
        raise ValueError("storage evidence contains duplicate value overrides")
    values = {value.value_id: value for value in bound_module.graph.values}
    if any(value_id not in values for value_id in overrides):
        raise ValueError("storage evidence overrides an unknown Bound IR value")
    op_index = {op.op_id: index for index, op in enumerate(bound_module.graph.ops)}
    producer = {
        value_id: op.op_id for op in bound_module.graph.ops for value_id in op.outputs
    }
    users: dict[str, list[str]] = {}
    for op in bound_module.graph.ops:
        for value_id in op.inputs:
            users.setdefault(value_id, []).append(op.op_id)

    offset = 0
    bindings: list[StorageBinding] = []
    for value in bound_module.graph.values:
        logical_size = _static_tensor_bytes(
            value.tensor_type.shape, value.tensor_type.dtype
        )
        if logical_size is None:
            raise ValueError(
                "reference storage builder requires static Bound IR shapes"
            )
        override = overrides.get(value.value_id)
        representation = (
            override.representation
            if override is not None
            else BoundRepresentation.DENSE
        )
        physical_size = (
            override.physical_size_bytes if override is not None else logical_size
        )
        if physical_size <= 0:
            raise ValueError("storage physical size must be positive")
        if representation == BoundRepresentation.DENSE and physical_size < logical_size:
            raise ValueError("dense storage evidence underallocates a value")
        offset = _align(offset, hardware.alignment_bytes)
        physical_size = _align(physical_size, hardware.alignment_bytes)
        live_from, live_to = _value_lifetime(
            value.value_id,
            bound_module=bound_module,
            op_index=op_index,
            producer=producer,
            users=users,
        )
        bindings.append(
            StorageBinding(
                value_id=value.value_id,
                arena_id=item.arena_id,
                offset_bytes=offset,
                logical_size_bytes=logical_size,
                size_bytes=physical_size,
                representation=representation,
                live_from_op_id=live_from,
                live_to_op_id=live_to,
            )
        )
        offset += physical_size

    batch_ids = tuple(
        _lookup_id(
            batch_id_by_evidence,
            evidence_id,
            label="batch evidence",
        )
        for evidence_id in item.compatible_batch_evidence_ids
    )
    representation_ids = tuple(
        _lookup_id(
            representation_id_by_evidence,
            evidence_id,
            label="representation evidence",
        )
        for evidence_id in item.compatible_representation_evidence_ids
    )
    return StorageCandidate(
        candidate_id=_candidate_id("storage", item.evidence_id),
        bindings=tuple(bindings),
        compatible_batch_candidate_ids=batch_ids,
        compatible_representation_candidate_ids=representation_ids,
        static_legal=not item.rejection_reasons,
        rejection_reasons=item.rejection_reasons,
        cost=PlanCost(
            predicted_latency_ms=item.cost.predicted_latency_ms,
            predicted_peak_bytes=offset,
            compile_cost_ms=item.cost.compile_cost_ms,
            setup_cost_ms=item.cost.setup_cost_ms,
            confidence=item.cost.confidence,
            risk_tags=item.cost.risk_tags,
        ),
    )


def _build_state_candidate(
    item: StateEvidence, *, bound_module: BFBoundModule
) -> StateCandidate:
    values = {value.value_id: value for value in bound_module.graph.values}
    value = values.get(item.source_value_id)
    if value is None:
        raise ValueError("state evidence references unknown Bound IR value")
    if value.state_version is None:
        raise ValueError("state evidence source has no state_version")
    return StateCandidate(
        candidate_id=_candidate_id("state", item.evidence_id),
        state_id=item.state_id,
        source_value_id=item.source_value_id,
        action=item.action,
        state_version=value.state_version,
        size_bytes=item.size_bytes,
        static_legal=not item.rejection_reasons,
        rejection_reasons=item.rejection_reasons,
        cost=item.cost,
    )


def _value_lifetime(
    value_id: str,
    *,
    bound_module: BFBoundModule,
    op_index: dict[str, int],
    producer: dict[str, str],
    users: dict[str, list[str]],
) -> tuple[str, str]:
    first_user_index = min(
        (op_index[op_id] for op_id in users.get(value_id, [])),
        default=0,
    )
    live_from_index = (
        op_index[producer[value_id]] if value_id in producer else first_user_index
    )
    live_to_index = max(
        (op_index[op_id] for op_id in users.get(value_id, [])),
        default=(
            len(bound_module.graph.ops) - 1
            if value_id in bound_module.graph.outputs
            else live_from_index
        ),
    )
    return (
        bound_module.graph.ops[live_from_index].op_id,
        bound_module.graph.ops[live_to_index].op_id,
    )


def _region_kind(op_kinds: Tuple[BoundOpKind, ...]) -> RegionKind:
    kinds = {_single_region_kind(op_kind) for op_kind in op_kinds}
    return next(iter(kinds)) if len(kinds) == 1 else RegionKind.MIXED


def _single_region_kind(op_kind: BoundOpKind) -> RegionKind:
    if op_kind == BoundOpKind.EXTERNAL_VERIFIER_CALL:
        return RegionKind.EXTERNAL_VERIFIER
    if op_kind in {BoundOpKind.SPEC_BIND, BoundOpKind.INPUT_BIND}:
        return RegionKind.BINDING
    if op_kind in {BoundOpKind.LINEAR_BACKWARD, BoundOpKind.CONV2D_BACKWARD}:
        return RegionKind.AFFINE
    if op_kind == BoundOpKind.RELU_RELAXATION:
        return RegionKind.RELAXATION
    if op_kind in {
        BoundOpKind.ADD_BACKWARD,
        BoundOpKind.CONCAT_BACKWARD,
        BoundOpKind.COEFFICIENT_COMPOSE,
    }:
        return RegionKind.ROUTING
    if op_kind == BoundOpKind.CONCRETIZE:
        return RegionKind.CONCRETIZATION
    return RegionKind.MIXED


def _evidence_hash(evidence: ReferencePlanEvidence) -> str:
    payload = {
        "evidence_set_id": evidence.evidence_set_id,
        "regions": [_evidence_payload(item) for item in evidence.regions],
        "transitions": [_evidence_payload(item) for item in evidence.transitions],
        "representations": [
            _evidence_payload(item) for item in evidence.representations
        ],
        "backends": [_evidence_payload(item) for item in evidence.backends],
        "batches": [_evidence_payload(item) for item in evidence.batches],
        "storage": [_evidence_payload(item) for item in evidence.storage],
        "states": [_evidence_payload(item) for item in evidence.states],
        "provenance": [item.to_dict() for item in evidence.provenance],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _evidence_payload(item: object) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in vars(item).items():
        if isinstance(value, PlanCost):
            payload[key] = value.to_dict()
        elif isinstance(value, tuple):
            payload[key] = [
                (
                    _evidence_payload(entry)
                    if hasattr(entry, "__dataclass_fields__")
                    else getattr(entry, "value", entry)
                )
                for entry in value
            ]
        else:
            payload[key] = getattr(value, "value", value)
    return payload


def _static_tensor_bytes(shape: Tuple[Optional[int], ...], dtype: str) -> Optional[int]:
    dtype_bytes = {
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "float32": 4,
        "int64": 8,
        "float64": 8,
    }.get(dtype)
    if dtype_bytes is None:
        raise ValueError(f"reference storage builder does not know dtype '{dtype}'")
    if any(dimension is None for dimension in shape):
        return None
    result = dtype_bytes
    for dimension in shape:
        if dimension is None:
            return None
        result *= dimension
    return result


def _candidate_id(kind: str, evidence_id: str) -> str:
    return f"{kind}:{evidence_id}"


def _lookup_region(
    regions: dict[str, RegionCandidate], evidence_id: str
) -> RegionCandidate:
    region = regions.get(evidence_id)
    if region is None:
        raise ValueError("plan evidence references unknown region evidence")
    return region


def _lookup_id(mapping: dict[str, str], evidence_id: str, *, label: str) -> str:
    candidate_id = mapping.get(evidence_id)
    if candidate_id is None:
        raise ValueError(f"plan evidence references unknown {label}")
    return candidate_id


def _deduplicate(values: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
