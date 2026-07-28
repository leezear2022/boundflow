"""Explicit migration adapters from PR-11/12 records into Plan IR candidates."""

# Declarative migration reports intentionally expose every mapped decision axis.
# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Mapping, Optional, Tuple

from ..ir.bound import BFBoundModule, BoundRepresentation
from ..ir.plan import (
    BackendCandidate,
    BackendKind,
    BatchCandidate,
    MaterializationCandidate as PlanMaterializationCandidate,
    PlanCost,
    RepresentationCandidate,
    StorageBinding,
    StorageCandidate,
)
from ..ir.task import StoragePlan
from ..runtime.fused_crown import FusedCrownExecutionStep
from .core import PlanBundle
from .execution_candidate import (
    BackendVariant,
    ExecutionCandidate,
    PlacementKind,
)
from .materialization import (
    MaterializationAction,
    MaterializationPlan,
)
from .materialization_placement import MaterializationPlacementPlan


class LegacyPlanKind(Enum):
    """Legacy object families covered by the IR-2 migration table."""

    MATERIALIZATION_PLAN = "materialization_plan"
    MATERIALIZATION_PLACEMENT_PLAN = "materialization_placement_plan"
    EXECUTION_CANDIDATE = "execution_candidate"
    STORAGE_PLAN = "storage_plan"
    FUSED_CROWN_EXECUTION_STEP = "fused_crown_execution_step"
    PLAN_BUNDLE_META = "plan_bundle_meta"


class LegacyMigrationStatus(Enum):
    """Whether a legacy object was fully, partly, or not migrated."""

    ADAPTED = "adapted"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class LegacyMigrationIssue:
    """One explicit legacy field/evidence gap that was not silently guessed."""

    field: str
    reason: str

    def validate(self) -> None:
        if not self.field or not self.reason:
            raise ValueError("legacy migration issue fields must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {"field": self.field, "reason": self.reason}


@dataclass(frozen=True)
class LegacyPlanMigration:
    """Typed migration result; partial mappings retain every unsupported gap."""

    source_kind: LegacyPlanKind
    source_schema_version: str
    source_hash: str
    status: LegacyMigrationStatus
    representation_candidates: Tuple[RepresentationCandidate, ...] = ()
    materialization_candidates: Tuple[PlanMaterializationCandidate, ...] = ()
    backend_candidates: Tuple[BackendCandidate, ...] = ()
    batch_candidates: Tuple[BatchCandidate, ...] = ()
    storage_candidates: Tuple[StorageCandidate, ...] = ()
    selected_candidate_ids: Tuple[str, ...] = ()
    issues: Tuple[LegacyMigrationIssue, ...] = ()

    def validate(self) -> None:
        if not self.source_schema_version or len(self.source_hash) != 64:
            raise ValueError("legacy migration source identity is incomplete")
        candidates = self.all_candidates()
        for candidate in candidates:
            candidate.validate()
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("legacy migration emits duplicate candidate IDs")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ValueError("legacy migration selects duplicate candidate IDs")
        if not set(self.selected_candidate_ids).issubset(set(candidate_ids)):
            raise ValueError("legacy migration selects an unemitted candidate")
        for issue in self.issues:
            issue.validate()
        if self.status == LegacyMigrationStatus.ADAPTED and self.issues:
            raise ValueError("fully adapted migration cannot retain issues")
        if self.status == LegacyMigrationStatus.PARTIAL and (
            not candidates or not self.issues
        ):
            raise ValueError("partial migration requires candidates and issues")
        if self.status == LegacyMigrationStatus.UNSUPPORTED and (
            candidates or not self.issues
        ):
            raise ValueError("unsupported migration requires only explicit issues")

    def all_candidates(self) -> tuple[
        RepresentationCandidate
        | PlanMaterializationCandidate
        | BackendCandidate
        | BatchCandidate
        | StorageCandidate,
        ...,
    ]:
        return (
            *self.representation_candidates,
            *self.materialization_candidates,
            *self.backend_candidates,
            *self.batch_candidates,
            *self.storage_candidates,
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "source_kind": self.source_kind.value,
            "source_schema_version": self.source_schema_version,
            "source_hash": self.source_hash,
            "status": self.status.value,
            "representation_candidates": [
                candidate.to_dict() for candidate in self.representation_candidates
            ],
            "materialization_candidates": [
                candidate.to_dict() for candidate in self.materialization_candidates
            ],
            "backend_candidates": [
                candidate.to_dict() for candidate in self.backend_candidates
            ],
            "batch_candidates": [
                candidate.to_dict() for candidate in self.batch_candidates
            ],
            "storage_candidates": [
                candidate.to_dict() for candidate in self.storage_candidates
            ],
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class LegacyStorageLifetime:
    """Missing lifetime/representation facts supplied explicitly to migration."""

    bound_value_id: str
    live_from_op_id: str
    live_to_op_id: str
    representation: BoundRepresentation

    def validate(self) -> None:
        if (
            not self.bound_value_id
            or not self.live_from_op_id
            or not self.live_to_op_id
        ):
            raise ValueError("legacy storage lifetime IDs must be non-empty")


def adapt_materialization_plan(
    plan: MaterializationPlan,
    *,
    region_id: str,
    structured_transition_candidate_ids: Tuple[str, ...],
    cost_by_action: Mapping[MaterializationAction, PlanCost],
    reduced_spec_batch_size: int,
    reduced_sample_batch_size: int,
) -> LegacyPlanMigration:
    """Map single-barrier PR-11 candidates without hiding missing cost evidence."""

    representations: list[RepresentationCandidate] = []
    issues: list[LegacyMigrationIssue] = []
    candidate_id_by_action: dict[MaterializationAction, str] = {}
    for legacy in plan.candidates:
        if legacy.action not in {
            MaterializationAction.DENSE,
            MaterializationAction.STRUCTURED,
        }:
            issues.append(
                LegacyMigrationIssue(
                    "candidates.action",
                    f"unsupported_materialization_action:{legacy.action.value}",
                )
            )
            continue
        cost = cost_by_action.get(legacy.action)
        if cost is None:
            issues.append(
                LegacyMigrationIssue(
                    f"candidates.{legacy.action.value}.cost",
                    "missing_complete_plan_cost_evidence",
                )
            )
            continue
        cost.validate()
        if (
            legacy.predicted_peak_bytes is not None
            and cost.predicted_peak_bytes != legacy.predicted_peak_bytes
        ):
            issues.append(
                LegacyMigrationIssue(
                    f"candidates.{legacy.action.value}.predicted_peak_bytes",
                    "supplied_cost_disagrees_with_legacy_peak",
                )
            )
            continue
        if legacy.predicted_latency_ms is None:
            issues.append(
                LegacyMigrationIssue(
                    f"candidates.{legacy.action.value}.predicted_latency_ms",
                    "latency_cost_supplied_outside_legacy_record",
                )
            )
        elif not math_isclose(cost.predicted_latency_ms, legacy.predicted_latency_ms):
            issues.append(
                LegacyMigrationIssue(
                    f"candidates.{legacy.action.value}.predicted_latency_ms",
                    "supplied_cost_disagrees_with_legacy_latency",
                )
            )
            continue
        candidate_id = f"legacy:materialization:{region_id}:{legacy.action.value}"
        candidate_id_by_action[legacy.action] = candidate_id
        static_legal = bool(legacy.capability_legal)
        reasons = (
            ()
            if static_legal
            else tuple(
                reason
                for reason in legacy.reasons
                if reason != "predicted_peak_exceeds_safe_budget"
            )
            or ("legacy_capability_rejected",)
        )
        representations.append(
            RepresentationCandidate(
                candidate_id=candidate_id,
                region_id=region_id,
                representation=(
                    BoundRepresentation.DENSE
                    if legacy.action == MaterializationAction.DENSE
                    else BoundRepresentation.STRUCTURED
                ),
                required_transition_candidate_ids=(
                    ()
                    if legacy.action == MaterializationAction.DENSE
                    else structured_transition_candidate_ids
                ),
                static_legal=static_legal,
                rejection_reasons=reasons,
                cost=cost,
            )
        )

    batches: tuple[BatchCandidate, ...] = ()
    selected: tuple[str, ...] = ()
    if plan.action == MaterializationAction.REDUCE_BATCH:
        issue = LegacyMigrationIssue(
            "action",
            "reduce_batch_requests_runtime_replan_but_does_not_select_representation",
        )
        issues.append(issue)
        batch_cost = cost_by_action.get(MaterializationAction.REDUCE_BATCH)
        if batch_cost is not None:
            batch = BatchCandidate(
                candidate_id=f"legacy:batch:reduce:{region_id}",
                domain_batch_size=plan.recommended_domain_batch_size,
                spec_batch_size=reduced_spec_batch_size,
                sample_batch_size=reduced_sample_batch_size,
                estimated_payload_bytes=0,
                static_legal=True,
                rejection_reasons=(),
                cost=batch_cost,
            )
            batches = (batch,)
            selected = (batch.candidate_id,)
    elif plan.action in candidate_id_by_action:
        selected = (candidate_id_by_action[plan.action],)
    else:
        issues.append(
            LegacyMigrationIssue("action", "selected_action_has_no_migrated_candidate")
        )

    return _migration(
        source_kind=LegacyPlanKind.MATERIALIZATION_PLAN,
        source_schema_version=plan.schema_version,
        source_payload=plan.to_dict(),
        representation_candidates=tuple(representations),
        batch_candidates=batches,
        selected_candidate_ids=selected,
        issues=tuple(issues),
    )


def adapt_materialization_placement_plan(
    plan: MaterializationPlacementPlan,
    *,
    region_by_barrier_id: Mapping[str, str],
    transition_ids_by_barrier_id: Mapping[str, Tuple[str, ...]],
    confidence: float,
) -> LegacyPlanMigration:
    """Map selected barrier placements and disclose the absent candidate table."""

    representations: list[RepresentationCandidate] = []
    issues: list[LegacyMigrationIssue] = [
        LegacyMigrationIssue(
            "candidate_space",
            "legacy_placement_plan_only_preserves_selected_barrier_actions",
        )
    ]
    selected: list[str] = []
    for placement in plan.placements:
        region_id = region_by_barrier_id.get(placement.barrier_id)
        if region_id is None:
            issues.append(
                LegacyMigrationIssue(
                    f"placements.{placement.barrier_id}",
                    "missing_bound_ir_region_mapping",
                )
            )
            continue
        representation = (
            BoundRepresentation.DENSE
            if placement.action == MaterializationAction.DENSE
            else BoundRepresentation.STRUCTURED
        )
        candidate_id = (
            f"legacy:placement:{placement.barrier_id}:{placement.action.value}"
        )
        candidate = RepresentationCandidate(
            candidate_id=candidate_id,
            region_id=region_id,
            representation=representation,
            required_transition_candidate_ids=(
                ()
                if representation == BoundRepresentation.DENSE
                else transition_ids_by_barrier_id.get(placement.barrier_id, ())
            ),
            static_legal=True,
            rejection_reasons=(),
            cost=PlanCost(
                predicted_latency_ms=placement.latency_ms,
                predicted_peak_bytes=(
                    placement.persistent_bytes + placement.ephemeral_bytes
                ),
                compile_cost_ms=0.0,
                setup_cost_ms=0.0,
                confidence=confidence,
                risk_tags=("legacy_local_barrier_cost",),
            ),
        )
        representations.append(candidate)
        selected.append(candidate_id)
    if plan.requires_replan:
        issues.append(
            LegacyMigrationIssue(
                "requires_replan",
                "legacy_replan_request_requires_runtime_query_context",
            )
        )
    return _migration(
        source_kind=LegacyPlanKind.MATERIALIZATION_PLACEMENT_PLAN,
        source_schema_version=plan.schema_version,
        source_payload=plan.to_dict(),
        representation_candidates=tuple(representations),
        selected_candidate_ids=tuple(selected),
        issues=tuple(issues),
    )


def adapt_execution_candidate(
    candidate: ExecutionCandidate,
    *,
    region_id: str,
    transition_id_by_materialization_point: Mapping[str, str],
    cost: PlanCost,
) -> LegacyPlanMigration:
    """Split one PR-12 placement/backend/batch record into Plan IR axes."""

    candidate.validate()
    cost.validate()
    representation_id = f"legacy:execution:representation:{region_id}"
    batch_id = f"legacy:execution:batch:{region_id}"
    backend_id = f"legacy:execution:backend:{region_id}"
    missing_points = tuple(
        point
        for point in candidate.materialization_points
        if point not in transition_id_by_materialization_point
    )
    transition_ids = tuple(
        transition_id_by_materialization_point[point]
        for point in candidate.materialization_points
        if point in transition_id_by_materialization_point
    )
    issues = [
        LegacyMigrationIssue(
            "schedule_id",
            "schedule_identity_belongs_to_IR3_Task_Schedule_lowering",
        )
    ]
    issues.extend(
        LegacyMigrationIssue(
            f"materialization_points.{point}",
            "missing_bound_ir_transition_mapping",
        )
        for point in missing_points
    )
    representation = RepresentationCandidate(
        candidate_id=representation_id,
        region_id=region_id,
        representation=(
            BoundRepresentation.DENSE
            if candidate.placement == PlacementKind.DENSE
            else BoundRepresentation.STRUCTURED
        ),
        required_transition_candidate_ids=transition_ids,
        static_legal=True,
        rejection_reasons=(),
        cost=cost,
    )
    backend = BackendCandidate(
        candidate_id=backend_id,
        region_id=region_id,
        backend=_backend_kind(candidate.backend),
        capability_id=candidate.capability_id,
        compatible_representation_candidate_ids=(representation_id,),
        compiled_artifact_key=None,
        static_legal=True,
        rejection_reasons=(),
        cost=cost,
    )
    batch = BatchCandidate(
        candidate_id=batch_id,
        domain_batch_size=candidate.domain_batch_size,
        spec_batch_size=candidate.spec_batch_size,
        sample_batch_size=1,
        estimated_payload_bytes=0,
        static_legal=True,
        rejection_reasons=(),
        cost=cost,
    )
    return _migration(
        source_kind=LegacyPlanKind.EXECUTION_CANDIDATE,
        source_schema_version=candidate.schema_version,
        source_payload=candidate.to_dict(),
        representation_candidates=(representation,),
        backend_candidates=(backend,),
        batch_candidates=(batch,),
        selected_candidate_ids=(representation_id, backend_id, batch_id),
        issues=tuple(issues),
    )


def adapt_storage_plan(
    storage_plan: StoragePlan,
    *,
    bound_module: BFBoundModule,
    lifetime_by_legacy_value: Mapping[str, LegacyStorageLifetime],
    compatible_batch_candidate_ids: Tuple[str, ...],
    compatible_representation_candidate_ids: Tuple[str, ...],
    cost: PlanCost,
) -> LegacyPlanMigration:
    """Map legacy logical/physical buffers when missing lifetimes are supplied."""

    storage_plan.validate()
    cost.validate()
    bound_values = {value.value_id: value for value in bound_module.graph.values}
    bindings: list[StorageBinding] = []
    issues: list[LegacyMigrationIssue] = []
    for legacy_value, logical_buffer_id in sorted(storage_plan.value_to_buffer.items()):
        lifetime = lifetime_by_legacy_value.get(legacy_value)
        if lifetime is None:
            issues.append(
                LegacyMigrationIssue(
                    f"value_to_buffer.{legacy_value}",
                    "missing_bound_ir_value_lifetime_mapping",
                )
            )
            continue
        lifetime.validate()
        bound_value = bound_values.get(lifetime.bound_value_id)
        if bound_value is None:
            issues.append(
                LegacyMigrationIssue(
                    f"value_to_buffer.{legacy_value}",
                    "mapped_bound_ir_value_does_not_exist",
                )
            )
            continue
        logical_spec = storage_plan.get_logical_spec(logical_buffer_id)
        physical_spec = storage_plan.get_physical_spec(logical_buffer_id)
        logical_bytes = _buffer_bytes(logical_spec.shape, logical_spec.dtype)
        physical_bytes = _buffer_bytes(physical_spec.shape, physical_spec.dtype)
        if logical_bytes is None or physical_bytes is None:
            issues.append(
                LegacyMigrationIssue(
                    f"value_to_buffer.{legacy_value}",
                    "dynamic_legacy_buffer_size_is_not_migratable",
                )
            )
            continue
        bindings.append(
            StorageBinding(
                value_id=bound_value.value_id,
                arena_id=f"legacy-physical:{storage_plan.to_physical(logical_buffer_id)}",
                offset_bytes=0,
                logical_size_bytes=logical_bytes,
                size_bytes=physical_bytes,
                representation=lifetime.representation,
                live_from_op_id=lifetime.live_from_op_id,
                live_to_op_id=lifetime.live_to_op_id,
            )
        )
    if not bindings:
        return _migration(
            source_kind=LegacyPlanKind.STORAGE_PLAN,
            source_schema_version="boundflow.storage_plan/legacy",
            source_payload=_storage_payload(storage_plan),
            issues=tuple(issues)
            or (
                LegacyMigrationIssue(
                    "storage_plan", "no_bound_ir_storage_bindings_emitted"
                ),
            ),
        )
    storage = StorageCandidate(
        candidate_id="legacy:storage:plan",
        bindings=tuple(bindings),
        compatible_batch_candidate_ids=compatible_batch_candidate_ids,
        compatible_representation_candidate_ids=(
            compatible_representation_candidate_ids
        ),
        static_legal=True,
        rejection_reasons=(),
        cost=cost,
    )
    return _migration(
        source_kind=LegacyPlanKind.STORAGE_PLAN,
        source_schema_version="boundflow.storage_plan/legacy",
        source_payload=_storage_payload(storage_plan),
        storage_candidates=(storage,),
        selected_candidate_ids=(storage.candidate_id,),
        issues=tuple(issues),
    )


def classify_fused_crown_step(
    step: FusedCrownExecutionStep,
) -> LegacyPlanMigration:
    """Keep fused steps out of Plan IR; they belong to IR-3 lowering."""

    return _migration(
        source_kind=LegacyPlanKind.FUSED_CROWN_EXECUTION_STEP,
        source_schema_version="boundflow.fused_crown_step/legacy",
        source_payload=asdict(step),
        issues=(
            LegacyMigrationIssue(
                "entire_object",
                "execution_step_belongs_to_Task_Schedule_IR_not_Plan_IR",
            ),
        ),
    )


def classify_plan_bundle_meta(bundle: PlanBundle) -> LegacyPlanMigration:
    """Reject semantic migration from untyped PlanBundle metadata."""

    return _migration(
        source_kind=LegacyPlanKind.PLAN_BUNDLE_META,
        source_schema_version="boundflow.plan_bundle/legacy",
        source_payload={
            "meta_keys": sorted(str(key) for key in bundle.meta),
            "lowering_plan_keys": sorted(str(key) for key in bundle.lowering_plan),
        },
        issues=(
            LegacyMigrationIssue(
                "meta",
                "untyped_metadata_may_only_be_retained_as_debug_provenance",
            ),
            LegacyMigrationIssue(
                "lowering_plan",
                "untyped_lowering_entries_require_explicit_IR2_or_IR3_adapter",
            ),
        ),
    )


def _migration(
    *,
    source_kind: LegacyPlanKind,
    source_schema_version: str,
    source_payload: object,
    representation_candidates: Tuple[RepresentationCandidate, ...] = (),
    materialization_candidates: Tuple[PlanMaterializationCandidate, ...] = (),
    backend_candidates: Tuple[BackendCandidate, ...] = (),
    batch_candidates: Tuple[BatchCandidate, ...] = (),
    storage_candidates: Tuple[StorageCandidate, ...] = (),
    selected_candidate_ids: Tuple[str, ...] = (),
    issues: Tuple[LegacyMigrationIssue, ...] = (),
) -> LegacyPlanMigration:
    has_candidates = bool(
        representation_candidates
        or materialization_candidates
        or backend_candidates
        or batch_candidates
        or storage_candidates
    )
    if issues and has_candidates:
        status = LegacyMigrationStatus.PARTIAL
    elif issues:
        status = LegacyMigrationStatus.UNSUPPORTED
    else:
        status = LegacyMigrationStatus.ADAPTED
    result = LegacyPlanMigration(
        source_kind=source_kind,
        source_schema_version=source_schema_version,
        source_hash=_stable_hash(source_payload),
        status=status,
        representation_candidates=representation_candidates,
        materialization_candidates=materialization_candidates,
        backend_candidates=backend_candidates,
        batch_candidates=batch_candidates,
        storage_candidates=storage_candidates,
        selected_candidate_ids=selected_candidate_ids,
        issues=issues,
    )
    result.validate()
    return result


def _backend_kind(backend: BackendVariant) -> BackendKind:
    return {
        BackendVariant.PYTORCH_EAGER: BackendKind.PYTORCH_DENSE,
        BackendVariant.PYTORCH_STRUCTURED: BackendKind.PYTORCH_STRUCTURED,
        BackendVariant.PYTORCH_CHUNKED: BackendKind.PYTORCH_CHUNKED,
        BackendVariant.TORCH_COMPILE: BackendKind.TORCH_COMPILE,
        BackendVariant.TVM_RELAX_UNFUSED: BackendKind.TVM_RELAX_UNFUSED,
        BackendVariant.TVM_TIR_DEFAULT: BackendKind.TVM_TIR_UNFUSED,
        BackendVariant.TVM_TIR_UNFUSED: BackendKind.TVM_TIR_UNFUSED,
        BackendVariant.TVM_FUSED_TIR: BackendKind.TVM_FUSED_TIR,
    }[backend]


def _storage_payload(storage_plan: StoragePlan) -> dict[str, object]:
    return {
        "buffers": {
            buffer_id: asdict(spec)
            for buffer_id, spec in sorted(storage_plan.buffers.items())
        },
        "value_to_buffer": dict(sorted(storage_plan.value_to_buffer.items())),
        "physical_buffers": {
            buffer_id: asdict(spec)
            for buffer_id, spec in sorted(storage_plan.physical_buffers.items())
        },
        "logical_to_physical": dict(sorted(storage_plan.logical_to_physical.items())),
    }


def _buffer_bytes(shape: list[Optional[int]], dtype: str) -> Optional[int]:
    if any(dimension is None for dimension in shape):
        return None
    item_size = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "bool": 1,
    }.get(dtype)
    if item_size is None:
        return None
    elements = 1
    for dimension in shape:
        if dimension is None:
            return None
        elements *= dimension
    return elements * item_size


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def math_isclose(left: float, right: float) -> bool:
    """Avoid accepting materially different legacy cost evidence."""

    return abs(float(left) - float(right)) <= 1e-9 * max(
        1.0, abs(float(left)), abs(float(right))
    )


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(
        f"legacy plan payload is not deterministic JSON: {type(value).__name__}"
    )
