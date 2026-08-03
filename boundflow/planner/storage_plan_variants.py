"""Deterministic retain-all and lifetime-reuse StorageCandidate variants."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from ..ir.bound import BFBoundModule, BoundRepresentation
from ..ir.plan import (
    PlanCost,
    PlanProvenance,
    PlanTemplate,
    StorageBinding,
    StorageCandidate,
)

NATIVE_STORAGE_VARIANT_COMPILER_VERSION = "boundflow.native-storage-variants/v1"


def build_native_storage_plan_variants(
    template: PlanTemplate,
    *,
    bound_module: BFBoundModule,
) -> PlanTemplate:
    """Replace one dense baseline with retain-all and lifetime-reuse plans.

    The retain-all plan models the existing eager environment: every produced
    Bound value remains resident until the query completes.  The reuse plan
    keeps the exact compiler-derived last-use lifetime and aliases physical
    arena slots only when those lifetimes do not overlap.
    """

    template.validate(bound_module=bound_module)
    if len(template.storage_candidates) != 1:
        raise ValueError("native storage variants require exactly one baseline")
    baseline = template.storage_candidates[0]
    if any(
        binding.representation != BoundRepresentation.DENSE
        for binding in baseline.bindings
    ):
        raise NotImplementedError("native storage variants v1 require dense bindings")

    final_op_id = bound_module.graph.ops[-1].op_id
    retained_bindings = tuple(
        replace(binding, live_to_op_id=final_op_id) for binding in baseline.bindings
    )
    retained_peak = _arena_peak(retained_bindings)
    reused_bindings = _allocate_lifetime_reuse(
        baseline.bindings,
        bound_module=bound_module,
        alignment_bytes=template.hardware.alignment_bytes,
    )
    reused_peak = _arena_peak(reused_bindings)
    if reused_peak >= retained_peak:
        raise ValueError("native lifetime reuse does not reduce planned arena bytes")

    common_risks = tuple(
        sorted(
            {
                *baseline.cost.risk_tags,
                "correctness_only",
                "policy_cost_not_benchmarked",
            }
        )
    )
    retained = StorageCandidate(
        candidate_id="storage:native-retain-all-v1",
        bindings=retained_bindings,
        compatible_batch_candidate_ids=baseline.compatible_batch_candidate_ids,
        compatible_representation_candidate_ids=(
            baseline.compatible_representation_candidate_ids
        ),
        static_legal=True,
        rejection_reasons=(),
        cost=PlanCost(
            predicted_latency_ms=0.0,
            predicted_peak_bytes=retained_peak,
            compile_cost_ms=0.0,
            setup_cost_ms=0.0,
            confidence=1.0,
            risk_tags=common_risks,
        ),
    )
    reused = StorageCandidate(
        candidate_id="storage:native-lifetime-reuse-v1",
        bindings=reused_bindings,
        compatible_batch_candidate_ids=baseline.compatible_batch_candidate_ids,
        compatible_representation_candidate_ids=(
            baseline.compatible_representation_candidate_ids
        ),
        static_legal=True,
        rejection_reasons=(),
        cost=PlanCost(
            # This deterministic policy penalty orders the unconstrained plan;
            # it is not measured latency and therefore cannot support a perf claim.
            predicted_latency_ms=0.001,
            predicted_peak_bytes=reused_peak,
            compile_cost_ms=0.0,
            setup_cost_ms=0.0,
            confidence=1.0,
            risk_tags=common_risks,
        ),
    )
    storage = (retained, reused)
    config_payload = json.dumps(
        {
            "parent_planner_config_hash": template.planner_config_hash,
            "storage": [candidate.to_dict() for candidate in storage],
            "compiler": NATIVE_STORAGE_VARIANT_COMPILER_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    result = replace(
        template,
        planner_config_hash=hashlib.sha256(config_payload.encode("utf-8")).hexdigest(),
        storage_candidates=storage,
        provenance=(
            *template.provenance,
            PlanProvenance(
                "storage_variant_compiler",
                NATIVE_STORAGE_VARIANT_COMPILER_VERSION,
            ),
            PlanProvenance("storage_runtime_contract", "last_use_eviction_v1"),
            PlanProvenance("storage_performance_claim", "forbidden"),
        ),
    )
    result.validate(bound_module=bound_module)
    return result


def _allocate_lifetime_reuse(
    bindings: tuple[StorageBinding, ...],
    *,
    bound_module: BFBoundModule,
    alignment_bytes: int,
) -> tuple[StorageBinding, ...]:
    op_index = {op.op_id: index for index, op in enumerate(bound_module.graph.ops)}
    original_index = {binding.value_id: index for index, binding in enumerate(bindings)}
    ordered = sorted(
        bindings,
        key=lambda binding: (
            op_index[binding.live_from_op_id],
            original_index[binding.value_id],
        ),
    )
    placed: dict[str, StorageBinding] = {}
    for binding in ordered:
        start = op_index[binding.live_from_op_id]
        active = tuple(
            item for item in placed.values() if op_index[item.live_to_op_id] >= start
        )
        offset = _first_aligned_gap(
            active,
            size_bytes=binding.size_bytes,
            alignment_bytes=alignment_bytes,
        )
        placed[binding.value_id] = replace(binding, offset_bytes=offset)
    return tuple(placed[binding.value_id] for binding in bindings)


def _first_aligned_gap(
    active: tuple[StorageBinding, ...],
    *,
    size_bytes: int,
    alignment_bytes: int,
) -> int:
    offset = 0
    for binding in sorted(active, key=lambda item: (item.offset_bytes, item.value_id)):
        offset = _align(offset, alignment_bytes)
        if offset + size_bytes <= binding.offset_bytes:
            return offset
        offset = max(offset, binding.end_bytes)
    return _align(offset, alignment_bytes)


def _arena_peak(bindings: tuple[StorageBinding, ...]) -> int:
    arena_peaks: dict[str, int] = {}
    for binding in bindings:
        arena_peaks[binding.arena_id] = max(
            arena_peaks.get(binding.arena_id, 0), binding.end_bytes
        )
    return sum(arena_peaks.values())


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


__all__ = [
    "NATIVE_STORAGE_VARIANT_COMPILER_VERSION",
    "build_native_storage_plan_variants",
]
