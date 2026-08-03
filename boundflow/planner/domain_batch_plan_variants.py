"""Deterministic native full/domain-sliced BatchCandidate variants."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from ..ir.bound import BFBoundModule
from ..ir.plan import PlanCost, PlanProvenance, PlanTemplate

NATIVE_DOMAIN_BATCH_VARIANT_COMPILER_VERSION = (
    "boundflow.native-domain-batch-variants/v1"
)


def build_native_domain_batch_plan_variants(
    template: PlanTemplate,
    *,
    bound_module: BFBoundModule,
    domain_batch_size: int,
) -> PlanTemplate:
    """Add one domain-sliced alternative without a physical performance claim."""

    template.validate(bound_module=bound_module)
    if len(template.batch_candidates) != 1:
        raise ValueError("native domain batching requires one full baseline")
    baseline = template.batch_candidates[0]
    workload = template.workload
    if (
        baseline.domain_batch_size != workload.domain_batch_size
        or baseline.spec_batch_size != workload.spec_batch_size
        or baseline.sample_batch_size != workload.sample_batch_size
    ):
        raise ValueError("native domain baseline does not cover the workload")
    if domain_batch_size <= 0 or domain_batch_size >= workload.domain_batch_size:
        raise ValueError("native domain slice must be positive and smaller than full")
    risks = tuple(
        sorted(
            {
                *baseline.cost.risk_tags,
                "correctness_only",
                "domain_batch_cost_not_benchmarked",
                "no_memory_claim",
            }
        )
    )
    sliced = replace(
        baseline,
        candidate_id=f"batch:native-domain-sliced-v1:{domain_batch_size:04d}",
        domain_batch_size=domain_batch_size,
        estimated_payload_bytes=(
            baseline.estimated_payload_bytes
            * domain_batch_size
            // workload.domain_batch_size
        ),
        cost=PlanCost(
            predicted_latency_ms=baseline.cost.predicted_latency_ms,
            predicted_peak_bytes=baseline.cost.predicted_peak_bytes,
            compile_cost_ms=baseline.cost.compile_cost_ms,
            setup_cost_ms=baseline.cost.setup_cost_ms,
            confidence=baseline.cost.confidence,
            risk_tags=risks,
        ),
    )
    batches = (baseline, sliced)
    batch_ids = tuple(item.candidate_id for item in batches)
    storage = tuple(
        replace(
            candidate,
            compatible_batch_candidate_ids=tuple(
                dict.fromkeys((*candidate.compatible_batch_candidate_ids, *batch_ids))
            ),
        )
        for candidate in template.storage_candidates
    )
    payload = json.dumps(
        {
            "parent_planner_config_hash": template.planner_config_hash,
            "compiler": NATIVE_DOMAIN_BATCH_VARIANT_COMPILER_VERSION,
            "batches": [item.to_dict() for item in batches],
            "storage": [item.to_dict() for item in storage],
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
    result = replace(
        template,
        template_id=(
            "plan-template:"
            + hashlib.sha256(template_seed.encode("utf-8")).hexdigest()[:20]
        ),
        planner_config_hash=planner_hash,
        batch_candidates=batches,
        storage_candidates=storage,
        provenance=(
            *template.provenance,
            PlanProvenance(
                "domain_batch_variant_compiler",
                NATIVE_DOMAIN_BATCH_VARIANT_COMPILER_VERSION,
            ),
            PlanProvenance("domain_batch_execution_claim", "correctness_only"),
            PlanProvenance("domain_batch_performance_claim", "forbidden"),
        ),
    )
    result.validate(bound_module=bound_module)
    return result


__all__ = [
    "NATIVE_DOMAIN_BATCH_VARIANT_COMPILER_VERSION",
    "build_native_domain_batch_plan_variants",
]
