"""Migration coverage for legacy PR-11/12 planning objects."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import torch

from boundflow.frontends.plain_crown_bound_ir import build_plain_crown_bound_ir
from boundflow.ir.bound import BoundRepresentation
from boundflow.ir.plan import PlanCost
from boundflow.ir.task import (
    BFTaskModule,
    BoundTask,
    BufferSpec,
    StoragePlan,
    TaskKind,
    TaskOp,
)
from boundflow.planner.core import PlanBundle
from boundflow.planner.execution_candidate import (
    BackendVariant,
    ExecutionCandidate,
    PlacementKind,
)
from boundflow.planner.materialization import (
    PLAN_SCHEMA_VERSION,
    MaterializationAction,
    MaterializationCandidate,
    MaterializationPlan,
    MaterializationPolicy,
)
from boundflow.planner.materialization_placement import (
    PLACEMENT_SCHEMA_VERSION,
    BarrierPlacement,
    MaterializationPlacementPlan,
    PlacementPolicy,
)
from boundflow.planner.plan_ir_legacy import (
    LegacyMigrationStatus,
    LegacyPlanKind,
    LegacyStorageLifetime,
    adapt_execution_candidate,
    adapt_materialization_placement_plan,
    adapt_materialization_plan,
    adapt_storage_plan,
    classify_fused_crown_step,
    classify_plan_bundle_meta,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.fused_crown import FusedCrownExecutionStep
from boundflow.runtime.task_executor import InputSpec


def _cost(latency: float, peak: int) -> PlanCost:
    return PlanCost(
        predicted_latency_ms=latency,
        predicted_peak_bytes=peak,
        compile_cost_ms=0.0,
        setup_cost_ms=0.0,
        confidence=0.75,
        risk_tags=("legacy_cost",),
    )


def _materialization_plan(
    *,
    action: MaterializationAction = MaterializationAction.STRUCTURED,
    structured_latency: float | None = 0.8,
) -> MaterializationPlan:
    return MaterializationPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        policy=MaterializationPolicy.GLOBAL,
        action=action,
        safe_memory_budget_bytes=800,
        recommended_domain_batch_size=1,
        reason="test",
        candidates=(
            MaterializationCandidate(
                action=MaterializationAction.DENSE,
                capability_legal=True,
                memory_feasible=False,
                predicted_peak_bytes=1_000,
                predicted_latency_ms=1.0,
                reasons=("predicted_peak_exceeds_safe_budget",),
            ),
            MaterializationCandidate(
                action=MaterializationAction.STRUCTURED,
                capability_legal=True,
                memory_feasible=True,
                predicted_peak_bytes=600,
                predicted_latency_ms=structured_latency,
                reasons=("feasible",),
            ),
        ),
    )


def _bound_module():
    task_module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="legacy-storage",
                kind=TaskKind.INTERVAL_IBP,
                ops=[TaskOp("linear", "linear", ["input", "weight", "bias"], ["out"])],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="legacy-storage",
        bindings={
            "params": {
                "weight": torch.eye(2),
                "bias": torch.zeros(2),
            }
        },
    )
    spec = InputSpec.linf(value_name="input", center=torch.zeros(2, 2), eps=0.1)
    interval_env, relu_pre = _forward_ibp_trace_mlp(task_module, spec)
    return build_plain_crown_bound_ir(
        task_module,
        spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    ).module


def test_materialization_plan_adapter_preserves_candidates_and_selection() -> None:
    plan = _materialization_plan()
    migration = adapt_materialization_plan(
        plan,
        region_id="region:linear",
        structured_transition_candidate_ids=("cast:u", "cast:l"),
        cost_by_action={
            MaterializationAction.DENSE: _cost(1.0, 1_000),
            MaterializationAction.STRUCTURED: _cost(0.8, 600),
        },
        reduced_spec_batch_size=2,
        reduced_sample_batch_size=1,
    )

    assert migration.status == LegacyMigrationStatus.ADAPTED
    assert len(migration.representation_candidates) == 2
    selected = migration.selected_candidate_ids[0]
    assert "structured" in selected
    structured = next(
        candidate
        for candidate in migration.representation_candidates
        if candidate.candidate_id == selected
    )
    assert structured.required_transition_candidate_ids == ("cast:u", "cast:l")
    assert (
        migration.source_hash
        == adapt_materialization_plan(
            plan,
            region_id="region:linear",
            structured_transition_candidate_ids=("cast:u", "cast:l"),
            cost_by_action={
                MaterializationAction.DENSE: _cost(1.0, 1_000),
                MaterializationAction.STRUCTURED: _cost(0.8, 600),
            },
            reduced_spec_batch_size=2,
            reduced_sample_batch_size=1,
        ).source_hash
    )


def test_materialization_plan_adapter_discloses_external_cost_and_replan_gap() -> None:
    missing_latency = adapt_materialization_plan(
        _materialization_plan(structured_latency=None),
        region_id="region:linear",
        structured_transition_candidate_ids=(),
        cost_by_action={
            MaterializationAction.DENSE: _cost(1.0, 1_000),
            MaterializationAction.STRUCTURED: _cost(0.8, 600),
        },
        reduced_spec_batch_size=2,
        reduced_sample_batch_size=1,
    )
    assert missing_latency.status == LegacyMigrationStatus.PARTIAL
    assert any(
        issue.reason == "latency_cost_supplied_outside_legacy_record"
        for issue in missing_latency.issues
    )

    reduce_batch = adapt_materialization_plan(
        _materialization_plan(action=MaterializationAction.REDUCE_BATCH),
        region_id="region:linear",
        structured_transition_candidate_ids=(),
        cost_by_action={
            MaterializationAction.DENSE: _cost(1.0, 1_000),
            MaterializationAction.STRUCTURED: _cost(0.8, 600),
            MaterializationAction.REDUCE_BATCH: _cost(1.5, 500),
        },
        reduced_spec_batch_size=2,
        reduced_sample_batch_size=1,
    )
    assert reduce_batch.status == LegacyMigrationStatus.PARTIAL
    assert reduce_batch.batch_candidates[0].domain_batch_size == 1
    assert reduce_batch.selected_candidate_ids == (
        reduce_batch.batch_candidates[0].candidate_id,
    )


def test_placement_and_execution_adapters_split_decision_axes() -> None:
    placement = MaterializationPlacementPlan(
        schema_version=PLACEMENT_SCHEMA_VERSION,
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
        placements=(
            BarrierPlacement(
                barrier_id="relu0",
                action=MaterializationAction.STRUCTURED,
                persistent_bytes=400,
                ephemeral_bytes=100,
                latency_ms=0.7,
                reason="selected_structured",
            ),
        ),
        predicted_peak_bytes=500,
        predicted_latency_ms=0.7,
        safe_memory_budget_bytes=800,
        requires_replan=False,
        recommended_domain_batch_size=2,
        reason="test",
    )
    placement_migration = adapt_materialization_placement_plan(
        placement,
        region_by_barrier_id={"relu0": "region:relu0"},
        transition_ids_by_barrier_id={"relu0": ("transition:relu0",)},
        confidence=0.6,
    )
    assert placement_migration.status == LegacyMigrationStatus.PARTIAL
    assert (
        placement_migration.representation_candidates[0].representation
        == BoundRepresentation.STRUCTURED
    )
    assert any(issue.field == "candidate_space" for issue in placement_migration.issues)

    execution = ExecutionCandidate(
        placement=PlacementKind.STRUCTURED,
        backend=BackendVariant.TVM_FUSED_TIR,
        domain_batch_size=3,
        spec_batch_size=17,
        materialization_points=("relu0",),
        capability_id="tvm-fused-linear-v1",
        schedule_id="legacy-two-kernel",
        reason="test",
    )
    execution_migration = adapt_execution_candidate(
        execution,
        region_id="region:linear",
        transition_id_by_materialization_point={"relu0": "transition:relu0"},
        cost=_cost(0.5, 700),
    )
    assert execution_migration.status == LegacyMigrationStatus.PARTIAL
    assert len(execution_migration.representation_candidates) == 1
    assert len(execution_migration.backend_candidates) == 1
    assert len(execution_migration.batch_candidates) == 1
    assert any(issue.field == "schedule_id" for issue in execution_migration.issues)


def test_storage_adapter_requires_explicit_bound_lifetime_mapping() -> None:
    module = _bound_module()
    lower_id = module.graph.outputs[0]
    concretize = module.graph.ops[-1]
    storage_plan = StoragePlan(
        buffers={
            "lower-buffer": BufferSpec(
                buffer_id="lower-buffer",
                dtype="float32",
                shape=[2, 2],
            )
        },
        value_to_buffer={"legacy-lower": "lower-buffer"},
    )
    adapted = adapt_storage_plan(
        storage_plan,
        bound_module=module,
        lifetime_by_legacy_value={
            "legacy-lower": LegacyStorageLifetime(
                bound_value_id=lower_id,
                live_from_op_id=concretize.op_id,
                live_to_op_id=concretize.op_id,
                representation=BoundRepresentation.DENSE,
            )
        },
        compatible_batch_candidate_ids=("batch:full",),
        compatible_representation_candidate_ids=("representation:dense",),
        cost=_cost(0.0, 16),
    )
    assert adapted.status == LegacyMigrationStatus.ADAPTED
    assert adapted.storage_candidates[0].bindings[0].value_id == lower_id

    unsupported = adapt_storage_plan(
        storage_plan,
        bound_module=module,
        lifetime_by_legacy_value={},
        compatible_batch_candidate_ids=("batch:full",),
        compatible_representation_candidate_ids=("representation:dense",),
        cost=_cost(0.0, 16),
    )
    assert unsupported.status == LegacyMigrationStatus.UNSUPPORTED
    assert any("lifetime" in issue.reason for issue in unsupported.issues)


def test_ir2_migration_table_classifies_schedule_and_untyped_meta() -> None:
    fused = classify_fused_crown_step(
        FusedCrownExecutionStep(
            kind="fused_relu_linear",
            relu_op_index=1,
            affine_op_index=0,
            consumed_outputs=("relu_out", "affine_out"),
            graph_fingerprint="graph-hash",
        )
    )
    fake_bundle = cast(
        PlanBundle,
        SimpleNamespace(
            meta={"semantic_choice": "bad"},
            lowering_plan={"backend": "tvm"},
        ),
    )
    bundle = classify_plan_bundle_meta(fake_bundle)

    assert fused.status == LegacyMigrationStatus.UNSUPPORTED
    assert bundle.status == LegacyMigrationStatus.UNSUPPORTED
    assert "Task_Schedule_IR" in fused.issues[0].reason
    assert {
        LegacyPlanKind.MATERIALIZATION_PLAN,
        LegacyPlanKind.MATERIALIZATION_PLACEMENT_PLAN,
        LegacyPlanKind.EXECUTION_CANDIDATE,
        LegacyPlanKind.STORAGE_PLAN,
        fused.source_kind,
        bundle.source_kind,
    } == set(LegacyPlanKind)
