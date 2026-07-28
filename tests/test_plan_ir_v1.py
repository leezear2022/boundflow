"""Contracts for Plan IR v1 template/instance and cross-decision verifier."""

# pylint: disable=missing-function-docstring,too-many-locals

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable

import pytest
import torch

from boundflow.frontends.plain_crown_bound_ir import build_plain_crown_bound_ir
from boundflow.ir.bound import BoundMethodKind, BoundOpKind, BoundRepresentation
from boundflow.ir.plan import (
    BackendCandidate,
    BackendCapabilitySpec,
    BackendDecision,
    BackendKind,
    BatchCandidate,
    BatchDecision,
    HardwareProfile,
    MaterializationCandidate,
    MaterializationDecision,
    PlanCost,
    PlanInstance,
    PlanProvenance,
    PlanTemplate,
    RegionCandidate,
    RegionDecision,
    RegionKind,
    RejectedCandidate,
    RepresentationCandidate,
    RepresentationDecision,
    StorageBinding,
    StorageCandidate,
    StorageDecision,
    TransitionKind,
    WorkloadProfile,
)
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.task_executor import InputSpec


def _bound_module():
    task_module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="plan-test",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["hidden"]),
                    TaskOp("relu", "relu1", ["hidden"], ["activated"]),
                    TaskOp("linear", "linear2", ["activated", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="plan-test",
        bindings={
            "params": {
                "W1": torch.tensor(
                    [[1.0, -0.5, 0.25], [-0.25, 0.75, 1.0], [0.5, 0.5, -1.0]]
                ),
                "b1": torch.tensor([0.1, -0.2, 0.05]),
                "W2": torch.tensor([[0.75, -1.0, 0.5], [-0.5, 0.25, 1.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )
    input_spec = InputSpec.linf(value_name="input", center=torch.zeros(2, 3), eps=0.2)
    interval_env, relu_pre = _forward_ibp_trace_mlp(task_module, input_spec)
    return build_plain_crown_bound_ir(
        task_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    ).module


def _cost(
    *,
    latency: float = 1.0,
    peak: int = 256,
    compile_ms: float = 0.0,
) -> PlanCost:
    return PlanCost(
        predicted_latency_ms=latency,
        predicted_peak_bytes=peak,
        compile_cost_ms=compile_ms,
        setup_cost_ms=0.25,
        confidence=0.8,
        risk_tags=("model_estimate",),
    )


def _region_kind(op_kind: BoundOpKind) -> RegionKind:
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


def _logical_bytes(shape: Iterable[int | None], dtype: str) -> int:
    assert dtype == "float32"
    result = 4
    for dimension in shape:
        assert dimension is not None
        result *= dimension
    return result


def _storage_bindings(module, *, structured_values: set[str] | None = None):
    structured_values = structured_values or set()
    op_index = {op.op_id: index for index, op in enumerate(module.graph.ops)}
    producer = {
        value_id: op.op_id for op in module.graph.ops for value_id in op.outputs
    }
    users: dict[str, list[str]] = {}
    for op in module.graph.ops:
        for value_id in op.inputs:
            users.setdefault(value_id, []).append(op.op_id)
    offset = 0
    bindings = []
    for value in module.graph.values:
        logical = _logical_bytes(value.tensor_type.shape, value.tensor_type.dtype)
        representation = (
            BoundRepresentation.STRUCTURED
            if value.value_id in structured_values
            else BoundRepresentation.DENSE
        )
        physical = (
            max(16, ((logical // 2 + 15) // 16) * 16)
            if representation == BoundRepresentation.STRUCTURED
            else ((logical + 15) // 16) * 16
        )
        first_user = min(
            (op_index[op_id] for op_id in users.get(value.value_id, [])),
            default=0,
        )
        live_from_index = (
            op_index[producer[value.value_id]]
            if value.value_id in producer
            else first_user
        )
        live_to_index = max(
            (op_index[op_id] for op_id in users.get(value.value_id, [])),
            default=(
                len(module.graph.ops) - 1
                if value.value_id in module.graph.outputs
                else live_from_index
            ),
        )
        bindings.append(
            StorageBinding(
                value_id=value.value_id,
                arena_id="cpu-main",
                offset_bytes=offset,
                logical_size_bytes=logical,
                size_bytes=physical,
                representation=representation,
                live_from_op_id=module.graph.ops[live_from_index].op_id,
                live_to_op_id=module.graph.ops[live_to_index].op_id,
            )
        )
        offset += physical
    return tuple(bindings), offset


def _template_and_instance():
    module = _bound_module()
    regions = tuple(
        RegionCandidate(
            candidate_id=f"region:{op.op_id}",
            region_id=f"r:{op.op_id}",
            kind=_region_kind(op.kind),
            op_ids=(op.op_id,),
            input_value_ids=op.inputs,
            output_value_ids=op.outputs,
            fused=False,
            cost=_cost(latency=0.1),
        )
        for op in module.graph.ops
    )
    target_op = next(
        op for op in module.graph.ops if op.kind == BoundOpKind.LINEAR_BACKWARD
    )
    target_region_id = f"r:{target_op.op_id}"
    dense_representations = tuple(
        RepresentationCandidate(
            candidate_id=f"representation:dense:{region.region_id}",
            region_id=region.region_id,
            representation=BoundRepresentation.DENSE,
            required_transition_candidate_ids=(),
            static_legal=True,
            rejection_reasons=(),
            cost=_cost(latency=0.2),
        )
        for region in regions
    )
    consumer = next(op for op in module.graph.ops if target_op.outputs[0] in op.inputs)
    transitions = tuple(
        [
            MaterializationCandidate(
                candidate_id=f"transition:cast:{value_id}",
                source_value_id=value_id,
                before_op_id=target_op.op_id,
                kind=TransitionKind.CAST,
                source_representation=BoundRepresentation.DENSE,
                target_representation=BoundRepresentation.STRUCTURED,
                static_legal=True,
                rejection_reasons=(),
                cost=_cost(latency=0.05),
            )
            for value_id in (target_op.inputs[0], target_op.inputs[2])
        ]
        + [
            MaterializationCandidate(
                candidate_id=f"transition:materialize:{value_id}",
                source_value_id=value_id,
                before_op_id=consumer.op_id,
                kind=TransitionKind.MATERIALIZE,
                source_representation=BoundRepresentation.STRUCTURED,
                target_representation=BoundRepresentation.DENSE,
                static_legal=True,
                rejection_reasons=(),
                cost=_cost(latency=0.1),
            )
            for value_id in (target_op.outputs[0], target_op.outputs[2])
        ]
    )
    structured_representation = RepresentationCandidate(
        candidate_id=f"representation:structured:{target_region_id}",
        region_id=target_region_id,
        representation=BoundRepresentation.STRUCTURED,
        required_transition_candidate_ids=tuple(
            transition.candidate_id for transition in transitions
        ),
        static_legal=True,
        rejection_reasons=(),
        cost=_cost(latency=0.4),
    )
    representations = (*dense_representations, structured_representation)

    reference_capability = BackendCapabilitySpec(
        capability_id="reference-dense-v1",
        backend=BackendKind.REFERENCE,
        supported_methods=(BoundMethodKind.CROWN,),
        supported_op_kinds=tuple(dict.fromkeys(op.kind for op in module.graph.ops)),
        supported_representations=(BoundRepresentation.DENSE,),
        supported_dtypes=("float32",),
        supported_devices=("cpu",),
        supports_grad=False,
        supports_alpha=False,
        supports_beta=False,
        supports_split_state=False,
        static_shape_only=True,
    )
    structured_capability = BackendCapabilitySpec(
        capability_id="pytorch-structured-linear-v1",
        backend=BackendKind.PYTORCH_STRUCTURED,
        supported_methods=(BoundMethodKind.CROWN,),
        supported_op_kinds=(BoundOpKind.LINEAR_BACKWARD,),
        supported_representations=(BoundRepresentation.STRUCTURED,),
        supported_dtypes=("float32",),
        supported_devices=("cpu",),
        supports_grad=False,
        supports_alpha=False,
        supports_beta=False,
        supports_split_state=False,
        static_shape_only=True,
    )
    dense_backends = tuple(
        BackendCandidate(
            candidate_id=f"backend:reference:{region.region_id}",
            region_id=region.region_id,
            backend=BackendKind.REFERENCE,
            capability_id=reference_capability.capability_id,
            compatible_representation_candidate_ids=(
                f"representation:dense:{region.region_id}",
            ),
            compiled_artifact_key=None,
            static_legal=True,
            rejection_reasons=(),
            cost=_cost(latency=0.5),
        )
        for region in regions
    )
    structured_backend = BackendCandidate(
        candidate_id=f"backend:structured:{target_region_id}",
        region_id=target_region_id,
        backend=BackendKind.PYTORCH_STRUCTURED,
        capability_id=structured_capability.capability_id,
        compatible_representation_candidate_ids=(
            structured_representation.candidate_id,
        ),
        compiled_artifact_key="torch-structured-linear:test",
        static_legal=True,
        rejection_reasons=(),
        cost=_cost(latency=0.3, compile_ms=0.1),
    )
    batches = (
        BatchCandidate(
            candidate_id="batch:full",
            domain_batch_size=2,
            spec_batch_size=2,
            sample_batch_size=1,
            estimated_payload_bytes=128,
            static_legal=True,
            rejection_reasons=(),
            cost=_cost(latency=1.0),
        ),
        BatchCandidate(
            candidate_id="batch:reduced",
            domain_batch_size=1,
            spec_batch_size=1,
            sample_batch_size=1,
            estimated_payload_bytes=64,
            static_legal=True,
            rejection_reasons=(),
            cost=_cost(latency=2.0),
        ),
    )
    dense_representation_ids = tuple(
        candidate.candidate_id for candidate in dense_representations
    )
    dense_bindings, dense_peak = _storage_bindings(module)
    structured_values = {
        target_op.inputs[0],
        target_op.inputs[2],
        target_op.outputs[0],
        target_op.outputs[2],
    }
    structured_bindings, structured_peak = _storage_bindings(
        module, structured_values=structured_values
    )
    storages = (
        StorageCandidate(
            candidate_id="storage:dense",
            bindings=dense_bindings,
            compatible_batch_candidate_ids=tuple(
                candidate.candidate_id for candidate in batches
            ),
            compatible_representation_candidate_ids=dense_representation_ids,
            static_legal=True,
            rejection_reasons=(),
            cost=_cost(latency=0.0, peak=dense_peak),
        ),
        StorageCandidate(
            candidate_id="storage:structured",
            bindings=structured_bindings,
            compatible_batch_candidate_ids=tuple(
                candidate.candidate_id for candidate in batches
            ),
            compatible_representation_candidate_ids=(
                *(
                    candidate.candidate_id
                    for candidate in dense_representations
                    if candidate.region_id != target_region_id
                ),
                structured_representation.candidate_id,
            ),
            static_legal=True,
            rejection_reasons=(),
            cost=_cost(latency=0.0, peak=structured_peak),
        ),
    )
    template = PlanTemplate(
        template_id="plan-template-test",
        bound_module_hash=module.stable_hash(),
        planner_config_hash="planner-config-sha256",
        hardware=HardwareProfile(
            profile_id="cpu-test",
            device="cpu",
            total_memory_bytes=1 << 30,
            supported_dtypes=("float32",),
            backend_capability_ids=(
                reference_capability.capability_id,
                structured_capability.capability_id,
            ),
            alignment_bytes=16,
        ),
        workload=WorkloadProfile(
            profile_id="plain-crown-b2-s2",
            method=BoundMethodKind.CROWN,
            requires_grad=False,
            alpha_enabled=False,
            beta_enabled=False,
            split_state_present=False,
            static_shapes=True,
            domain_batch_size=2,
            spec_batch_size=2,
            sample_batch_size=1,
            dtype="float32",
            device="cpu",
            numeric_policy="float32_dense_reference",
        ),
        capabilities=(reference_capability, structured_capability),
        region_candidates=regions,
        representation_candidates=representations,
        materialization_candidates=transitions,
        backend_candidates=(*dense_backends, structured_backend),
        batch_candidates=batches,
        storage_candidates=storages,
        provenance=(PlanProvenance("test_case", "plan_ir_v1"),),
    )
    selected_representation_ids = tuple(
        (
            structured_representation.candidate_id
            if candidate.region_id == target_region_id
            else candidate.candidate_id
        )
        for candidate in dense_representations
    )
    selected_backend_ids = tuple(
        (
            structured_backend.candidate_id
            if candidate.region_id == target_region_id
            else candidate.candidate_id
        )
        for candidate in dense_backends
    )
    selected_ids = {
        *(region.candidate_id for region in regions),
        *selected_representation_ids,
        *(transition.candidate_id for transition in transitions),
        *selected_backend_ids,
        batches[0].candidate_id,
        storages[1].candidate_id,
    }
    rejected = tuple(
        RejectedCandidate(
            candidate_id=candidate.candidate_id,
            reasons=("not_selected_by_test_policy",),
        )
        for candidate in template.all_candidates()
        if candidate.candidate_id not in selected_ids
    )
    instance = PlanInstance(
        instance_id="plan-instance-test",
        template_hash=template.stable_hash(bound_module=module),
        query_bucket_id="bucket:plain-crown-b2-s2",
        available_memory_bytes=1 << 29,
        memory_budget_bytes=1 << 28,
        deadline_us=1_000_000,
        region_decisions=tuple(
            RegionDecision(region.region_id, region.candidate_id) for region in regions
        ),
        representation_decisions=tuple(
            RepresentationDecision(region.region_id, candidate_id)
            for region, candidate_id in zip(regions, selected_representation_ids)
        ),
        materialization_decisions=tuple(
            MaterializationDecision(candidate.candidate_id) for candidate in transitions
        ),
        backend_decisions=tuple(
            BackendDecision(region.region_id, candidate_id)
            for region, candidate_id in zip(regions, selected_backend_ids)
        ),
        batch_decision=BatchDecision(batches[0].candidate_id),
        storage_decision=StorageDecision(storages[1].candidate_id),
        state_decisions=(),
        rejected_candidates=rejected,
        cost_summary=PlanCost(
            predicted_latency_ms=10.0,
            predicted_peak_bytes=structured_peak,
            compile_cost_ms=1.0,
            setup_cost_ms=1.0,
            confidence=0.7,
            risk_tags=("model_estimate",),
        ),
        provenance=(PlanProvenance("selection_policy", "test_structured"),),
    )
    return module, template, instance


def test_plan_ir_v1_template_instance_dump_hash_are_deterministic() -> None:
    module, template, instance = _template_and_instance()
    repeated_module, repeated_template, repeated_instance = _template_and_instance()

    template.validate(bound_module=module)
    instance.validate(template=template, bound_module=module)

    assert template.canonical_json(
        bound_module=module
    ) == repeated_template.canonical_json(bound_module=repeated_module)
    assert template.stable_hash(bound_module=module) == repeated_template.stable_hash(
        bound_module=repeated_module
    )
    assert instance.canonical_json(
        template=template, bound_module=module
    ) == repeated_instance.canonical_json(
        template=repeated_template, bound_module=repeated_module
    )
    assert instance.stable_hash(
        template=template, bound_module=module
    ) == repeated_instance.stable_hash(
        template=repeated_template, bound_module=repeated_module
    )
    replayed = PlanInstance.from_canonical_json(
        instance.canonical_json(template=template, bound_module=module),
        template=template,
        bound_module=module,
    )
    assert replayed == instance


def test_plan_ir_v1_replay_rejects_noncanonical_and_tampered_selection() -> None:
    module, template, instance = _template_and_instance()
    encoded = instance.canonical_json(template=template, bound_module=module)
    pretty = json.dumps(json.loads(encoded), indent=2, sort_keys=True)
    with pytest.raises(ValueError, match="not canonical"):
        PlanInstance.from_canonical_json(pretty, template=template, bound_module=module)

    payload = json.loads(encoded)
    payload["backend_decisions"][0]["candidate_id"] = "unknown"
    tampered = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with pytest.raises(ValueError, match="unknown candidate"):
        PlanInstance.from_canonical_json(
            tampered, template=template, bound_module=module
        )


def test_plan_ir_v1_requires_full_candidate_accounting() -> None:
    module, template, instance = _template_and_instance()
    incomplete = replace(
        instance, rejected_candidates=instance.rejected_candidates[:-1]
    )

    with pytest.raises(ValueError, match="account for every candidate"):
        incomplete.validate(template=template, bound_module=module)


def test_plan_ir_v1_rejects_backend_representation_and_memory_conflicts() -> None:
    module, template, instance = _template_and_instance()
    structured_decision_index = next(
        index
        for index, decision in enumerate(instance.representation_decisions)
        if "structured" in decision.candidate_id
    )
    region_id = instance.representation_decisions[structured_decision_index].region_id
    dense_backend = next(
        candidate
        for candidate in template.backend_candidates
        if candidate.region_id == region_id
        and candidate.backend == BackendKind.REFERENCE
    )
    changed_backends = tuple(
        (
            replace(decision, candidate_id=dense_backend.candidate_id)
            if decision.region_id == region_id
            else decision
        )
        for decision in instance.backend_decisions
    )
    selected_now = {
        *(decision.candidate_id for decision in instance.region_decisions),
        *(decision.candidate_id for decision in instance.representation_decisions),
        *(decision.candidate_id for decision in instance.materialization_decisions),
        *(decision.candidate_id for decision in changed_backends),
        instance.batch_decision.candidate_id,
        instance.storage_decision.candidate_id,
    }
    incompatible = replace(
        instance,
        backend_decisions=changed_backends,
        rejected_candidates=tuple(
            RejectedCandidate(candidate.candidate_id, ("not_selected",))
            for candidate in template.all_candidates()
            if candidate.candidate_id not in selected_now
        ),
    )
    with pytest.raises(ValueError, match="backend/representation"):
        incompatible.validate(template=template, bound_module=module)

    over_budget = replace(instance, memory_budget_bytes=1)
    with pytest.raises(ValueError, match="memory budget"):
        over_budget.validate(template=template, bound_module=module)


def test_plan_ir_v1_rejects_unsafe_storage_alias_and_underallocation() -> None:
    module, template, _instance = _template_and_instance()
    storage = template.storage_candidates[0]
    left, right = storage.bindings[:2]
    alias_bindings = (
        left,
        replace(
            right,
            offset_bytes=left.offset_bytes,
        ),
        *storage.bindings[2:],
    )
    alias_storage = replace(storage, bindings=alias_bindings)
    alias_template = replace(
        template,
        storage_candidates=(alias_storage, *template.storage_candidates[1:]),
    )
    with pytest.raises(ValueError, match="aliases simultaneously live"):
        alias_template.validate(bound_module=module)

    underallocated = replace(
        storage.bindings[0],
        size_bytes=storage.bindings[0].logical_size_bytes - 1,
    )
    bad_storage = replace(storage, bindings=(underallocated, *storage.bindings[1:]))
    bad_template = replace(
        template,
        storage_candidates=(bad_storage, *template.storage_candidates[1:]),
    )
    with pytest.raises(ValueError, match="dense storage"):
        bad_template.validate(bound_module=module)


def test_plan_ir_v1_rejects_bound_hash_and_partition_gaps() -> None:
    module, template, instance = _template_and_instance()
    stale = replace(template, bound_module_hash="stale")
    with pytest.raises(ValueError, match="Bound IR hash mismatch"):
        stale.validate(bound_module=module)

    missing_region = instance.region_decisions[:-1]
    selected_ids = {
        *(decision.candidate_id for decision in missing_region),
        *(decision.candidate_id for decision in instance.representation_decisions),
        *(decision.candidate_id for decision in instance.materialization_decisions),
        *(decision.candidate_id for decision in instance.backend_decisions),
        instance.batch_decision.candidate_id,
        instance.storage_decision.candidate_id,
    }
    partition_gap = replace(
        instance,
        region_decisions=missing_region,
        rejected_candidates=tuple(
            RejectedCandidate(candidate.candidate_id, ("not_selected",))
            for candidate in template.all_candidates()
            if candidate.candidate_id not in selected_ids
        ),
    )
    with pytest.raises(ValueError, match="partition does not cover"):
        partition_gap.validate(template=template, bound_module=module)


def test_plan_ir_core_has_no_legacy_planner_runtime_or_torch_dependency() -> None:
    source = Path("boundflow/ir/plan.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    direct_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "torch" not in direct_imports
    assert not any("runtime" in module for module in imported_modules)
    assert not any("planner" in module for module in imported_modules)
