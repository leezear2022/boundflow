"""IR-4B typed PyTorch backend registry and fused-region execution."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from boundflow.frontends.plain_crown_bound_ir import build_plain_crown_bound_ir
from boundflow.ir.bound import BoundRepresentation
from boundflow.ir.bound_rewrite import rewrite_plain_crown_structured_regions
from boundflow.ir.plan import (
    BackendCandidate,
    BackendCapabilitySpec,
    BackendKind,
    HardwareProfile,
    PlanCost,
    PlanProvenance,
    WorkloadProfile,
)
from boundflow.ir.schedule import (
    FallbackAction,
    LaunchAction,
    RetryAction,
    lower_plan_instance_to_reference_schedule,
)
from boundflow.ir.task_v1 import lower_plan_instance_to_task_ir
from boundflow.planner.plan_ir_builder import (
    BackendEvidence,
    BatchEvidence,
    ReferencePlanEvidence,
    RegionEvidence,
    RepresentationEvidence,
    StorageEvidence,
    build_reference_plan_template,
)
from boundflow.planner.plan_ir_selector import select_plan_instance
from boundflow.runtime.bound_ir_interpreter import execute_plain_crown_bound_ir
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.task_backend_dispatch import (
    PyTorchTaskBackendRegistry,
    TVMTaskBackendRegistry,
    TypedTaskBackendRegistry,
)
from boundflow.runtime.task_executor import InputSpec
from boundflow.runtime.task_ir_executor import execute_task_ir_semantics
from boundflow.runtime.schedule_ir_executor import ScheduleOutOfMemoryError
from tests.test_task_ir_v1 import _semantic_case


def _cost(latency: float) -> PlanCost:
    return PlanCost(
        predicted_latency_ms=latency,
        predicted_peak_bytes=0,
        compile_cost_ms=0.0,
        setup_cost_ms=0.0,
        confidence=1.0,
        risk_tags=("ir4b_test_evidence",),
    )


def _typed_template(
    bound_module,
    *,
    backend_kind: BackendKind,
    device: str,
    fused_pair: tuple[str, str] | None,
):
    single_regions = tuple(
        RegionEvidence(f"single:{op.op_id}", (op.op_id,), _cost(1.0))
        for op in bound_module.graph.ops
    )
    fused_regions = (
        ()
        if fused_pair is None
        else (RegionEvidence("fused:relu-affine", fused_pair, _cost(0.01)),)
    )
    regions = (*single_regions, *fused_regions)
    representations = tuple(
        RepresentationEvidence(
            evidence_id=f"dense:{region.evidence_id}",
            region_evidence_id=region.evidence_id,
            representation=BoundRepresentation.DENSE,
            required_transition_evidence_ids=(),
            cost=_cost(0.0),
        )
        for region in regions
    )
    reference_capability = BackendCapabilitySpec(
        capability_id="reference-fallback-v1",
        backend=BackendKind.REFERENCE,
        supported_methods=(bound_module.domain.method,),
        supported_op_kinds=tuple(
            dict.fromkeys(op.kind for op in bound_module.graph.ops)
        ),
        supported_representations=(BoundRepresentation.DENSE,),
        supported_dtypes=("float32",),
        supported_devices=(device,),
        supports_grad=False,
        supports_alpha=False,
        supports_beta=False,
        supports_split_state=False,
        static_shape_only=True,
    )
    selected_capability = replace(
        reference_capability,
        capability_id=f"{backend_kind.value}-v1",
        backend=backend_kind,
    )
    capabilities = (
        (selected_capability,)
        if fused_pair is None
        else (reference_capability, selected_capability)
    )
    backends: list[BackendEvidence] = []
    for region in single_regions:
        capability = selected_capability if fused_pair is None else reference_capability
        backends.append(
            BackendEvidence(
                evidence_id=f"{capability.backend.value}:{region.evidence_id}",
                region_evidence_id=region.evidence_id,
                representation_evidence_id=f"dense:{region.evidence_id}",
                capability_id=capability.capability_id,
                compiled_artifact_key=(
                    None
                    if capability.backend == BackendKind.REFERENCE
                    else f"typed:{capability.backend.value}:{region.evidence_id}"
                ),
                cost=_cost(0.1),
            )
        )
    if fused_pair is not None:
        backends.append(
            BackendEvidence(
                evidence_id=f"{backend_kind.value}:fused",
                region_evidence_id="fused:relu-affine",
                representation_evidence_id="dense:fused:relu-affine",
                capability_id=selected_capability.capability_id,
                compiled_artifact_key=f"typed:{backend_kind.value}:fused",
                cost=_cost(0.01),
            )
        )
    objective_shape = bound_module.graph.values[0].tensor_type.shape
    domain_batch = int(objective_shape[0] or 1)
    spec_batch = int(objective_shape[1] or 1)
    hardware = HardwareProfile(
        profile_id=f"ir4b-{device}",
        device=device,
        total_memory_bytes=1 << 30,
        supported_dtypes=("float32",),
        backend_capability_ids=tuple(
            capability.capability_id for capability in capabilities
        ),
        alignment_bytes=16,
    )
    workload = WorkloadProfile(
        profile_id=f"ir4b-{backend_kind.value}",
        method=bound_module.domain.method,
        requires_grad=False,
        alpha_enabled=False,
        beta_enabled=False,
        split_state_present=False,
        static_shapes=True,
        domain_batch_size=domain_batch,
        spec_batch_size=spec_batch,
        sample_batch_size=1,
        dtype="float32",
        device=device,
        numeric_policy="float32_typed_backend_test",
    )
    evidence = ReferencePlanEvidence(
        evidence_set_id=f"ir4b-{backend_kind.value}",
        regions=regions,
        transitions=(),
        representations=representations,
        backends=tuple(backends),
        batches=(
            BatchEvidence(
                "full",
                domain_batch,
                spec_batch,
                1,
                64,
                _cost(0.0),
            ),
        ),
        storage=(
            StorageEvidence(
                "all",
                ("full",),
                tuple(item.evidence_id for item in representations),
                (),
                f"{device}-main",
                _cost(0.0),
            ),
        ),
        provenance=(PlanProvenance("test_scope", "ir4b_typed_registry"),),
    )
    return build_reference_plan_template(
        bound_module,
        hardware=hardware,
        workload=workload,
        capabilities=capabilities,
        evidence=evidence,
    )


def _execute_typed(
    legacy_module,
    input_spec: InputSpec,
    *,
    backend_kind: BackendKind,
    structured: bool = False,
    cache_dir: Path | None = None,
):
    interval_env, relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    dense = build_plain_crown_bound_ir(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    ).module
    bound_module = (
        rewrite_plain_crown_structured_regions(dense) if structured else dense
    )
    pair = None
    if backend_kind in {
        BackendKind.PYTORCH_DENSE,
        BackendKind.PYTORCH_CHUNKED,
        BackendKind.TVM_FUSED_TIR,
        BackendKind.TVM_TIR_UNFUSED,
    }:
        for index, op in enumerate(bound_module.graph.ops[:-1]):
            next_op = bound_module.graph.ops[index + 1]
            if op.kind.value == "relu_relaxation" and next_op.kind.value in {
                "linear_backward",
                "conv2d_backward",
            }:
                pair = (op.op_id, next_op.op_id)
                break
        assert pair is not None
    template = _typed_template(
        bound_module,
        backend_kind=backend_kind,
        device=input_spec.center.device.type,
        fused_pair=pair,
    )
    instance = select_plan_instance(
        template,
        bound_module=bound_module,
        query_bucket_id=f"ir4b-{backend_kind.value}",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    task_module = lower_plan_instance_to_task_ir(
        bound_module, template=template, instance=instance
    )
    schedule = lower_plan_instance_to_reference_schedule(
        bound_module,
        template=template,
        instance=instance,
        query_ids=("query:0",),
    )
    registry = (
        TVMTaskBackendRegistry(cache_dir=cache_dir)
        if backend_kind in {BackendKind.TVM_FUSED_TIR, BackendKind.TVM_TIR_UNFUSED}
        else PyTorchTaskBackendRegistry(chunk_rows=2)
    )
    actual, trace = execute_task_ir_semantics(
        task_module,
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        backend=registry,
    )
    expected = execute_plain_crown_bound_ir(
        bound_module,
        task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    return actual, expected, task_module, trace, registry


@pytest.mark.parametrize("case", ["mlp", "cnn"])
def test_typed_dense_fused_region_matches_whole_bound(case: str) -> None:
    module, spec = _semantic_case(case)
    actual, expected, task_module, trace, registry = _execute_typed(
        module, spec, backend_kind=BackendKind.PYTORCH_DENSE
    )
    torch.testing.assert_close(actual.lower, expected.lower, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.upper, expected.upper, atol=2e-6, rtol=2e-6)
    fused = [task for task in task_module.tasks if len(task.op_refs) == 2]
    assert len(fused) == 1
    assert fused[0].backend.reference_implementation_id == (
        "pytorch_dense_bound_region/v1"
    )
    assert registry.cache_misses == 1
    assert any(event.task_id == fused[0].task_id for event in trace.events)


def test_typed_structured_registry_executes_explicit_operator_ir() -> None:
    module, spec = _semantic_case("mlp")
    actual, expected, task_module, _trace, registry = _execute_typed(
        module,
        spec,
        backend_kind=BackendKind.PYTORCH_STRUCTURED,
        structured=True,
    )
    torch.testing.assert_close(actual.lower, expected.lower, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.upper, expected.upper, atol=2e-6, rtol=2e-6)
    assert all(
        task.backend.reference_implementation_id == "pytorch_structured_bound_region/v1"
        for task in task_module.tasks
    )
    assert registry.cache_misses == len(task_module.tasks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for chunked")
def test_typed_chunked_fused_region_matches_whole_bound_cuda() -> None:
    module, _spec = _semantic_case("mlp")
    params = {name: value.cuda() for name, value in module.bindings["params"].items()}
    module = replace(module, bindings={"params": params})
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(2, 4, device="cuda"),
        eps=0.2,
    )
    actual, expected, task_module, _trace, registry = _execute_typed(
        module, spec, backend_kind=BackendKind.PYTORCH_CHUNKED
    )
    torch.testing.assert_close(actual.lower, expected.lower, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual.upper, expected.upper, atol=2e-5, rtol=2e-5)
    fused = [task for task in task_module.tasks if len(task.op_refs) == 2]
    assert fused[0].backend.reference_implementation_id == (
        "pytorch_chunked_fused_relu_affine/v1"
    )
    assert registry.cache_misses == 1


def test_chunked_registry_rejects_nonfused_task() -> None:
    module, spec = _semantic_case("mlp")
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
    bound_module = build_plain_crown_bound_ir(
        module, spec, interval_env=interval_env, relu_pre=relu_pre
    ).module
    template = _typed_template(
        bound_module,
        backend_kind=BackendKind.PYTORCH_CHUNKED,
        device="cpu",
        fused_pair=None,
    )
    instance = select_plan_instance(
        template,
        bound_module=bound_module,
        query_bucket_id="ir4b-invalid-chunked",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    task_module = lower_plan_instance_to_task_ir(
        bound_module, template=template, instance=instance
    )
    schedule = lower_plan_instance_to_reference_schedule(
        bound_module,
        template=template,
        instance=instance,
        query_ids=("query:0",),
    )
    with pytest.raises(ValueError, match="requires fused"):
        execute_task_ir_semantics(
            task_module,
            schedule,
            bound_module=bound_module,
            template=template,
            instance=instance,
            legacy_task_module=module,
            input_spec=spec,
            relu_pre=relu_pre,
            backend=PyTorchTaskBackendRegistry(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TVM")
@pytest.mark.parametrize(
    ("case", "backend_kind"),
    [
        ("mlp", BackendKind.TVM_FUSED_TIR),
        ("cnn", BackendKind.TVM_FUSED_TIR),
        ("mlp", BackendKind.TVM_TIR_UNFUSED),
        ("cnn", BackendKind.TVM_TIR_UNFUSED),
    ],
)
def test_typed_tvm_fused_and_unfused_match_whole_bound_cuda(
    tmp_path: Path, case: str, backend_kind: BackendKind
) -> None:
    module, _spec = _semantic_case(case)
    params = {name: value.cuda() for name, value in module.bindings["params"].items()}
    if case == "cnn":
        params["Wc"] = torch.randn(2, 1, 3, 3, device="cuda")
        params["Wl"] = torch.randn(3, 18, device="cuda")
    module = replace(module, bindings={"params": params})
    if case == "mlp":
        spec = InputSpec.linf(
            value_name="input",
            center=torch.randn(2, 4, device="cuda"),
            eps=0.2,
        )
    else:
        spec = InputSpec.box(
            value_name="input",
            lower=torch.full((1, 1, 5, 5), -0.4, device="cuda"),
            upper=torch.full((1, 1, 5, 5), 0.6, device="cuda"),
        )
    actual, expected, task_module, _trace, registry = _execute_typed(
        module,
        spec,
        backend_kind=backend_kind,
        cache_dir=tmp_path / "cache",
    )
    torch.testing.assert_close(actual.lower, expected.lower, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual.upper, expected.upper, atol=2e-4, rtol=2e-4)
    fused = [task for task in task_module.tasks if len(task.op_refs) == 2]
    assert len(fused) == 1
    assert fused[0].backend.reference_implementation_id.startswith(
        "tvm_fused_tir"
        if backend_kind == BackendKind.TVM_FUSED_TIR
        else "tvm_tir_unfused"
    )
    assert registry.cache_misses == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TVM")
def test_typed_tvm_fused_cache_uses_dispatch_namespace_and_disk_replay(
    tmp_path: Path,
) -> None:
    module, _spec = _semantic_case("mlp")
    params = {name: value.cuda() for name, value in module.bindings["params"].items()}
    module = replace(module, bindings={"params": params})
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(2, 4, device="cuda"),
        eps=0.2,
    )
    cache_dir = tmp_path / "cache"
    first = _execute_typed(
        module,
        spec,
        backend_kind=BackendKind.TVM_FUSED_TIR,
        cache_dir=cache_dir,
    )
    second = _execute_typed(
        module,
        spec,
        backend_kind=BackendKind.TVM_FUSED_TIR,
        cache_dir=cache_dir,
    )
    first_registry = first[4]
    second_registry = second[4]
    assert isinstance(first_registry, TVMTaskBackendRegistry)
    assert isinstance(second_registry, TVMTaskBackendRegistry)
    assert first_registry.cache is not None
    assert second_registry.cache is not None
    assert first_registry.cache.events[0].event == "miss"
    assert second_registry.cache.events[0].event == "disk_hit"
    manifest = next(cache_dir.glob("*.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    dispatch_key = payload["cache_payload"]["backend_dispatch_key"]
    assert len(dispatch_key) == 64
    assert payload["schema_version"] == "boundflow.fused_crown_cache/v2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TVM")
def test_typed_tvm_cache_replays_from_fresh_python_process(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    program = r"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
from boundflow.ir.plan import BackendKind
from boundflow.runtime.task_executor import InputSpec
from tests.test_task_backend_dispatch_v1 import _execute_typed
from tests.test_task_ir_v1 import _semantic_case

module, _ = _semantic_case("mlp")
params = {name: value.cuda() for name, value in module.bindings["params"].items()}
module = replace(module, bindings={"params": params})
spec = InputSpec.linf(
    value_name="input",
    center=torch.randn(2, 4, device="cuda"),
    eps=0.2,
)
result = _execute_typed(
    module,
    spec,
    backend_kind=BackendKind.TVM_FUSED_TIR,
    cache_dir=Path(sys.argv[1]),
)
registry = result[4]
print(json.dumps({
    "event": registry.cache.events[0].event,
    "cache_key": registry.cache.events[0].cache_key,
}, sort_keys=True))
"""
    first = subprocess.run(
        [sys.executable, "-c", program, str(cache_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [sys.executable, "-c", program, str(cache_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    first_payload = json.loads(first.stdout.strip().splitlines()[-1])
    second_payload = json.loads(second.stdout.strip().splitlines()[-1])
    assert first_payload["event"] == "miss"
    assert second_payload["event"] == "disk_hit"
    assert first_payload["cache_key"] == second_payload["cache_key"]


class _SelectedBackendOom:
    def __init__(self, selected_candidate_id: str) -> None:
        self.selected_candidate_id = selected_candidate_id
        self.registry = TypedTaskBackendRegistry()

    def dispatch(self, task, key, *, session, template):
        if key.backend_candidate_id == self.selected_candidate_id:
            raise ScheduleOutOfMemoryError("injected semantic backend OOM")
        return self.registry.dispatch(
            task,
            key,
            session=session,
            template=template,
        )


def test_schedule_retry_executes_typed_semantic_fallback_and_records_attempts() -> None:
    module, spec = _semantic_case("mlp")
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
    bound_module = build_plain_crown_bound_ir(
        module, spec, interval_env=interval_env, relu_pre=relu_pre
    ).module
    relu_index = next(
        index
        for index, op in enumerate(bound_module.graph.ops[:-1])
        if op.kind.value == "relu_relaxation"
    )
    pair = (
        bound_module.graph.ops[relu_index].op_id,
        bound_module.graph.ops[relu_index + 1].op_id,
    )
    template = _typed_template(
        bound_module,
        backend_kind=BackendKind.PYTORCH_DENSE,
        device="cpu",
        fused_pair=pair,
    )
    selected = next(
        candidate
        for candidate in template.backend_candidates
        if candidate.backend == BackendKind.PYTORCH_DENSE
    )
    reference_capability = next(
        capability
        for capability in template.capabilities
        if capability.backend == BackendKind.REFERENCE
    )
    fallback_backend = BackendCandidate(
        candidate_id="backend:reference:fused-fallback",
        region_id=selected.region_id,
        backend=BackendKind.REFERENCE,
        capability_id=reference_capability.capability_id,
        compatible_representation_candidate_ids=(
            selected.compatible_representation_candidate_ids
        ),
        compiled_artifact_key=None,
        static_legal=True,
        rejection_reasons=(),
        cost=_cost(100.0),
    )
    template = replace(
        template,
        backend_candidates=(*template.backend_candidates, fallback_backend),
    )
    instance = select_plan_instance(
        template,
        bound_module=bound_module,
        query_bucket_id="ir4c-semantic-fallback",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    task_module = lower_plan_instance_to_task_ir(
        bound_module, template=template, instance=instance
    )
    schedule = lower_plan_instance_to_reference_schedule(
        bound_module,
        template=template,
        instance=instance,
        query_ids=("query:0", "query:1"),
    )
    launch_index = next(
        index
        for index, action in enumerate(schedule.actions)
        if isinstance(action, LaunchAction)
        and action.backend_candidate_id == selected.candidate_id
    )
    launch = schedule.actions[launch_index]
    assert isinstance(launch, LaunchAction)
    retry = RetryAction(
        action_id="retry:semantic",
        launch_action_id=launch.action_id,
        fallback_action_ids=("fallback:semantic",),
        max_attempts=2,
        retry_on=("oom",),
    )
    fallback = FallbackAction(
        action_id="fallback:semantic",
        retry_action_id=retry.action_id,
        backend_candidate_id=fallback_backend.candidate_id,
        reason="selected_backend_oom",
    )
    schedule = replace(
        schedule,
        actions=(
            *schedule.actions[:launch_index],
            retry,
            fallback,
            *schedule.actions[launch_index:],
        ),
    )
    actual, trace = execute_task_ir_semantics(
        task_module,
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
        legacy_task_module=module,
        input_spec=spec,
        relu_pre=relu_pre,
        backend=_SelectedBackendOom(selected.candidate_id),
    )
    expected = execute_plain_crown_bound_ir(
        bound_module,
        task_module=module,
        input_spec=spec,
        relu_pre=relu_pre,
    )
    torch.testing.assert_close(actual.lower, expected.lower)
    torch.testing.assert_close(actual.upper, expected.upper)
    fused_event = next(
        event
        for event in trace.events
        if event.attempted_backend_candidate_ids
        == (selected.candidate_id, fallback_backend.candidate_id)
    )
    assert fused_event.backend_candidate_id == fallback_backend.candidate_id
    assert fused_event.reference_implementation_id == ("bound_ir_region_reference/v1")
    assert schedule.query_ids == ("query:0", "query:1")
