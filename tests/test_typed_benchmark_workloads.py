"""IR-5 measured workload construction contracts."""

# pylint: disable=duplicate-code,missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.plan import BackendKind
from boundflow.ir.schedule import lower_plan_instance_to_reference_schedule
from boundflow.planner.typed_benchmark_workloads import (
    build_cnn_candidate,
    build_mlp_candidate,
    build_residual_cnn_candidate,
)
from boundflow.runtime.task_backend_dispatch import PyTorchTaskBackendRegistry
from boundflow.runtime.task_ir_executor import (
    TaskTraceMode,
    execute_task_ir_semantics,
    prepare_task_ir_execution,
)


def test_ir5_mlp_candidate_executes_typed_reference_and_dense() -> None:
    results = []
    hashes = []
    for backend in (BackendKind.REFERENCE, BackendKind.PYTORCH_DENSE):
        prepared = build_mlp_candidate(
            workload_id="contract-small",
            backend=backend,
            device="cpu",
            batch=2,
            input_dim=4,
            hidden_dim=5,
            output_dim=3,
            seed=71,
        )
        result, _trace = execute_task_ir_semantics(
            prepared.task_module,
            prepared.schedule,
            bound_module=prepared.bound_module,
            template=prepared.template,
            instance=prepared.instance,
            legacy_task_module=prepared.legacy_module,
            input_spec=prepared.input_spec,
            relu_pre=prepared.relu_pre,
            backend=PyTorchTaskBackendRegistry(),
        )
        results.append(result)
        hashes.append(
            prepared.instance.stable_hash(
                template=prepared.template,
                bound_module=prepared.bound_module,
            )
        )
    torch.testing.assert_close(results[0].lower, results[1].lower)
    torch.testing.assert_close(results[0].upper, results[1].upper)
    assert hashes[0] != hashes[1]


def test_ir5_cnn_candidate_executes_typed_reference_and_dense() -> None:
    results = []
    task_kinds = []
    for backend in (BackendKind.REFERENCE, BackendKind.PYTORCH_DENSE):
        prepared = build_cnn_candidate(
            workload_id="cnn-contract",
            backend=backend,
            device="cpu",
            batch=1,
            input_channels=1,
            image_size=4,
            conv1_channels=2,
            conv2_channels=3,
            output_dim=2,
            seed=72,
        )
        result, _trace = execute_task_ir_semantics(
            prepared.task_module,
            prepared.schedule,
            bound_module=prepared.bound_module,
            template=prepared.template,
            instance=prepared.instance,
            legacy_task_module=prepared.legacy_module,
            input_spec=prepared.input_spec,
            relu_pre=prepared.relu_pre,
            backend=PyTorchTaskBackendRegistry(),
        )
        results.append(result)
        task_kinds.append(
            tuple(
                op_ref.kind.value
                for task in prepared.task_module.tasks
                for op_ref in task.op_refs
            )
        )
    torch.testing.assert_close(results[0].lower, results[1].lower)
    torch.testing.assert_close(results[0].upper, results[1].upper)
    assert "conv2d_backward" in task_kinds[1]


def test_ir5_residual_cnn_candidate_executes_typed_reference_and_dense() -> None:
    results = []
    task_kinds = []
    for backend in (BackendKind.REFERENCE, BackendKind.PYTORCH_DENSE):
        prepared = build_residual_cnn_candidate(
            workload_id="residual-cnn-contract",
            backend=backend,
            device="cpu",
            batch=2,
            input_channels=1,
            image_size=4,
            block_channels=2,
            output_dim=2,
            seed=75,
        )
        result, _trace = execute_task_ir_semantics(
            prepared.task_module,
            prepared.schedule,
            bound_module=prepared.bound_module,
            template=prepared.template,
            instance=prepared.instance,
            legacy_task_module=prepared.legacy_module,
            input_spec=prepared.input_spec,
            relu_pre=prepared.relu_pre,
            backend=PyTorchTaskBackendRegistry(),
        )
        results.append(result)
        task_kinds.append(
            tuple(
                op_ref.kind.value
                for task in prepared.task_module.tasks
                for op_ref in task.op_refs
            )
        )
    torch.testing.assert_close(results[0].lower, results[1].lower)
    torch.testing.assert_close(results[0].upper, results[1].upper)
    assert "add_backward" in task_kinds[1]


def test_residual_single_query_binds_exact_batched_input_slice() -> None:
    batched = build_residual_cnn_candidate(
        workload_id="residual-explicit-batch",
        backend=BackendKind.REFERENCE,
        device="cpu",
        batch=3,
        input_channels=1,
        image_size=5,
        block_channels=2,
        output_dim=3,
        seed=76,
    )
    single = build_residual_cnn_candidate(
        workload_id="residual-explicit-single",
        backend=BackendKind.REFERENCE,
        device="cpu",
        batch=1,
        input_channels=1,
        image_size=5,
        block_channels=2,
        output_dim=3,
        seed=76,
        input_center=batched.input_spec.center[:1],
    )
    assert torch.equal(single.input_spec.center, batched.input_spec.center[:1])
    results = []
    for prepared in (batched, single):
        result, _trace = execute_task_ir_semantics(
            prepared.task_module,
            prepared.schedule,
            bound_module=prepared.bound_module,
            template=prepared.template,
            instance=prepared.instance,
            legacy_task_module=prepared.legacy_module,
            input_spec=prepared.input_spec,
            relu_pre=prepared.relu_pre,
            backend=PyTorchTaskBackendRegistry(),
        )
        results.append(result)
    torch.testing.assert_close(results[0].lower[:1], results[1].lower)
    torch.testing.assert_close(results[0].upper[:1], results[1].upper)


def test_prepared_task_execution_reuses_static_validation_and_rejects_drift() -> None:
    prepared = build_mlp_candidate(
        workload_id="prepared-contract",
        backend=BackendKind.PYTORCH_DENSE,
        device="cpu",
        batch=2,
        input_dim=4,
        hidden_dim=5,
        output_dim=3,
        seed=73,
    )
    execution = prepare_task_ir_execution(
        prepared.task_module,
        prepared.schedule,
        bound_module=prepared.bound_module,
        template=prepared.template,
        instance=prepared.instance,
        legacy_task_module=prepared.legacy_module,
    )
    registry = PyTorchTaskBackendRegistry()
    results = []
    traces = []
    for _index in range(2):
        result, trace = execute_task_ir_semantics(
            prepared.task_module,
            prepared.schedule,
            bound_module=prepared.bound_module,
            template=prepared.template,
            instance=prepared.instance,
            legacy_task_module=prepared.legacy_module,
            input_spec=prepared.input_spec,
            relu_pre=prepared.relu_pre,
            backend=registry,
            prepared=execution,
        )
        results.append(result)
        traces.append(trace)
    torch.testing.assert_close(results[0].lower, results[1].lower)
    torch.testing.assert_close(results[0].upper, results[1].upper)
    assert traces[0].stable_hash() == traces[1].stable_hash()
    assert registry.cache_hits > 0
    production_result, production_trace = execute_task_ir_semantics(
        prepared.task_module,
        prepared.schedule,
        bound_module=prepared.bound_module,
        template=prepared.template,
        instance=prepared.instance,
        legacy_task_module=prepared.legacy_module,
        input_spec=prepared.input_spec,
        relu_pre=prepared.relu_pre,
        backend=registry,
        prepared=execution,
        trace_mode=TaskTraceMode.PRODUCTION,
    )
    torch.testing.assert_close(results[0].lower, production_result.lower)
    torch.testing.assert_close(results[0].upper, production_result.upper)
    assert all(not event.output_value_hashes for event in production_trace.events)
    production_trace.validate()
    dynamic_schedule = lower_plan_instance_to_reference_schedule(
        prepared.bound_module,
        template=prepared.template,
        instance=prepared.instance,
        query_ids=("different-query",),
    )
    _dynamic_result, dynamic_trace = execute_task_ir_semantics(
        prepared.task_module,
        dynamic_schedule,
        bound_module=prepared.bound_module,
        template=prepared.template,
        instance=prepared.instance,
        legacy_task_module=prepared.legacy_module,
        input_spec=prepared.input_spec,
        relu_pre=prepared.relu_pre,
        backend=registry,
        prepared=execution,
        trace_mode=TaskTraceMode.PRODUCTION,
    )
    assert dynamic_trace.schedule_hash != production_trace.schedule_hash
    assert dynamic_trace.schedule_hash == dynamic_schedule.stable_hash(
        bound_module=prepared.bound_module,
        template=prepared.template,
        instance=prepared.instance,
    )

    with pytest.raises(ValueError, match="Schedule structure drift"):
        execute_task_ir_semantics(
            prepared.task_module,
            replace(prepared.schedule, query_ids=("different-query",)),
            bound_module=prepared.bound_module,
            template=prepared.template,
            instance=prepared.instance,
            legacy_task_module=prepared.legacy_module,
            input_spec=prepared.input_spec,
            relu_pre=prepared.relu_pre,
            backend=registry,
            prepared=execution,
        )


def test_prepared_task_execution_snapshots_static_parameters() -> None:
    workload = build_mlp_candidate(
        workload_id="prepared-snapshot",
        backend=BackendKind.PYTORCH_DENSE,
        device="cpu",
        batch=2,
        input_dim=4,
        hidden_dim=5,
        output_dim=3,
        seed=74,
    )
    execution = prepare_task_ir_execution(
        workload.task_module,
        workload.schedule,
        bound_module=workload.bound_module,
        template=workload.template,
        instance=workload.instance,
        legacy_task_module=workload.legacy_module,
    )
    before, _trace = execute_task_ir_semantics(
        workload.task_module,
        workload.schedule,
        bound_module=workload.bound_module,
        template=workload.template,
        instance=workload.instance,
        legacy_task_module=workload.legacy_module,
        input_spec=workload.input_spec,
        relu_pre=workload.relu_pre,
        backend=PyTorchTaskBackendRegistry(),
        prepared=execution,
        trace_mode=TaskTraceMode.PRODUCTION,
    )
    params = workload.legacy_module.bindings["params"]
    assert isinstance(params, dict)
    tensor = next(value for value in params.values() if torch.is_tensor(value))
    tensor.add_(10.0)

    after, _trace = execute_task_ir_semantics(
        workload.task_module,
        workload.schedule,
        bound_module=workload.bound_module,
        template=workload.template,
        instance=workload.instance,
        legacy_task_module=workload.legacy_module,
        input_spec=workload.input_spec,
        relu_pre=workload.relu_pre,
        backend=PyTorchTaskBackendRegistry(),
        prepared=execution,
        trace_mode=TaskTraceMode.PRODUCTION,
    )
    torch.testing.assert_close(before.lower, after.lower)
    torch.testing.assert_close(before.upper, after.upper)
    with pytest.raises(ValueError, match="fingerprint"):
        execute_task_ir_semantics(
            workload.task_module,
            workload.schedule,
            bound_module=workload.bound_module,
            template=workload.template,
            instance=workload.instance,
            legacy_task_module=workload.legacy_module,
            input_spec=workload.input_spec,
            relu_pre=workload.relu_pre,
            backend=PyTorchTaskBackendRegistry(),
        )
