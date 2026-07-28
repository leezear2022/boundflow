"""IR-5 measured workload construction contracts."""

# pylint: disable=duplicate-code,missing-function-docstring

from __future__ import annotations

import torch

from boundflow.ir.plan import BackendKind
from boundflow.planner.typed_benchmark_workloads import (
    build_cnn_candidate,
    build_mlp_candidate,
)
from boundflow.runtime.task_backend_dispatch import PyTorchTaskBackendRegistry
from boundflow.runtime.task_ir_executor import execute_task_ir_semantics


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
