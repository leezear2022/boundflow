"""Typed Bound/Plan/Task/Schedule contracts for external verifier calls."""

from __future__ import annotations

import torch

from boundflow.ir.bound import (
    BoundMethodKind,
    BoundOpKind,
    ExternalVerifierCallAttrs,
)
from boundflow.ir.plan import BackendKind, RegionKind
from boundflow.ir.schedule import EmitResultAction, LaunchAction
from boundflow.ir.task_v1 import TaskIRKind
from boundflow.runtime.verifier_ir_integration import (
    ExternalVerifierCallSpec,
    compile_external_verifier_call,
    execute_external_verifier_call,
)


def _activation_query(*, observed_method: str = "CROWN") -> dict[str, object]:
    return {
        "query_id": "real-activation-00000108",
        "parent_query_id": None,
        "sequence_number": 108,
        "model_structure_hash": "onnx:model",
        "weight_version": "onnx:weights",
        "input_region_hash": "input-region-hash",
        "output_spec_hash": "objective-hash",
        "split_signature": "abcrown:beta-enabled:state-unresolved",
        "bound_method": observed_method,
        "requires_grad": True,
        "alpha_state_version": "alpha-content-hash",
        "beta_state_version": None,
        "cuts_version": None,
        "dtype": "torch.float32",
        "device": "cuda:0",
        "numeric_policy": "fp32_strict",
        "requested_outputs": ["bounds"],
        "compatibility_key": {
            "input_shape": [192, 3, 32, 32],
            "spec_shape": [192, 1, 10],
            "dtype": "torch.float32",
            "device": "cuda:0",
        },
        "execution_options": {
            "solver_phase": "activation_bab_bound",
            "split_state_present": True,
            "bound_lower_requested": True,
            "bound_upper_requested": False,
            "identity_limitations": ["split_state_values_unresolved"],
        },
    }


def test_activation_query_compiles_through_all_ir_layers() -> None:
    spec = ExternalVerifierCallSpec.from_query_dict(_activation_query())
    compiled = compile_external_verifier_call(spec)
    op = compiled.bound_module.graph.ops[0]
    attrs = op.attrs
    task = compiled.task_module.tasks[0]
    region = compiled.template.region_candidates[0]
    backend = compiled.template.backend_candidates[0]
    launches = [
        action
        for action in compiled.schedule.actions
        if isinstance(action, LaunchAction)
    ]
    emits = [
        action
        for action in compiled.schedule.actions
        if isinstance(action, EmitResultAction)
    ]

    assert spec.effective_method == BoundMethodKind.ALPHA_BETA_CROWN
    assert spec.observed_method == "CROWN"
    assert spec.split_present
    assert tuple(value.value for value in spec.requested_bounds) == ("lower",)
    assert op.kind == BoundOpKind.EXTERNAL_VERIFIER_CALL
    assert isinstance(attrs, ExternalVerifierCallAttrs)
    assert attrs.semantics_owner == "external_verifier"
    assert attrs.beta_state_version == (
        "external-live:beta-state:abcrown:beta-enabled:state-unresolved"
    )
    assert region.kind == RegionKind.EXTERNAL_VERIFIER
    assert backend.backend == BackendKind.EXTERNAL_ABCROWN
    assert task.kind == TaskIRKind.EXTERNAL_VERIFIER_CALL
    assert task.backend.reference_implementation_id == (
        "external_abcrown_exact_call/v1"
    )
    assert len(launches) == len(emits) == 1
    assert len(set(compiled.hashes().values())) == 5


def test_external_schedule_executes_exact_provider_once() -> None:
    compiled = compile_external_verifier_call(
        ExternalVerifierCallSpec.from_query_dict(
            _activation_query(observed_method="alpha-beta-CROWN")
        )
    )
    calls = 0

    def exact_call() -> tuple[torch.Tensor, None]:
        nonlocal calls
        calls += 1
        return torch.tensor([[-0.25]]), None

    first = execute_external_verifier_call(compiled, exact_call)
    second = execute_external_verifier_call(compiled, exact_call)

    assert calls == 2
    assert first.query_id == "real-activation-00000108"
    assert first.sequence_number == 108
    assert first.result_hash == second.result_hash
    assert first.ir_hashes == second.ir_hashes
