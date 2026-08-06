"""Native first-class IR execution behind the RVIR-v3 replacement API."""

# pylint: disable=missing-function-docstring,assignment-from-no-return
# pylint: disable=unpacking-non-sequence

from __future__ import annotations

import torch

from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.rvir_v3_native_crown import (
    NativePlainCrownRVIRV3Backend,
    build_native_initial_crown_payload,
)
from boundflow.runtime.rvir_v3_replacement import (
    VerifierTensorRole,
    execute_rvir_v3_replacement,
)
from tests.test_task_ir_v1 import _semantic_case


def test_native_residual_replacement_executes_without_provider_callback() -> None:
    module, input_spec = _semantic_case("residual")
    input_lower, input_upper = input_spec.perturbation.bounding_box(input_spec.center)
    _interval_env, relu_pre = _forward_ibp_trace_mlp(module, input_spec)
    linear_spec = torch.tensor(
        [[1.0, -1.0, 0.5], [-0.5, 0.25, 1.0]], dtype=torch.float32
    )
    payload = build_native_initial_crown_payload(
        query_id="native-rvir-v3-residual-0",
        sequence_number=0,
        parent_query_id=None,
        module=module,
        input_lower=input_lower,
        input_upper=input_upper,
        linear_spec_c=linear_spec,
        intermediate_lowers=[item.lower for item in relu_pre.values()],
        intermediate_uppers=[item.upper for item in relu_pre.values()],
        requested_polarities=("lower", "upper"),
    )
    backend = NativePlainCrownRVIRV3Backend(module, input_spec.value_name)

    first = execute_rvir_v3_replacement(payload, backend)
    second = execute_rvir_v3_replacement(payload, backend)

    torch.testing.assert_close(first.lower, second.lower)
    torch.testing.assert_close(first.upper, second.upper)
    assert first.result_hash == second.result_hash
    assert first.original_callback_count == 0
    assert first.fallback_dispatch_count == 0
    assert backend.last_ir_hashes is not None
    assert len(backend.last_ir_hashes) == 5


def test_native_replacement_rejects_program_and_intermediate_mutation() -> None:
    module, input_spec = _semantic_case("residual")
    input_lower, input_upper = input_spec.perturbation.bounding_box(input_spec.center)
    _interval_env, relu_pre = _forward_ibp_trace_mlp(module, input_spec)
    payload = build_native_initial_crown_payload(
        query_id="native-rvir-v3-residual-1",
        sequence_number=1,
        parent_query_id="native-rvir-v3-residual-0",
        module=module,
        input_lower=input_lower,
        input_upper=input_upper,
        linear_spec_c=torch.ones((1, 3)),
        intermediate_lowers=[item.lower for item in relu_pre.values()],
        intermediate_uppers=[item.upper for item in relu_pre.values()],
        requested_polarities=("lower",),
    )
    backend = NativePlainCrownRVIRV3Backend(module, input_spec.value_name)
    parameters = module.bindings["params"]
    first_parameter = next(iter(parameters.values()))
    first_parameter.add_(1)

    try:
        try:
            execute_rvir_v3_replacement(payload, backend)
        except ValueError as error:
            assert "program parameter differs" in str(error)
        else:
            raise AssertionError("mutated native program was accepted")
    finally:
        first_parameter.sub_(1)

    intermediate = payload.tensors_with_role(VerifierTensorRole.INTERMEDIATE_LOWER)[0]
    intermediate.value.add_(1)
    try:
        try:
            execute_rvir_v3_replacement(payload, backend)
        except ValueError as error:
            assert "content differs" in str(error)
        else:
            raise AssertionError("mutated intermediate payload was accepted")
    finally:
        intermediate.value.sub_(1)
