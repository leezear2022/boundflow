"""CIBC-parity horizontal sparse Conv Triton correctness gates."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals
# pylint: disable=duplicate-code

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_ir_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b2_cibc_triton import (
    CIBC_TRITON_CONFIGS_V2,
    CIBCTritonExecutorV2,
    _CIBCTritonFunctionV2,
    execute_cibc_triton_v2,
)
from boundflow.runtime.fsg4_b4b2_sparse_conv_timing import (
    compare_sparse_conv_executions_v1,
    execute_sparse_conv_pytorch_baseline_v1,
)
from boundflow.runtime.fsg4_b4b2_sparse_conv_tir import (
    build_b4b2_sparse_conv_template_v1,
    build_b4b2_sparse_conv_tensors_v1,
)

ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")


def _prepared(run_ordinal: int = 0):
    payload = torch.load(
        ARTIFACT / f"run_{run_ordinal:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    capture = production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][1]
    )
    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    template = build_b4b2_sparse_conv_template_v1(
        lower_ir, capture, compute_capability="sm_89"
    )
    tensors = build_b4b2_sparse_conv_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    return template, tensors


def test_cibc_triton_frozen_schedule_inventory() -> None:
    assert len(CIBC_TRITON_CONFIGS_V2) == 12
    assert tuple(config.ordinal for config in CIBC_TRITON_CONFIGS_V2) == tuple(
        range(12)
    )
    assert (
        len(
            {
                (config.block_m, config.block_k, config.num_warps)
                for config in CIBC_TRITON_CONFIGS_V2
            }
        )
        == 12
    )
    for config in CIBC_TRITON_CONFIGS_V2:
        config.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cibc_triton_all_schedules_match_public_pytorch_oracle() -> None:
    template, tensors = _prepared()
    baseline = execute_sparse_conv_pytorch_baseline_v1(tensors, template)
    for config in CIBC_TRITON_CONFIGS_V2:
        candidate, executor = execute_cibc_triton_v2(
            tensors, template, config_ordinal=config.ordinal
        )
        metric = compare_sparse_conv_executions_v1(baseline, candidate)
        assert metric.maximum_absolute_difference <= 2.0e-4
        assert metric.allclose and metric.sign_exact
        assert executor.forward_launch_count == 1
        assert executor.backward_launch_count == 1
        assert executor.fallback_count == 0
        assert executor.eager_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cibc_triton_five_raw_match_public_pytorch_oracle() -> None:
    maximum = 0.0
    for run_ordinal in range(5):
        template, tensors = _prepared(run_ordinal)
        baseline = execute_sparse_conv_pytorch_baseline_v1(tensors, template)
        candidate, _executor = execute_cibc_triton_v2(
            tensors, template, config_ordinal=2
        )
        metric = compare_sparse_conv_executions_v1(baseline, candidate)
        maximum = max(maximum, metric.maximum_absolute_difference)
        assert metric.allclose and metric.sign_exact
        assert metric.element_count == 12_810
    assert maximum <= 2.0e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cibc_triton_nonfinite_dtype_shape_and_alias_fail_closed() -> None:
    template, tensors = _prepared()
    nonfinite = tensors.preactivation_lower.detach().clone()
    nonfinite.flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="tensor differs"):
        execute_cibc_triton_v2(
            replace(tensors, preactivation_lower=nonfinite),
            template,
            config_ordinal=2,
        )
    with pytest.raises(ValueError, match="tensor differs"):
        execute_cibc_triton_v2(
            replace(tensors, operator_weight=tensors.operator_weight.to(torch.float16)),
            template,
            config_ordinal=2,
        )
    with pytest.raises(ValueError, match="tensor differs"):
        execute_cibc_triton_v2(
            replace(tensors, operator_bias=tensors.operator_bias[:-1]),
            template,
            config_ordinal=2,
        )
    executor = CIBCTritonExecutorV2(
        template, CIBC_TRITON_CONFIGS_V2[2], device=torch.device("cuda:0")
    )
    with pytest.raises(ValueError, match="aliases differ"):
        executor._validate_call((tensors.operator_weight, tensors.operator_weight))
    assert executor.fallback_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cibc_triton_higher_order_gradient_fails_closed() -> None:
    template, tensors = _prepared()
    executor = CIBCTritonExecutorV2(
        template, CIBC_TRITON_CONFIGS_V2[2], device=torch.device("cuda:0")
    )
    output_a, output_bias = _CIBCTritonFunctionV2.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.compressed_alpha,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        executor,
    )
    with pytest.raises(RuntimeError, match="higher-order"):
        torch.autograd.grad(
            (output_a, output_bias),
            (tensors.incoming_lower_a, tensors.compressed_alpha),
            grad_outputs=(
                tensors.output_lower_a_gradient,
                tensors.output_bias_gradient,
            ),
            create_graph=True,
        )
    assert executor.fallback_count == 1
