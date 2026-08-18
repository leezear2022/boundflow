"""Contracts for the B4-B production differentiable-region capture."""

# pylint: disable=missing-function-docstring

from dataclasses import replace

import pytest
import torch

from boundflow.runtime.fsg4_b4b_production_region_capture import (
    B4B_PERFORMANCE_ANCHOR_V1,
    B4B_SEMANTIC_ANCHOR_V1,
    CapturedCudaTensorV1,
    ProductionDifferentiableRegionCaptureV1,
    b4b_v1_anchors,
    capture_production_differentiable_region_v1,
)
from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

_HASH = "a" * 64


def _snapshot(
    name: str, shape: tuple[int, ...], *, requires_grad: bool = False
) -> CapturedCudaTensorV1:
    value = torch.zeros(shape, dtype=torch.float32)
    return CapturedCudaTensorV1(
        name=name,
        value=value,
        source_shape=shape,
        source_dtype=str(value.dtype),
        source_device="cuda:0",
        source_strides=tuple(value.stride()),
        source_requires_grad=requires_grad,
        content_sha256=production_tensor_sha256(value),
    )


def _capture(*, performance: bool = False) -> ProductionDifferentiableRegionCaptureV1:
    anchor = B4B_PERFORMANCE_ANCHOR_V1 if performance else B4B_SEMANTIC_ANCHOR_V1
    values = {
        "incoming_lower_a": _snapshot(
            "incoming_lower_a", anchor.coefficient_shape, requires_grad=performance
        ),
        "preactivation_lower": _snapshot(
            "preactivation_lower", anchor.preactivation_shape
        ),
        "preactivation_upper": _snapshot(
            "preactivation_upper", anchor.preactivation_shape
        ),
        "production_alpha": _snapshot(
            "production_alpha", anchor.production_alpha_shape
        ),
        "native_alpha": _snapshot(
            "native_alpha", anchor.native_alpha_shape, requires_grad=True
        ),
        "production_beta": _snapshot("production_beta", anchor.production_beta_shape),
        "native_beta": _snapshot(
            "native_beta", anchor.native_beta_shape, requires_grad=True
        ),
        "relu_pre_add_coeff_l": _snapshot(
            "relu_pre_add_coeff_l", anchor.native_beta_shape, requires_grad=True
        ),
        "operator_weight": _snapshot(
            "operator_weight",
            (16, 16, 3, 3) if performance else (100, 1024),
        ),
        "output_lower_a": _snapshot(
            "output_lower_a", (6, 1, 16, 8, 8) if performance else (6, 1, 1024)
        ),
        "output_bias": _snapshot("output_bias", (6, 1)),
        "loss_seed": _snapshot("loss_seed", (6, 1)),
    }
    gradients = {
        "native_alpha": _snapshot("native_alpha", anchor.native_alpha_shape),
        "native_beta": _snapshot("native_beta", anchor.native_beta_shape),
    }
    if performance:
        gradients["incoming_lower_a"] = _snapshot(
            "incoming_lower_a", anchor.coefficient_shape
        )
    attributes: dict[str, object] = {
        "operator_kind": anchor.producer_op_type,
        "weight_shape": list(values["operator_weight"].source_shape),
    }
    if performance:
        attributes.update(
            {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1}
        )
    return ProductionDifferentiableRegionCaptureV1(
        source_state_hash=_HASH,
        primal_graph_hash=_HASH,
        split_state_hash=_HASH,
        topology_hash=_HASH,
        anchor=anchor,
        values=tuple(sorted(values.items())),
        gradients=tuple(sorted(gradients.items())),
        operator_attributes=tuple(sorted(attributes.items())),
    )


def test_b4b_v1_freezes_semantic_and_performance_anchors() -> None:
    semantic, performance = b4b_v1_anchors()
    assert semantic.anchor_id == "semantic-active-beta-gemm-14"
    assert semantic.production_beta_shape == (6, 1)
    assert semantic.native_beta_shape == (6, 100)
    assert semantic.beta_must_be_nonempty is True
    assert performance.anchor_id == "performance-conv-8-candidate"
    assert performance.production_beta_shape == (6, 0)
    assert performance.native_beta_shape == (6, 16, 8, 8)
    assert performance.beta_must_be_nonempty is False
    assert semantic.stable_hash() != performance.stable_hash()


@pytest.mark.parametrize("performance", [False, True])
def test_b4b_capture_accepts_both_preregistered_anchors(performance: bool) -> None:
    capture = _capture(performance=performance)
    capture.validate()
    assert capture.metadata()["capture_hash"]


def test_b4b_capture_rejects_nonzero_evaluation_ordinal() -> None:
    with pytest.raises(ValueError, match="production capture differs"):
        replace(_capture(), evaluation_ordinal=1).validate()


def test_b4b_capture_rejects_missing_beta_gradient() -> None:
    capture = _capture()
    gradients = tuple(item for item in capture.gradients if item[0] != "native_beta")
    with pytest.raises(ValueError, match="production capture differs"):
        replace(capture, gradients=gradients).validate()


def test_b4b_capture_rejects_spurious_incoming_gradient() -> None:
    capture = _capture()
    gradients = dict(capture.gradients)
    gradients["incoming_lower_a"] = _snapshot(
        "incoming_lower_a", capture.anchor.coefficient_shape
    )
    with pytest.raises(ValueError, match="gradient ownership differs"):
        replace(capture, gradients=tuple(sorted(gradients.items()))).validate()


def test_b4b_capture_rejects_empty_semantic_beta() -> None:
    capture = _capture()
    values = dict(capture.values)
    values["production_beta"] = _snapshot("production_beta", (6, 0))
    with pytest.raises(ValueError, match="value tensor shape differs"):
        replace(capture, values=tuple(sorted(values.items()))).validate()


def test_b4b_capture_rejects_incomplete_conv_attributes() -> None:
    capture = _capture(performance=True)
    attributes = dict(capture.operator_attributes)
    del attributes["stride"]
    with pytest.raises(ValueError, match="Conv attributes are incomplete"):
        replace(
            capture, operator_attributes=tuple(sorted(attributes.items()))
        ).validate()


def test_b4b_capture_rejects_resigned_tensor_content() -> None:
    capture = _capture()
    values = dict(capture.values)
    incoming = values["incoming_lower_a"]
    tampered_value = incoming.value.clone()
    tampered_value.reshape(-1)[0] = 1.0
    values["incoming_lower_a"] = replace(incoming, value=tampered_value)
    with pytest.raises(ValueError, match="captured tensor differs"):
        replace(capture, values=tuple(sorted(values.items()))).validate()


def test_b4b_live_capture_rejects_cpu_placeholder() -> None:
    tensor = torch.zeros(B4B_SEMANTIC_ANCHOR_V1.coefficient_shape)
    with pytest.raises(ValueError, match="production CUDA tensor"):
        capture_production_differentiable_region_v1(
            source_state_hash=_HASH,
            primal_graph_hash=_HASH,
            split_state_hash=_HASH,
            topology_hash=_HASH,
            anchor=B4B_SEMANTIC_ANCHOR_V1,
            values={"incoming_lower_a": tensor},
            gradients={},
            operator_attributes={},
        )
