"""Unit contracts for the root residual CROWN TIR and runtime ABI."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.backends.tvm.root_crown_residual import (
    build_root_crown_residual_modules_v1,
    RootCrownResidualTemplateV1,
)
from boundflow.runtime.root_crown_residual_tir import (
    RootCrownResidualTensorsV1,
    validate_root_crown_residual_tensors_v1,
)


def _template() -> RootCrownResidualTemplateV1:
    return RootCrownResidualTemplateV1(
        spec_count=3,
        domain_count=1,
        channels=16,
        height=8,
        width=8,
        entry_alpha_coordinates=((0, 0, 0), (15, 7, 7)),
        inner_alpha_coordinates=((1, 2, 3),),
        compute_capability="sm_89",
    )


def test_root_crown_residual_template_is_deterministic() -> None:
    template = _template()
    template.validate()
    assert template.stable_hash() == template.stable_hash()
    assert template.coefficient_shape == (3, 1, 16, 8, 8)
    assert template.bound_shape == (1, 16, 8, 8)
    assert template.weight_shape == (16, 16, 3, 3)
    assert template.entry_alpha_count == 2
    assert template.inner_alpha_count == 1


@pytest.mark.parametrize(
    "changed",
    (
        {"spec_count": 0},
        {"domain_count": 0},
        {"channels": 0},
        {"height": 0},
        {"width": 0},
        {"entry_alpha_coordinates": ()},
        {"inner_alpha_coordinates": ()},
        {"entry_alpha_coordinates": ((0, 0, 0), (0, 0, 0))},
        {"entry_alpha_coordinates": ((16, 0, 0),)},
        {"inner_alpha_coordinates": ((0, 8, 0),)},
        {"compute_capability": "cuda"},
        {"thread_extent": 48},
        {"kernel": (1, 1)},
        {"padding": (0, 0)},
        {"target": "llvm"},
    ),
)
def test_root_crown_residual_template_rejects_invalid_abi(
    changed: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="template differs"):
        replace(_template(), **changed).validate()  # type: ignore[arg-type]


def test_root_crown_residual_runtime_rejects_cpu_tensors() -> None:
    template = _template()
    coefficient = torch.zeros(template.coefficient_shape)
    bound = torch.zeros(template.bound_shape)
    entry_alpha = torch.zeros((2, 3, 1, 2))
    inner_alpha = torch.zeros((2, 3, 1, 1))
    weight = torch.zeros(template.weight_shape)
    bias = torch.zeros((template.channels,))
    tensors = RootCrownResidualTensorsV1(
        coefficient,
        bound,
        bound,
        entry_alpha,
        weight,
        bias,
        bound,
        bound,
        inner_alpha,
        weight,
        bias,
    )
    with pytest.raises(ValueError, match="runtime tensor differs"):
        validate_root_crown_residual_tensors_v1(tensors, template)


def test_root_crown_residual_forward_parallelizes_bias_reductions() -> None:
    _unscheduled, scheduled, inventory = build_root_crown_residual_modules_v1(
        _template()
    )
    script = scheduled.script(show_meta=False)
    assert "entry_bias_delta_rf" in script
    assert "inner_bias_delta_rf" in script
    assert 'T.thread_binding(128, thread="threadIdx.x")' in script
    assert ("entry_bias_delta.rf", (3, 1, 128)) in inventory
    assert ("inner_bias_delta.rf", (3, 1, 128)) in inventory
