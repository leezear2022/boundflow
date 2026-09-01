"""Unit contracts for root projection TIR and expanded-owner boundaries."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.backends.tvm.root_crown_projection import (
    build_root_crown_projection_modules_v1,
    RootCrownProjectionTemplateV1,
)
from boundflow.backends.tvm.root_crown_residual import RootCrownResidualTemplateV1
from boundflow.backends.tvm.root_crown_terminal_linear import (
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_expanded_suffix_tir import (
    validate_root_crown_expanded_templates_v1,
)
from boundflow.runtime.root_crown_projection_tir import (
    RootCrownProjectionTensorsV1,
    validate_root_crown_projection_tensors_v1,
)


def _template() -> RootCrownProjectionTemplateV1:
    return RootCrownProjectionTemplateV1(
        spec_count=3,
        domain_count=1,
        output_channels=16,
        output_height=8,
        output_width=8,
        input_channels=8,
        input_height=16,
        input_width=16,
        entry_alpha_coordinates=((0, 0, 0), (15, 7, 7)),
        inner_alpha_coordinates=((1, 2, 3),),
        compute_capability="sm_89",
    )


def test_root_crown_projection_template_is_deterministic() -> None:
    template = _template()
    template.validate()
    assert template.stable_hash() == template.stable_hash()
    assert template.incoming_shape == (3, 1, 16, 8, 8)
    assert template.output_shape == (3, 1, 8, 16, 16)
    assert template.bound_shape == (1, 16, 8, 8)
    assert template.outer_weight_shape == (16, 16, 3, 3)
    assert template.inner_weight_shape == (16, 8, 3, 3)
    assert template.skip_weight_shape == (16, 8, 1, 1)


@pytest.mark.parametrize(
    "changed",
    (
        {"spec_count": 0},
        {"domain_count": 0},
        {"output_channels": 0},
        {"output_height": 0},
        {"output_width": 0},
        {"input_channels": 0},
        {"input_height": 15},
        {"input_width": 15},
        {"entry_alpha_coordinates": ()},
        {"inner_alpha_coordinates": ()},
        {"entry_alpha_coordinates": ((0, 0, 0), (0, 0, 0))},
        {"entry_alpha_coordinates": ((16, 0, 0),)},
        {"inner_alpha_coordinates": ((0, 8, 0),)},
        {"compute_capability": "cuda"},
        {"thread_extent": 48},
        {"stride": (1, 1)},
        {"main_kernel": (1, 1)},
        {"main_padding": (0, 0)},
        {"skip_kernel": (3, 3)},
        {"skip_padding": (1, 1)},
        {"target": "llvm"},
    ),
)
def test_root_crown_projection_template_rejects_invalid_abi(
    changed: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="template differs"):
        replace(_template(), **changed).validate()  # type: ignore[arg-type]


def test_root_crown_projection_runtime_rejects_cpu_tensors() -> None:
    template = _template()
    incoming = torch.zeros(template.incoming_shape)
    bound = torch.zeros(template.bound_shape)
    tensors = RootCrownProjectionTensorsV1(
        incoming,
        bound,
        bound,
        torch.zeros((2, 3, 1, 2)),
        torch.zeros(template.outer_weight_shape),
        torch.zeros((template.output_channels,)),
        bound,
        bound,
        torch.zeros((2, 3, 1, 1)),
        torch.zeros(template.inner_weight_shape),
        torch.zeros((template.output_channels,)),
        torch.zeros(template.skip_weight_shape),
        torch.zeros((template.output_channels,)),
    )
    with pytest.raises(ValueError, match="runtime tensor differs"):
        validate_root_crown_projection_tensors_v1(tensors, template)


def test_root_crown_projection_forward_parallelizes_bias_reductions() -> None:
    _unscheduled, scheduled, inventory = build_root_crown_projection_modules_v1(
        _template()
    )
    script = scheduled.script(show_meta=False)
    assert "entry_bias_delta_rf" in script
    assert "inner_bias_delta_rf" in script
    assert 'T.thread_binding(128, thread="threadIdx.x")' in script
    assert ("entry_bias_delta.rf", (3, 1, 128)) in inventory
    assert ("inner_bias_delta.rf", (3, 1, 128)) in inventory


def test_expanded_owner_accepts_only_exact_zero_copy_boundaries() -> None:
    projection = _template()
    residual = RootCrownResidualTemplateV1(
        spec_count=3,
        domain_count=1,
        channels=16,
        height=8,
        width=8,
        entry_alpha_coordinates=((0, 0, 0),),
        inner_alpha_coordinates=((1, 0, 0),),
        compute_capability="sm_89",
    )
    terminal = RootCrownTerminalLinearTemplateV1(
        spec_count=3,
        domain_count=1,
        current_features=100,
        previous_features=1024,
        alpha_feature_indices=(0,),
        compute_capability="sm_89",
    )
    validate_root_crown_expanded_templates_v1(terminal, residual, projection)
    with pytest.raises(ValueError, match="boundary differs"):
        validate_root_crown_expanded_templates_v1(
            replace(terminal, previous_features=1000), residual, projection
        )
    with pytest.raises(ValueError, match="boundary differs"):
        validate_root_crown_expanded_templates_v1(
            terminal, residual, replace(projection, spec_count=4)
        )
