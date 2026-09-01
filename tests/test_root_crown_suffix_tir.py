"""Unit contracts for the cumulative root CROWN suffix owner."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest

from boundflow.backends.tvm.root_crown_residual import (
    RootCrownResidualTemplateV1,
)
from boundflow.backends.tvm.root_crown_terminal_linear import (
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_suffix_tir import (
    validate_root_crown_suffix_templates_v1,
)


def _terminal() -> RootCrownTerminalLinearTemplateV1:
    return RootCrownTerminalLinearTemplateV1(
        spec_count=3,
        domain_count=1,
        current_features=100,
        previous_features=1024,
        alpha_feature_indices=(0, 4, 9),
        compute_capability="sm_89",
    )


def _residual() -> RootCrownResidualTemplateV1:
    return RootCrownResidualTemplateV1(
        spec_count=3,
        domain_count=1,
        channels=16,
        height=8,
        width=8,
        entry_alpha_coordinates=((0, 0, 0),),
        inner_alpha_coordinates=((15, 7, 7),),
        compute_capability="sm_89",
    )


def test_root_crown_suffix_accepts_exact_zero_copy_boundary() -> None:
    validate_root_crown_suffix_templates_v1(_terminal(), _residual())


@pytest.mark.parametrize(
    "terminal,residual",
    (
        (replace(_terminal(), spec_count=2), _residual()),
        (_terminal(), replace(_residual(), domain_count=2)),
        (replace(_terminal(), previous_features=1000), _residual()),
        (
            _terminal(),
            replace(_residual(), compute_capability="sm_80"),
        ),
    ),
)
def test_root_crown_suffix_rejects_mismatched_boundary(
    terminal: RootCrownTerminalLinearTemplateV1,
    residual: RootCrownResidualTemplateV1,
) -> None:
    with pytest.raises(ValueError, match="suffix template boundary differs"):
        validate_root_crown_suffix_templates_v1(terminal, residual)
