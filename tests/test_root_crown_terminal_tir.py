"""Unit tests for the root CROWN terminal TIR ABI."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import replace

import pytest

from boundflow.backends.tvm.root_crown_terminal_linear import (
    RootCrownTerminalLinearTemplateV1,
)


def _template() -> RootCrownTerminalLinearTemplateV1:
    return RootCrownTerminalLinearTemplateV1(
        spec_count=3,
        domain_count=1,
        current_features=100,
        previous_features=1024,
        alpha_feature_indices=tuple(range(0, 81, 3)),
        compute_capability="sm_89",
    )


def test_root_crown_terminal_template_is_deterministic() -> None:
    template = _template()
    template.validate()
    assert template.stable_hash() == template.stable_hash()
    assert template.alpha_feature_count == 27


@pytest.mark.parametrize(
    "changed",
    (
        {"spec_count": 0},
        {"domain_count": 0},
        {"current_features": 0},
        {"previous_features": 0},
        {"alpha_feature_indices": ()},
        {"alpha_feature_indices": (1, 1)},
        {"alpha_feature_indices": (2, 1)},
        {"alpha_feature_indices": (100,)},
        {"compute_capability": "cuda"},
        {"thread_extent": 48},
        {"target": "llvm"},
    ),
)
def test_root_crown_terminal_template_rejects_invalid_abi(
    changed: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="template differs"):
        replace(_template(), **changed).validate()  # type: ignore[arg-type]
