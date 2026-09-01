"""CPU gates for the MR6 guard attribution policy and ledger."""

# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=import-error

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.run_mr6_guard_attribution_worker import _GuardLedger, _validate_structural


class _Tensor:
    def __init__(
        self, shape: tuple[int, ...], *, grad: bool = False, contiguous: bool = True
    ) -> None:
        self.shape = shape
        self.dtype = "torch.float32"
        self.device = SimpleNamespace(type="cuda")
        self.requires_grad = grad
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous


def _signature() -> SimpleNamespace:
    return SimpleNamespace(
        validate=lambda: None,
        incoming_shape=(6, 1, 16, 8, 8),
        relaxation_shape=(6, 16, 8, 8),
        domain_count=6,
        spec_count=1,
        weight_shape=(16, 16, 3, 3),
        output_channels=16,
    )


def _tensors() -> SimpleNamespace:
    return SimpleNamespace(
        incoming=_Tensor((6, 1, 16, 8, 8), grad=True),
        lower=_Tensor((6, 16, 8, 8)),
        upper=_Tensor((6, 16, 8, 8)),
        alpha=_Tensor((6, 16, 8, 8), grad=True),
        incoming_bias=_Tensor((6, 1)),
        weight=_Tensor((16, 16, 3, 3)),
        operator_bias=_Tensor((16,)),
    )


def test_mr6_structural_validator_accepts_frozen_structure_without_values() -> None:
    _validate_structural(_signature(), _tensors())


def test_mr6_structural_validator_rejects_layout_and_gradient_drift() -> None:
    tensors = _tensors()
    tensors.weight._contiguous = False
    with pytest.raises(ValueError, match="structural tensor differs"):
        _validate_structural(_signature(), tensors)
    tensors = _tensors()
    tensors.alpha.requires_grad = False
    with pytest.raises(ValueError, match="gradient ownership"):
        _validate_structural(_signature(), tensors)


def test_mr6_full_and_diagnostic_guard_ledgers_are_exact() -> None:
    full = _GuardLedger("full")
    diagnostic = _GuardLedger("diagnostic")
    full.validation_calls = diagnostic.validation_calls = 30
    full.content_calls = diagnostic.content_calls = 30
    full_receipt = full.to_dict(site_evaluations=30)
    diagnostic_receipt = diagnostic.to_dict(site_evaluations=30)
    assert full_receipt["synchronizing_guards_executed"] == 360
    assert diagnostic_receipt["synchronizing_guards_executed"] == 60
    assert diagnostic_receipt["input_value_guards_elided"] == 270
    assert diagnostic_receipt["handoff_content_guards_elided"] == 30
    assert diagnostic_receipt["production_admitted"] is False
