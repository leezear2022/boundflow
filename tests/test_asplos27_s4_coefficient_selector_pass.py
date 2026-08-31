"""S4-1B ordered coefficient-selector pass tests."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from boundflow.runtime.asplos27_s4_coefficient_selector_pass import (
    PreparedS4CoefficientSelectorPassV1,
    S4_SELECTOR_ACTIONS,
    S4_SELECTOR_SPECS,
    S4SelectorPassError,
    S4SelectorPhase,
)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda", torch.cuda.current_device())


def _run_selector_pass(
    device: torch.device,
) -> PreparedS4CoefficientSelectorPassV1:
    owner = PreparedS4CoefficientSelectorPassV1(
        device=device, exact_call_id="test-exact-call"
    )
    sizes = {action: numel for _name, action, numel, _policy in S4_SELECTOR_SPECS}
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        owner.begin()
        for action in S4_SELECTOR_ACTIONS:
            coefficient = None
            if action in sizes:
                coefficient = torch.linspace(
                    -1.0, 1.0, sizes[action], dtype=torch.float32, device=device
                )
            owner.record(action, coefficient)
    stream.synchronize()
    return owner


def test_s4_selector_pass_exact_order_and_policies() -> None:
    device = _require_cuda()
    owner = _run_selector_pass(device)
    receipt = owner.receipt()
    receipt.validate()
    assert owner.phase == S4SelectorPhase.SELECTORS_READY
    assert receipt.action_order == S4_SELECTOR_ACTIONS
    assert receipt.selector_count == 6
    assert receipt.invalid_selector_count == 0
    endpoint = owner.selector("endpoint_ainput_v2")
    assert set(endpoint.cpu().tolist()) == {-1, 1}
    for name in ("sign_a18", "sign_a20", "sign_a24", "sign_a26", "sign_a29"):
        assert set(owner.selector(name).cpu().tolist()) == {0, 1}
    encoded = json.dumps(receipt.__dict__, default=lambda value: value.__dict__)
    assert "data_pointer" not in encoded and "cuda_stream" not in encoded
    mutations = (
        replace(receipt, action_count=18),
        replace(receipt, selector_count=5),
        replace(receipt, endpoint_selector_count=0),
        replace(receipt, binary_selector_count=4),
        replace(receipt, launch_count=18),
        replace(receipt, fallback_count=1),
        replace(receipt, performance_claimed=True),
        replace(receipt, timing_recorded=True),
        replace(receipt, selector_generation=receipt.coefficient_generation),
        replace(receipt, action_sequence_hash="0" * 64),
    )
    for changed in mutations:
        with pytest.raises(S4SelectorPassError):
            changed.validate()


def test_s4_selector_pass_zero_and_nonfinite_are_explicit() -> None:
    device = _require_cuda()
    owner = PreparedS4CoefficientSelectorPassV1(
        device=device, exact_call_id="nonfinite"
    )
    sizes = {action: numel for _name, action, numel, _policy in S4_SELECTOR_SPECS}
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        owner.begin()
        for action in S4_SELECTOR_ACTIONS:
            coefficient = None
            if action in sizes:
                coefficient = torch.zeros(
                    sizes[action], dtype=torch.float32, device=device
                )
                coefficient[0] = torch.nan
            owner.record(action, coefficient)
    stream.synchronize()
    assert owner.selector("endpoint_ainput_v2")[0].item() == -128
    assert owner.selector("endpoint_ainput_v2")[1].item() == 0
    assert owner.selector("sign_a18")[0].item() == -128
    assert owner.selector("sign_a18")[1].item() == 1
    assert owner.receipt().invalid_selector_count == 6


def test_s4_selector_pass_wrong_order_poisoned_without_retry() -> None:
    device = _require_cuda()
    owner = PreparedS4CoefficientSelectorPassV1(
        device=device, exact_call_id="bad-order"
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        owner.begin()
        with pytest.raises(S4SelectorPassError, match="SELECTOR_ACTION_ORDER_MISMATCH"):
            owner.record("linear16_right")
        assert owner.phase == S4SelectorPhase.POISONED
        with pytest.raises(S4SelectorPassError, match="SELECTOR_PHASE_MISMATCH"):
            owner.record("seed")


def test_s4_selector_pass_requires_pack_payload() -> None:
    device = _require_cuda()
    owner = PreparedS4CoefficientSelectorPassV1(device=device, exact_call_id="missing")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        owner.begin()
        for action in S4_SELECTOR_ACTIONS[:4]:
            owner.record(action)
        with pytest.raises(S4SelectorPassError, match="SELECTOR_COEFFICIENT_ABSENT"):
            owner.record("pack_a29")
    assert owner.phase == S4SelectorPhase.POISONED


def test_s4_selector_pass_rejects_generation_alias() -> None:
    device = _require_cuda()
    with pytest.raises(S4SelectorPassError, match="SELECTOR_GENERATION_MISMATCH"):
        PreparedS4CoefficientSelectorPassV1(
            device=device,
            exact_call_id="generation-alias",
            evaluation_generation=1,
            parameter_generation=2,
            coefficient_generation=3,
            selector_generation=3,
        )
