"""Synthetic contracts for the MR5 three-site production bridge."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest

from boundflow.runtime.mr5_multi_conv_production_bridge import (
    MR5MultiConvBridgeReceiptV1,
    MR5MultiConvProductionBridgeV1,
    MR5_SITE_ORDER,
    mr5_frozen_signatures,
)


def _pairs(value: int) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((site, value) for site in MR5_SITE_ORDER))


def _receipt() -> MR5MultiConvBridgeReceiptV1:
    return MR5MultiConvBridgeReceiptV1(
        evaluation_count=10,
        site_order_count=30,
        forward_launches=_pairs(10),
        backward_launches=_pairs(9),
        beta_tensor_count=_pairs(10),
        beta_numel=_pairs(0),
        handoff_content_count=_pairs(10),
        handoff_pointer_count=_pairs(0),
        cache_miss_count=_pairs(1),
        cache_hit_count=_pairs(9),
        signature_hashes=tuple((site, site.lower() * 32) for site in MR5_SITE_ORDER),
        module_receipts=tuple((site, (("site_id", site),)) for site in MR5_SITE_ORDER),
        pending_site_count=0,
        fallback_count=0,
        eager_count=0,
        native_shadow_count=0,
        timing_recorded=False,
        performance_claimed=False,
    )


def test_mr5_frozen_signatures_are_three_distinct_shape_stride_instances() -> None:
    signatures = mr5_frozen_signatures("sm_89")
    assert tuple(sorted(signatures)) == ("C0", "C1", "C2")
    assert len({signature.stable_hash() for signature in signatures.values()}) == 3
    assert signatures["C0"].result_shape == (6, 1, 3, 32, 32)
    assert signatures["C1"].result_shape == (6, 1, 8, 16, 16)
    assert signatures["C2"].result_shape == (6, 1, 16, 8, 8)
    assert signatures["C0"].stride == signatures["C1"].stride == (2, 2)
    assert signatures["C2"].stride == (1, 1)


def test_mr5_bridge_receipt_accepts_only_closed_30_27_lifecycle() -> None:
    receipt = _receipt()
    receipt.validate()
    assert receipt.to_dict()["performance_claimed"] is False


def _prewarmed_bridge() -> MR5MultiConvProductionBridgeV1:
    bridge = object.__new__(MR5MultiConvProductionBridgeV1)
    bridge.evaluation_count = 10
    bridge.site_order_count = 30
    bridge.forward = {site: 10 for site in MR5_SITE_ORDER}
    bridge.backward = {site: 9 for site in MR5_SITE_ORDER}
    bridge.beta_count = {site: 10 for site in MR5_SITE_ORDER}
    bridge.beta_numel = {site: 0 for site in MR5_SITE_ORDER}
    bridge.handoff_content = {site: 10 for site in MR5_SITE_ORDER}
    bridge.handoff_pointer = {site: 0 for site in MR5_SITE_ORDER}
    bridge.cache_miss = {site: 0 for site in MR5_SITE_ORDER}
    bridge.cache_hit = {site: 10 for site in MR5_SITE_ORDER}
    bridge.signatures = mr5_frozen_signatures("sm_89")
    bridge.module_receipts = {site: {"site_id": site} for site in MR5_SITE_ORDER}
    bridge.pending = {}
    bridge.fallback_count = 0
    bridge.eager_count = 0
    bridge.native_shadow_count = 0
    return bridge


def test_mr5_prewarmed_timing_receipt_requires_all_cache_hits() -> None:
    bridge = _prewarmed_bridge()
    receipt = bridge.timing_receipt()
    assert receipt["cache_miss_count"] == {"C2": 0, "C1": 0, "C0": 0}
    assert receipt["cache_hit_count"] == {"C2": 10, "C1": 10, "C0": 10}
    assert receipt["prewarmed_before_outer"] is True
    bridge.cache_miss["C1"] = 1
    with pytest.raises(ValueError, match="prewarmed timing receipt differs"):
        bridge.timing_receipt()


@pytest.mark.parametrize(
    "mutated",
    (
        replace(_receipt(), backward_launches=_pairs(8)),
        replace(_receipt(), pending_site_count=1),
        replace(_receipt(), performance_claimed=True),
        replace(
            _receipt(),
            module_receipts=(("C0", (("site_id", "C0"),)),) * 3,
        ),
        replace(_receipt(), handoff_pointer_count=_pairs(11)),
    ),
)
def test_mr5_bridge_receipt_rejects_lifecycle_or_claim_tamper(
    mutated: MR5MultiConvBridgeReceiptV1,
) -> None:
    with pytest.raises(ValueError, match="receipt differs"):
        mutated.validate()
