"""Synthetic validation for MR5 formal receipts and raw provenance."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from __future__ import annotations

import pytest

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash
from boundflow.runtime.mr5_multi_conv_formal import (
    FORMAL_SCHEMA,
    SOURCE_COMMIT,
    _validate_bridge_receipt,
    derive_summary,
)


def _module(site: str, adjoint_shape: list[int]) -> dict[str, object]:
    token = {"C0": "a", "C1": "b", "C2": "c"}[site] * 64
    return {
        "site_id": site,
        "signature_hash": token,
        "unscheduled_tir_hash": "d" * 64,
        "scheduled_tir_hash": "e" * 64,
        "device_source_hash": "f" * 64,
        "tvm_version": "0.23.dev0",
        "torch_version": "2.11.0+cu130",
        "workspace_inventory": [
            {"name": "adjoint_conv", "shape": adjoint_shape},
            {"name": "bias_delta", "shape": [6, 1]},
        ],
    }


def _bridge_receipt() -> dict[str, object]:
    sites = ("C2", "C1", "C0")
    signatures = {"C0": "a" * 64, "C1": "b" * 64, "C2": "c" * 64}
    return {
        "evaluation_count": 10,
        "site_order_count": 30,
        "forward_launches": {site: 10 for site in sites},
        "backward_launches": {site: 9 for site in sites},
        "beta_tensor_count": {site: 10 for site in sites},
        "beta_numel": {site: 0 for site in sites},
        "handoff_content_count": {site: 10 for site in sites},
        "handoff_pointer_count": {site: 0 for site in sites},
        "cache_miss_count": {site: 1 for site in sites},
        "cache_hit_count": {site: 9 for site in sites},
        "signature_hashes": signatures,
        "module_receipts": {
            "C0": _module("C0", [6, 1, 8, 16, 16]),
            "C1": _module("C1", [6, 1, 16, 8, 8]),
            "C2": _module("C2", [6, 1, 16, 8, 8]),
        },
        "pending_site_count": 0,
        "fallback_count": 0,
        "eager_count": 0,
        "native_shadow_count": 0,
        "timing_recorded": False,
        "performance_claimed": False,
    }


def test_mr5_formal_bridge_receipt_accepts_three_distinct_modules() -> None:
    _validate_bridge_receipt(_bridge_receipt())


def test_mr5_formal_bridge_receipt_rejects_signature_module_alias() -> None:
    receipt = _bridge_receipt()
    receipt["signature_hashes"]["C0"] = "9" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="module receipt differs"):
        _validate_bridge_receipt(receipt)


def test_mr5_formal_bridge_receipt_rejects_workspace_shape_tamper() -> None:
    receipt = _bridge_receipt()
    receipt["module_receipts"]["C0"]["workspace_inventory"][0]["shape"] = [1]  # type: ignore[index]
    with pytest.raises(ValueError, match="module receipt differs"):
        _validate_bridge_receipt(receipt)


def test_mr5_formal_raw_provenance_rejects_empty_resigned_payload() -> None:
    raw: dict[str, object] = {
        "schema_version": FORMAL_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "run_order": [],
        "runs": [],
        "rollback_probe": {},
        "timing_recorded": False,
        "performance_claimed": False,
    }
    raw["raw_hash"] = canonical_hash(raw)
    with pytest.raises(ValueError, match="raw provenance differs"):
        derive_summary(raw)
