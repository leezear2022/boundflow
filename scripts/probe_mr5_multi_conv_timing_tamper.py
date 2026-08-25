#!/usr/bin/env python3
"""Run fully re-signed attacks against the MR5 multi-site timing artifact."""

# pylint: disable=protected-access,wrong-import-position,missing-function-docstring

from __future__ import annotations

from typing import Any, Callable
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash  # noqa: E402
from scripts import probe_mr3_production_bridge_timing_tamper as legacy  # noqa: E402
from scripts import run_mr5_multi_conv_timing_formal as formal  # noqa: E402

REPORT_SCHEMA = "boundflow.mr5-multi-conv-timing-tamper/v1"


def _bridge(raw: dict[str, Any], ordinal: int = 1) -> dict[str, Any]:
    return raw["runs"][ordinal]["worker"]


def _module_hash(raw: dict[str, Any]) -> None:
    receipt = _bridge(raw)["candidate_module_receipt"]
    receipt["module_receipts"]["C0"]["scheduled_tir_hash"] = "3" * 64
    unsigned = dict(receipt)
    unsigned.pop("receipt_hash", None)
    receipt["receipt_hash"] = canonical_hash(unsigned)


def _module_inner_hash(raw: dict[str, Any]) -> None:
    _bridge(raw)["candidate_module_receipt"]["receipt_hash"] = "4" * 64


def _bridge_count(raw: dict[str, Any]) -> None:
    _bridge(raw)["bridge_receipt"]["backward_launches"]["C1"] = 8


def _cache_miss(raw: dict[str, Any]) -> None:
    _bridge(raw)["bridge_receipt"]["cache_miss_count"]["C0"] = 1


def _signature(raw: dict[str, Any]) -> None:
    _bridge(raw)["bridge_receipt"]["signature_hashes"]["C2"] = "5" * 64


def _workspace(raw: dict[str, Any]) -> None:
    _bridge(raw)["bridge_receipt"]["module_receipts"]["C0"]["workspace_inventory"][0][
        "shape"
    ] = [1]


def _fresh_module_drift(raw: dict[str, Any]) -> None:
    tampered_hash = "6" * 64
    _bridge(raw, 2)["candidate_module_receipt"]["module_receipts"]["C2"][
        "device_source_hash"
    ] = tampered_hash
    receipt = _bridge(raw, 2)["candidate_module_receipt"]
    unsigned = dict(receipt)
    unsigned.pop("receipt_hash", None)
    receipt["receipt_hash"] = canonical_hash(unsigned)


ATTACKS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("host-zero-full-resign", legacy._host_zero),
    ("event-zero-full-resign", legacy._event_zero),
    ("peak-below-base-full-resign", legacy._peak_below_base),
    ("source-full-resign", legacy._source_commit),
    ("protocol-full-resign", legacy._protocol),
    ("solver-full-resign", legacy._solver),
    ("semantic-full-resign", legacy._semantic),
    ("module-stability-full-resign", _module_hash),
    ("module-inner-hash-full-resign", _module_inner_hash),
    ("bridge-count-full-resign", _bridge_count),
    ("cache-miss-full-resign", _cache_miss),
    ("signature-full-resign", _signature),
    ("workspace-full-resign", _workspace),
    ("fresh-module-drift-full-resign", _fresh_module_drift),
    ("stream-drift-full-resign", legacy._stream_drift),
    ("device-projection-full-resign", legacy._device_projection),
    ("run-order-full-resign", legacy._order),
    ("run-mode-full-resign", legacy._mode),
    ("delete-run-full-resign", legacy._delete_run),
    ("worker-performance-claim-full-resign", legacy._performance_claim),
)


def _configure() -> None:
    formal._configure_legacy_generator()
    legacy.formal = formal
    legacy.REPORT_SCHEMA = REPORT_SCHEMA
    legacy.ATTACKS = ATTACKS


def main() -> None:
    _configure()
    legacy.main()


if __name__ == "__main__":
    main()
