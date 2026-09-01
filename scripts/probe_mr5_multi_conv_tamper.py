#!/usr/bin/env python3
"""Probe fully re-signed MR5 semantic, lifecycle, compiler, and rollback attacks."""

# pylint: disable=missing-function-docstring,too-many-locals,wrong-import-position

from __future__ import annotations

import argparse
import copy
import json
import lzma
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash  # noqa: E402
from boundflow.runtime.mr5_multi_conv_formal import derive_summary  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(lzma.decompress(path.read_bytes()).decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("MR5 raw root must be an object")
    return value


def _resign(raw: dict[str, Any]) -> None:
    for wrapper in raw.get("runs", []):
        worker = wrapper["worker"]
        unsigned = dict(worker)
        unsigned.pop("worker_hash", None)
        worker["worker_hash"] = canonical_hash(unsigned)
    rollback = raw.get("rollback_probe")
    if isinstance(rollback, dict):
        unsigned = dict(rollback)
        unsigned.pop("worker_hash", None)
        rollback["worker_hash"] = canonical_hash(unsigned)
    unsigned_raw = dict(raw)
    unsigned_raw.pop("raw_hash", None)
    raw["raw_hash"] = canonical_hash(unsigned_raw)


def _bridge(raw: dict[str, Any], ordinal: int = 1) -> dict[str, Any]:
    return raw["runs"][ordinal]["worker"]


def _provider(raw: dict[str, Any]) -> dict[str, Any]:
    return raw["runs"][0]["worker"]


def _mutations() -> dict[str, Callable[[dict[str, Any]], None]]:
    def region_value(raw):
        _bridge(raw)["region_states"][0]["lower_a"]["values"][0] += 0.25

    def optimizer_gradient(raw):
        _bridge(raw)["mutation_trajectory"][0]["gradient"]["values"][0] += 0.1

    def module_hash(raw):
        tampered_hash = "a" * 64
        _bridge(raw)["bridge_receipt"]["module_receipts"]["C0"][
            "scheduled_tir_hash"
        ] = tampered_hash

    return {
        "solver_visited": lambda raw: _bridge(raw)["solver_result"].update(
            visited_domains=[7]
        ),
        "protocol_site_order": lambda raw: _bridge(raw)["protocol"].update(
            site_order=["C2", "C0", "C1"]
        ),
        "region_site": lambda raw: _bridge(raw)["region_states"][0].update(site="C1"),
        "region_value": region_value,
        "optimizer_gradient": optimizer_gradient,
        "optimizer_step": lambda raw: _bridge(raw)["mutation_trajectory"][0].update(
            optimizer_step=2.0
        ),
        "forward_count": lambda raw: _bridge(raw)["bridge_receipt"][
            "forward_launches"
        ].update(C0=9),
        "backward_count": lambda raw: _bridge(raw)["bridge_receipt"][
            "backward_launches"
        ].update(C1=8),
        "beta_numel": lambda raw: _bridge(raw)["bridge_receipt"]["beta_numel"].update(
            C2=1
        ),
        "pending_site": lambda raw: _bridge(raw)["bridge_receipt"].update(
            pending_site_count=1
        ),
        "performance_claim": lambda raw: _bridge(raw)["bridge_receipt"].update(
            performance_claimed=True
        ),
        "signature_hash": lambda raw: _bridge(raw)["bridge_receipt"][
            "signature_hashes"
        ].update(C0="b" * 64),
        "module_site": lambda raw: _bridge(raw)["bridge_receipt"]["module_receipts"][
            "C0"
        ].update(site_id="C1"),
        "module_hash": module_hash,
        "workspace_shape": lambda raw: _bridge(raw)["bridge_receipt"][
            "module_receipts"
        ]["C1"]["workspace_inventory"][0].update(shape=[1]),
        "fresh_module_drift": lambda raw: _bridge(raw, 2)["bridge_receipt"][
            "module_receipts"
        ]["C2"].update(device_source_hash="c" * 64),
        "rollback_pointer": lambda raw: raw["rollback_probe"]["atomic_receipt"].update(
            owner_pointer_hash_after="d" * 64
        ),
        "rollback_site": lambda raw: raw["rollback_probe"].update(
            injected_failure_site="C0"
        ),
        "run_order": lambda raw: raw["run_order"].reverse(),
        "source_commit": lambda raw: raw.update(source_commit="e" * 40),
        "provider_bridge_receipt": lambda raw: _provider(raw).update(bridge_receipt={}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raw = _load(args.artifact / "raw.json.xz")
    results = []
    for name, mutate in _mutations().items():
        variant = copy.deepcopy(raw)
        mutate(variant)
        _resign(variant)
        try:
            derive_summary(variant)
        except (ValueError, TypeError, KeyError, StopIteration) as error:
            results.append({"name": name, "rejected": True, "error": str(error)})
        else:
            results.append({"name": name, "rejected": False, "error": None})
    if not all(result["rejected"] for result in results):
        raise RuntimeError("MR5 tamper probe admitted a fully re-signed attack")
    report: dict[str, object] = {
        "schema_version": "boundflow.mr5-multi-conv-tamper/v1",
        "attack_count": len(results),
        "rejected_count": sum(bool(result["rejected"]) for result in results),
        "results": results,
    }
    report["report_hash"] = canonical_hash(report)
    args.report.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"rejected={report['rejected_count']}/{report['attack_count']}")
    print(f"report_hash={report['report_hash']}")


if __name__ == "__main__":
    main()
