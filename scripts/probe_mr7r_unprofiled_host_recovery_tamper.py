#!/usr/bin/env python3
"""Run fully re-signed attacks against the MR7-R artifact."""

# pylint: disable=missing-function-docstring,protected-access,wrong-import-position

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash  # noqa: E402
from scripts import run_mr7r_unprofiled_host_recovery_formal as formal  # noqa: E402

REPORT_SCHEMA = "boundflow.mr7r-unprofiled-host-recovery-tamper/v1"


def _worker(raw: dict[str, Any], ordinal: int) -> dict[str, Any]:
    return raw["runs"][ordinal]["worker"]


def _base(raw: dict[str, Any], ordinal: int) -> dict[str, Any]:
    return _worker(raw, ordinal)["base_worker"]


def _resign_base(base: dict[str, Any]) -> None:
    candidate = base.get("candidate_module_receipt")
    if isinstance(candidate, dict):
        candidate.pop("receipt_hash", None)
        candidate["receipt_hash"] = canonical_hash(candidate)
    base.pop("worker_hash", None)
    base["worker_hash"] = canonical_hash(base)


def _resign(raw: dict[str, Any]) -> None:
    for wrapper in raw.get("runs", []):
        worker = wrapper.get("worker")
        if not isinstance(worker, dict):
            continue
        for name in ("guard_receipt", "host_receipt"):
            receipt = worker.get(name)
            if isinstance(receipt, dict):
                receipt.pop("receipt_hash", None)
                receipt["receipt_hash"] = canonical_hash(receipt)
        events = worker.get("device_events")
        if isinstance(events, list):
            worker["device_event_hash"] = canonical_hash(events)
        base = worker.get("base_worker")
        if isinstance(base, dict):
            _resign_base(base)
        worker.pop("worker_hash", None)
        worker["worker_hash"] = canonical_hash(worker)
    raw.pop("raw_hash", None)
    raw["raw_hash"] = canonical_hash(raw)


def _host_ratio(raw: dict[str, Any]) -> None:
    worker = _worker(raw, 1)
    base = worker["base_worker"]
    host = worker["host_receipt"]
    delta = 10_000_000
    base["measurement"]["host_ns"] += delta
    host["outer_host_ns"] += delta
    host["category_ns"]["optimizer_and_residual"] += delta


def _boundary(raw: dict[str, Any]) -> None:
    host = _worker(raw, 1)["host_receipt"]
    host["category_ns"]["ffi_dlpack_stream"] += 1_000_000
    host["category_ns"]["optimizer_and_residual"] -= 1_000_000


def _semantic(raw: dict[str, Any]) -> None:
    values = _base(raw, 1)["outer_result_state"][0]["values"]
    values[0] = float(values[0]) + 1.0


def _launch(raw: dict[str, Any]) -> None:
    _worker(raw, 1)["launch_marker_counts"]["forward.C0"] = 9


def _fallback(raw: dict[str, Any]) -> None:
    _base(raw, 1)["bridge_receipt"]["fallback_count"] = 1


def _module(raw: dict[str, Any]) -> None:
    base = _base(raw, 1)
    for owner in (base["candidate_module_receipt"], base["bridge_receipt"]):
        owner["module_receipts"]["C0"]["scheduled_tir_hash"] = "6" * 64


def _guard(raw: dict[str, Any]) -> None:
    _worker(raw, 0)["guard_receipt"]["synchronizing_guards_executed"] = 0


def _stream(raw: dict[str, Any]) -> None:
    base = _base(raw, 1)
    base["stream_after"] = 1
    base["measurement"]["stream_after"] = 1


def _order(raw: dict[str, Any]) -> None:
    raw["runs"][0], raw["runs"][1] = raw["runs"][1], raw["runs"][0]


def _delete(raw: dict[str, Any]) -> None:
    raw["runs"].pop()


def _performance(raw: dict[str, Any]) -> None:
    _worker(raw, 1)["performance_claimed"] = True


def _source(raw: dict[str, Any]) -> None:
    raw["source_commit"] = "0" * 40


ATTACKS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("host-ratio-full-resign", _host_ratio),
    ("boundary-category-full-resign", _boundary),
    ("semantic-full-resign", _semantic),
    ("launch-full-resign", _launch),
    ("fallback-full-resign", _fallback),
    ("module-full-resign", _module),
    ("guard-full-resign", _guard),
    ("stream-full-resign", _stream),
    ("run-order-full-resign", _order),
    ("delete-run-full-resign", _delete),
    ("performance-claim-full-resign", _performance),
    ("source-full-resign", _source),
)


def probe(artifact: Path) -> dict[str, object]:
    formal.replay_artifact(artifact)
    original = formal._load_json(artifact / "raw.json")
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-mr7r-tamper-") as root:
        for ordinal, (name, attack) in enumerate(ATTACKS):
            copied = Path(root) / f"attack-{ordinal:02d}"
            shutil.copytree(artifact, copied)
            raw = deepcopy(original)
            attack(raw)
            _resign(raw)
            formal._write_json(copied / "raw.json", raw)
            formal.refresh_manifest(copied)
            rejected = False
            error = ""
            try:
                formal.replay_artifact(copied)
            except (ValueError, TypeError, KeyError, StopIteration) as exception:
                rejected = True
                error = type(exception).__name__
            if not rejected:
                raise ValueError(f"MR7-R tamper accepted: {name}")
            results.append(
                {"ordinal": ordinal, "attack": name, "rejected": True, "error": error}
            )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "attack_count": len(results),
        "rejected_count": len(results),
        "all_rejected": True,
        "attacks": results,
        "performance_claimed": False,
    }
    report["report_hash"] = canonical_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = probe(args.artifact)
    if args.write_report:
        formal._write_json(args.artifact / "tamper_report.json", report)
        formal.refresh_manifest(args.artifact)
        formal.replay_artifact(args.artifact)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
