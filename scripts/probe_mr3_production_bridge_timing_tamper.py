#!/usr/bin/env python3
"""Run fully re-signed negative probes against an MR3 timing artifact."""

# pylint: disable=missing-function-docstring,protected-access,too-many-statements
# pylint: disable=wrong-import-position

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

from boundflow.runtime.mr3_production_bridge_timing import (  # noqa: E402
    canonical_hash,
)
from scripts import run_mr3_production_bridge_timing_formal as formal  # noqa: E402

REPORT_SCHEMA = "boundflow.mr3-production-bridge-timing-tamper/v1"


def _resign(raw: dict[str, Any]) -> None:
    for wrapper in raw.get("runs", []):
        worker = wrapper.get("worker")
        if isinstance(worker, dict):
            worker.pop("worker_hash", None)
            worker["worker_hash"] = canonical_hash(worker)
    raw.pop("raw_hash", None)
    raw["raw_hash"] = canonical_hash(raw)


def _host_zero(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["measurement"]["host_ns"] = 0


def _event_zero(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["measurement"]["cuda_event_ms"] = 0.0


def _peak_below_base(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["measurement"]["peak_allocated_bytes"] = 0


def _source_commit(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["source"]["abcrown_commit"] = "0" * 40


def _protocol(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["protocol"]["beta_steps"] = 11


def _solver(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["solver_result"]["success"] = False


def _semantic(raw: dict[str, Any]) -> None:
    values = raw["runs"][1]["worker"]["outer_result_state"][0]["values"]
    values[0] = float(values[0]) + 1.0


def _module_stability(raw: dict[str, Any]) -> None:
    receipt = raw["runs"][1]["worker"]["candidate_module_receipt"]
    receipt["module_hash"] = "3" * 64
    receipt.pop("receipt_hash", None)
    receipt["receipt_hash"] = canonical_hash(receipt)


def _module_inner_hash(raw: dict[str, Any]) -> None:
    raw["runs"][1]["worker"]["candidate_module_receipt"]["receipt_hash"] = "4" * 64


def _bridge_count(raw: dict[str, Any]) -> None:
    raw["runs"][1]["worker"]["bridge_receipt"]["backward_launch_count"] = 8


def _stream_drift(raw: dict[str, Any]) -> None:
    worker = raw["runs"][0]["worker"]
    worker["stream_after"] = int(worker["stream_after"]) + 1
    worker["measurement"]["stream_after"] = worker["stream_after"]


def _device_projection(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["device_after"] = 1


def _order(raw: dict[str, Any]) -> None:
    raw["runs"][0], raw["runs"][1] = raw["runs"][1], raw["runs"][0]


def _mode(raw: dict[str, Any]) -> None:
    raw["runs"][0]["mode"] = "bridge"


def _delete_run(raw: dict[str, Any]) -> None:
    raw["runs"].pop()


def _performance_claim(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["performance_claimed"] = True


ATTACKS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("host-zero-full-resign", _host_zero),
    ("event-zero-full-resign", _event_zero),
    ("peak-below-base-full-resign", _peak_below_base),
    ("source-full-resign", _source_commit),
    ("protocol-full-resign", _protocol),
    ("solver-full-resign", _solver),
    ("semantic-full-resign", _semantic),
    ("module-stability-full-resign", _module_stability),
    ("module-inner-hash-full-resign", _module_inner_hash),
    ("bridge-count-full-resign", _bridge_count),
    ("stream-drift-full-resign", _stream_drift),
    ("device-projection-full-resign", _device_projection),
    ("run-order-full-resign", _order),
    ("run-mode-full-resign", _mode),
    ("delete-run-full-resign", _delete_run),
    ("worker-performance-claim-full-resign", _performance_claim),
)


def probe(artifact: Path) -> dict[str, object]:
    formal.replay_artifact(artifact)
    original_raw = formal._load_json(artifact / "raw.json")
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-mr3-timing-tamper-") as root:
        for ordinal, (name, attack) in enumerate(ATTACKS):
            copied = Path(root) / f"attack-{ordinal:02d}"
            shutil.copytree(artifact, copied)
            raw = deepcopy(original_raw)
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
                raise ValueError(f"MR3 timing tamper was accepted: {name}")
            results.append(
                {"ordinal": ordinal, "attack": name, "rejected": True, "error": error}
            )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "attack_count": len(results),
        "rejected_count": sum(bool(row["rejected"]) for row in results),
        "all_rejected": all(bool(row["rejected"]) for row in results),
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
