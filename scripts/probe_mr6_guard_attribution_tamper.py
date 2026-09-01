#!/usr/bin/env python3
"""Run fully re-signed attacks against the MR6 guard artifact."""

# pylint: disable=missing-function-docstring,protected-access,wrong-import-position
# pylint: disable=superfluous-parens

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
from scripts import run_mr6_guard_attribution_formal as formal  # noqa: E402

REPORT_SCHEMA = "boundflow.mr6-hot-path-guard-attribution-tamper/v1"


def _base(raw: dict[str, Any], ordinal: int) -> dict[str, Any]:
    return raw["runs"][ordinal]["worker"]["base_worker"]


def _resign(raw: dict[str, Any]) -> None:
    for wrapper in raw.get("runs", []):
        worker = wrapper.get("worker")
        if not isinstance(worker, dict):
            continue
        guard = worker.get("guard_receipt")
        if isinstance(guard, dict):
            guard.pop("receipt_hash", None)
            guard["receipt_hash"] = canonical_hash(guard)
        base = worker.get("base_worker")
        if isinstance(base, dict):
            candidate = base.get("candidate_module_receipt")
            if isinstance(candidate, dict):
                candidate.pop("receipt_hash", None)
                candidate["receipt_hash"] = canonical_hash(candidate)
            base.pop("worker_hash", None)
            base["worker_hash"] = canonical_hash(base)
        worker.pop("worker_hash", None)
        worker["worker_hash"] = canonical_hash(worker)
    raw.pop("raw_hash", None)
    raw["raw_hash"] = canonical_hash(raw)


def _host(raw: dict[str, Any]) -> None:
    _base(raw, 0)["measurement"]["host_ns"] = 0


def _event(raw: dict[str, Any]) -> None:
    _base(raw, 0)["measurement"]["cuda_event_ms"] = 0.0


def _semantic(raw: dict[str, Any]) -> None:
    values = _base(raw, 1)["outer_result_state"][0]["values"]
    values[0] = float(values[0]) + 1.0


def _guard(raw: dict[str, Any]) -> None:
    raw["runs"][2]["worker"]["guard_receipt"]["synchronizing_guards_executed"] = 0


def _policy(raw: dict[str, Any]) -> None:
    raw["runs"][2]["worker"]["guard_receipt"]["policy"] = "full"


def _base_mode(raw: dict[str, Any]) -> None:
    _base(raw, 1)["mode"] = "provider"


def _module(raw: dict[str, Any]) -> None:
    _base(raw, 1)["candidate_module_receipt"]["module_receipts"]["C0"][
        "scheduled_tir_hash"
    ] = ("4" * 64)


def _bridge(raw: dict[str, Any]) -> None:
    _base(raw, 1)["bridge_receipt"]["backward_launches"]["C1"] = 8


def _order(raw: dict[str, Any]) -> None:
    raw["runs"][0], raw["runs"][1] = raw["runs"][1], raw["runs"][0]


def _delete(raw: dict[str, Any]) -> None:
    raw["runs"].pop()


def _performance(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["performance_claimed"] = True


def _source(raw: dict[str, Any]) -> None:
    raw["source_commit"] = "0" * 40


ATTACKS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("host-zero-full-resign", _host),
    ("event-zero-full-resign", _event),
    ("semantic-full-resign", _semantic),
    ("guard-count-full-resign", _guard),
    ("guard-policy-full-resign", _policy),
    ("base-mode-full-resign", _base_mode),
    ("module-full-resign", _module),
    ("bridge-count-full-resign", _bridge),
    ("run-order-full-resign", _order),
    ("delete-run-full-resign", _delete),
    ("performance-claim-full-resign", _performance),
    ("source-full-resign", _source),
)


def probe(artifact: Path) -> dict[str, object]:
    formal.replay_artifact(artifact)
    original = formal._load_json(artifact / "raw.json")
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-mr6-guard-tamper-") as root:
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
                raise ValueError(f"MR6 guard tamper accepted: {name}")
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
