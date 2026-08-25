#!/usr/bin/env python3
"""Run fully re-signed negative probes against an MR4 census artifact."""

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

from boundflow.runtime.mr4_production_conv_site_census import (  # noqa: E402
    canonical_hash,
)
from scripts import run_mr4_production_conv_site_census_formal as formal  # noqa: E402

REPORT_SCHEMA = "boundflow.mr4-production-conv-site-census-tamper/v1"


def _resign(raw: dict[str, Any]) -> None:
    for wrapper in raw.get("runs", []):
        worker = wrapper.get("worker")
        if isinstance(worker, dict):
            worker.pop("worker_hash", None)
            worker["worker_hash"] = canonical_hash(worker)
    raw.pop("raw_hash", None)
    raw["raw_hash"] = canonical_hash(raw)


def _source(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["source"]["abcrown_commit"] = "0" * 40


def _protocol(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["protocol"]["candidate_executed"] = True


def _run_index(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["run_index"] = 1


def _solver(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["solver_result"]["visited_domains"] = [5]


def _topology(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["topology"].pop()


def _delete_row(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"].pop()


def _row_order(raw: dict[str, Any]) -> None:
    rows = raw["runs"][0]["worker"]["census"]["rows"]
    rows[0], rows[1] = rows[1], rows[0]


def _evaluation(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["evaluation_ordinal"] = 1


def _grad(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["grad_enabled"] = False


def _beta(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["beta_numel"] = 1


def _handoff(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0][
        "relu_conv_handoff_content_exact"
    ] = False


def _shape(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["conv_weight"]["shape"] = [1]


def _mac(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["forward_mac_units"] = 1


def _materialization(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0][
        "candidate_minimum_materialization_bytes"
    ] = 1


def _timing(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["timing_recorded"] = True


def _semantic(raw: dict[str, Any]) -> None:
    values = raw["runs"][1]["worker"]["outer_result_state"][0]["values"]
    values[0] = float(values[0]) + 1.0


ATTACKS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("source-full-resign", _source),
    ("protocol-full-resign", _protocol),
    ("run-index-full-resign", _run_index),
    ("solver-full-resign", _solver),
    ("topology-full-resign", _topology),
    ("delete-row-full-resign", _delete_row),
    ("row-order-full-resign", _row_order),
    ("evaluation-full-resign", _evaluation),
    ("grad-mode-full-resign", _grad),
    ("beta-full-resign", _beta),
    ("handoff-full-resign", _handoff),
    ("shape-full-resign", _shape),
    ("mac-full-resign", _mac),
    ("materialization-full-resign", _materialization),
    ("timing-claim-full-resign", _timing),
    ("semantic-full-resign", _semantic),
)


def probe(artifact: Path) -> dict[str, object]:
    formal.replay_artifact(artifact)
    original_raw = formal._load_json(artifact / "raw.json")
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-mr4-census-tamper-") as root:
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
                raise ValueError(f"MR4 census tamper was accepted: {name}")
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
