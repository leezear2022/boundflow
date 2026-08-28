#!/usr/bin/env python3
"""Apply ten outer-resigned semantic attacks to an S2 formal artifact."""

# pylint: disable=protected-access

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

from scripts import run_asplos27_s2_crown_artifact as artifact_tool


def _resign(root: Path) -> None:
    artifact_tool._write_manifest(root)


def _slow_latency(rows):  # type: ignore[no-untyped-def]
    values = rows[0]["latency_ns"]["P"]
    rows[0]["latency_ns"]["P"] = [int(value) * 10 for value in values]
    rows[0]["median_latency_ns"]["P"] = __import__("statistics").median(
        rows[0]["latency_ns"]["P"]
    )


def _mutations():  # type: ignore[no-untyped-def]
    return (
        ("latency", _slow_latency),
        ("lower", lambda rows: rows[0]["semantic"]["P"]["lower"].__setitem__(0, 99.0)),
        (
            "gradient",
            lambda rows: rows[0]["semantic"]["P"]["gradient"].__setitem__(0, 99.0),
        ),
        (
            "cudnn-call-count",
            lambda rows: rows[0]["canonical_receipt"].__setitem__(
                "cudnn_conv_call_count", 4
            ),
        ),
        (
            "forward-replay-count",
            lambda rows: rows[0]["canonical_receipt"].__setitem__(
                "forward_graph_replay_count", 2
            ),
        ),
        (
            "active-beta",
            lambda rows: rows[0]["canonical_receipt"].__setitem__("active_beta", False),
        ),
        (
            "performance-claim",
            lambda rows: rows[0].__setitem__("performance_claimed", True),
        ),
        ("order", lambda rows: rows[0].__setitem__("order", "PDN")),
        (
            "plan-owner",
            lambda rows: rows[0]["canonical_receipt"].__setitem__(
                "production_plan_hash", "0" * 64
            ),
        ),
        (
            "source-identity",
            lambda rows: rows[0].__setitem__("source_capture_sha256", "0" * 64),
        ),
    )


def probe(source: Path) -> dict[str, object]:
    rows = artifact_tool.load_records(source / "raw/workers.jsonl")
    results: list[dict[str, object]] = []
    for name, mutate in _mutations():
        with tempfile.TemporaryDirectory(prefix=f"s2-tamper-{name}-") as directory:
            target = Path(directory) / "artifact"
            shutil.copytree(source, target)
            changed = copy.deepcopy(rows)
            mutate(changed)
            for row in changed:
                receipt = row.get("canonical_receipt")
                if isinstance(receipt, dict):
                    receipt.pop("receipt_hash", None)
                    receipt["receipt_hash"] = (
                        __import__("hashlib")
                        .sha256(artifact_tool.canonical(receipt).encode())
                        .hexdigest()
                    )
            (target / "raw/workers.jsonl").write_text(
                "".join(artifact_tool.canonical(row) + "\n" for row in changed),
                encoding="utf-8",
            )
            try:
                recomputed = artifact_tool.validate_records(
                    changed, artifact_tool.load_json(target / "protocol.json")
                )
            except (ValueError, TypeError, KeyError):
                recomputed = None
            if recomputed is not None:
                (target / "summary.json").write_text(
                    artifact_tool.canonical(recomputed) + "\n", encoding="utf-8"
                )
            _resign(target)
            rejected = False
            error = ""
            try:
                artifact_tool.replay(target)
            except (ValueError, TypeError, KeyError) as exception:
                rejected = True
                error = str(exception)
            results.append(
                {
                    "name": name,
                    "outer_resigned": True,
                    "rejected": rejected,
                    "error": error,
                }
            )
    report = {
        "schema_version": "boundflow.asplos27-s2-crown-tamper/v1",
        "case_count": len(results),
        "rejected_count": sum(int(row["rejected"]) for row in results),
        "rows": results,
        "performance_claimed": False,
    }
    if report["case_count"] != 10 or report["rejected_count"] != 10:
        raise RuntimeError("S2 tamper probe did not reject every case")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe(args.artifact.resolve())
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
