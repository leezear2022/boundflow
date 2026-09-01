#!/usr/bin/env python3
"""Probe ten outer-resigned attacks against BAB4 five-fresh replay."""

# pylint: disable=protected-access,too-many-locals

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, cast

from scripts import run_bab4_same_solver_five_fresh as bab4

Mutation = Callable[[dict[str, Any]], None]


def _candidate(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "raw/pair-00/BAB4/worker.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("BAB4 tamper worker root differs")
    return path, cast(dict[str, Any], payload)


def _receipt(payload: dict[str, Any]) -> dict[str, Any]:
    receipts = payload["s4_exact_call_receipts"]
    if not isinstance(receipts, list) or len(receipts) != 1:
        raise ValueError("BAB4 tamper receipt cardinality differs")
    receipt = receipts[0]
    if not isinstance(receipt, dict):
        raise TypeError("BAB4 tamper receipt differs")
    return cast(dict[str, Any], receipt)


def _resign_receipt(receipt: dict[str, Any]) -> None:
    excluded = {
        "receipt_hash",
        "static_prepare_ns",
        "static_prepare_excluded_from_query",
        "source_capture_runtime_dependency",
        "plan_template_relative_path",
        "static_warmup_receipt_hash",
        "four_segment_static_warmup",
    }
    core = {key: value for key, value in receipt.items() if key not in excluded}
    receipt["receipt_hash"] = bab4.implementation._canonical_hash(core)


def _mutations() -> tuple[tuple[str, Mutation], ...]:
    def core_latency(payload: dict[str, Any]) -> None:
        payload["run"]["metrics"]["core_wall_ns"] //= 2

    def query_latency(payload: dict[str, Any]) -> None:
        payload["run"]["metrics"]["query_wall_ns"] //= 2

    def lower(payload: dict[str, Any]) -> None:
        payload["run"]["semantics"]["lower_values"][0] += 1.0

    def discrete(payload: dict[str, Any]) -> None:
        payload["run"]["semantics"]["visited_domains"].append(999)

    def environment(payload: dict[str, Any]) -> None:
        payload["run"]["environment"]["admitted"] = False

    def assets(payload: dict[str, Any]) -> None:
        receipt = _receipt(payload)
        receipt["assets_hash"] = "0" * 64
        _resign_receipt(receipt)

    def launches(payload: dict[str, Any]) -> None:
        receipt = _receipt(payload)
        receipt["compiled_forward_launch_count"] = 75
        _resign_receipt(receipt)

    def fallback(payload: dict[str, Any]) -> None:
        receipt = _receipt(payload)
        receipt["fallback_count"] = 1
        _resign_receipt(receipt)

    def warmup_dependency(payload: dict[str, Any]) -> None:
        receipt = _receipt(payload)
        receipt["four_segment_static_warmup"][
            "source_capture_runtime_dependency"
        ] = True

    def claim(payload: dict[str, Any]) -> None:
        payload["performance_claimed"] = True

    return (
        ("core-latency", core_latency),
        ("query-latency", query_latency),
        ("lower", lower),
        ("discrete-semantics", discrete),
        ("environment-admission", environment),
        ("compiled-assets", assets),
        ("forward-launch-count", launches),
        ("fallback-count", fallback),
        ("warmup-source-dependency", warmup_dependency),
        ("performance-claim", claim),
    )


def run(artifact: Path) -> dict[str, object]:
    """Return a stable rejection ledger for ten outer-resigned mutations."""

    bab4.configure()
    rows: list[dict[str, object]] = []
    for name, mutate in _mutations():
        with tempfile.TemporaryDirectory(prefix=f"bab4-tamper-{name}-") as raw:
            root = Path(raw) / "artifact"
            shutil.copytree(artifact, root)
            worker_path, payload = _candidate(root)
            mutate(payload)
            worker_path.write_text(
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            manifest = bab4.implementation._manifest(root)
            (root / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            try:
                bab4.implementation._replay(argparse.Namespace(artifact=root))
            except (TypeError, ValueError) as error:
                rows.append(
                    {
                        "attack": name,
                        "rejected": True,
                        "reason": str(error),
                    }
                )
            else:
                raise AssertionError(f"BAB4 tamper was accepted: {name}")
    result: dict[str, object] = {
        "schema_version": "boundflow.bab4-five-fresh-tamper/v1",
        "attack_count": len(rows),
        "rejected_count": sum(bool(row["rejected"]) for row in rows),
        "attack_model": "raw-worker-mutation-plus-outer-manifest-resign",
        "coherent-full-resign_claimed": False,
        "rows": rows,
    }
    result["result_hash"] = bab4.implementation._canonical_hash(result)
    return result


def main() -> None:
    """Run the frozen attack inventory for one artifact root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(run(args.artifact.resolve()), sort_keys=True, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
