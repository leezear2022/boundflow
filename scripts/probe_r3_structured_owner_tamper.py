#!/usr/bin/env python3
"""Fully re-sign R3-0 artifact mutations and require semantic replay rejection."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, cast

import scripts.run_r3_structured_owner_artifact as artifact

Mutation = Callable[[dict[str, object], dict[str, object]], None]


def _hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _nested(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload[name]
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not an object")
    return value


def _list(payload: dict[str, object], name: str) -> list[object]:
    value = payload[name]
    if not isinstance(value, list):
        raise ValueError(f"{name} is not a list")
    return value


def _resign(root: Path, bundle: dict[str, object], summary: dict[str, object]) -> None:
    template = _nested(bundle, "template")
    instance = _nested(bundle, "instance")
    receipt = _nested(bundle, "receipt")
    template_hash = _hash(template)
    instance["template_hash"] = template_hash
    receipt["template_hash"] = template_hash
    receipt["instance_hash"] = _hash(instance)
    unsigned_bundle = {name: bundle[name] for name in bundle if name != "bundle_hash"}
    bundle["bundle_hash"] = _hash(unsigned_bundle)
    summary["bundle_hash"] = bundle["bundle_hash"]
    summary["template_hash"] = receipt["template_hash"]
    summary["instance_hash"] = receipt["instance_hash"]
    unsigned_summary = {
        name: summary[name] for name in summary if name != "summary_hash"
    }
    summary["summary_hash"] = _hash(unsigned_summary)
    artifact._write(root / "bundle.json", bundle)  # pylint: disable=protected-access
    artifact._write(root / "summary.json", summary)  # pylint: disable=protected-access
    protocol = artifact._load(
        root / "protocol.json"
    )  # pylint: disable=protected-access
    artifact._write(  # pylint: disable=protected-access
        root / "manifest.json",
        artifact._manifest(root, protocol, summary),  # pylint: disable=protected-access
    )


def _start_node(bundle: dict[str, object], _: dict[str, object]) -> None:
    _nested(bundle, "template")["start_node_id"] = "31/Gemm_14"


def _topology(bundle: dict[str, object], _: dict[str, object]) -> None:
    nodes = _list(_nested(bundle, "template"), "nodes")
    cast(dict[str, object], nodes[2])["input_ids"] = ["seed"]


def _beta_shape(bundle: dict[str, object], _: dict[str, object]) -> None:
    bindings = _list(_nested(bundle, "instance"), "bindings")
    cast(dict[str, object], bindings[1])["shape"] = [6, 1]


def _split_history(bundle: dict[str, object], _: dict[str, object]) -> None:
    _nested(bundle, "instance")["split_history_hash"] = "0" * 64


def _consumer(bundle: dict[str, object], _: dict[str, object]) -> None:
    nodes = _list(_nested(bundle, "template"), "nodes")
    cast(dict[str, object], nodes[2])["declared_consumer_count"] = 2


def _bias_fraction(bundle: dict[str, object], _: dict[str, object]) -> None:
    witnesses = _list(_nested(bundle, "template"), "bias_witnesses")
    cast(dict[str, object], witnesses[0])["numerators"] = [1, 2]


def _scratch_overlap(bundle: dict[str, object], _: dict[str, object]) -> None:
    intervals = _list(_nested(bundle, "template"), "scratch_intervals")
    cast(dict[str, object], intervals[2])["first_ordinal"] = 2


def _receipt_field(name: str, value: object) -> Mutation:
    def mutate(bundle: dict[str, object], _: dict[str, object]) -> None:
        _nested(bundle, "receipt")[name] = value

    return mutate


def _summary_gate(_: dict[str, object], summary: dict[str, object]) -> None:
    summary["r3_1_open"] = False


MUTATIONS: tuple[tuple[str, Mutation], ...] = (
    ("start_node", _start_node),
    ("topology", _topology),
    ("beta_shape", _beta_shape),
    ("split_history", _split_history),
    ("consumer_count", _consumer),
    ("bias_fraction", _bias_fraction),
    ("scratch_overlap", _scratch_overlap),
    ("dense_escape", _receipt_field("dense_escape_count", 1)),
    ("context_tensor", _receipt_field("context_tensor_count", 1)),
    ("production_connected", _receipt_field("production_connected", True)),
    ("performance_claimed", _receipt_field("performance_claimed", True)),
    ("summary_gate", _summary_gate),
)


def run(root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for name, mutation in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="r3-r30-tamper-") as directory:
            candidate = Path(directory) / "artifact"
            shutil.copytree(root, candidate)
            bundle = artifact._load(
                candidate / "bundle.json"
            )  # pylint: disable=protected-access
            summary = artifact._load(
                candidate / "summary.json"
            )  # pylint: disable=protected-access
            mutation(bundle, summary)
            _resign(candidate, bundle, summary)
            try:
                artifact.replay_artifact(candidate)
            except (ValueError, subprocess.CalledProcessError) as error:
                rows.append({"name": name, "rejected": True, "error": str(error)})
            else:
                rows.append({"name": name, "rejected": False, "error": ""})
    report: dict[str, object] = {
        "probe_count": len(rows),
        "rejected_count": sum(row["rejected"] is True for row in rows),
        "rows": rows,
    }
    report["tamper_hash"] = _hash(report)
    if report["rejected_count"] != len(rows):
        raise ValueError("R3-0 tamper probe admitted a mutation")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.artifact), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
