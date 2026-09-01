#!/usr/bin/env python3
"""Apply fully outer-re-signed tamper probes to one CIBC R1-A artifact."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.cibc_r1_attribution import canonical_hash, canonical_json
from scripts import run_cibc_r1_attribution_artifact as artifact

Mutator = Callable[[Path], None]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("R1-A tamper JSON root differs")
    return value


def _resign_payload(value: dict[str, Any], hash_field: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != hash_field}
    value[hash_field] = canonical_hash(unsigned)


def _resign_artifact(root: Path) -> None:
    protocol = _load(root / "protocol.json")
    summary = _load(root / "summary.json")
    _resign_payload(protocol, "protocol_hash")
    _resign_payload(summary, "summary_hash")
    artifact._write_json(root / "protocol.json", protocol)
    artifact._write_json(root / "summary.json", summary)
    artifact._write_json(
        root / "manifest.json", artifact._manifest(root, protocol, summary)
    )


def _mutate_protocol(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = root / "protocol.json"
    value = _load(path)
    mutate(value)
    _resign_payload(value, "protocol_hash")
    artifact._write_json(path, value)


def _mutate_profile(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = root / "raw/pair_00_profile.json"
    value = _load(path)
    mutate(value)
    _resign_payload(value, "worker_hash")
    artifact._write_json(path, value)


def _target(root: Path) -> None:
    _mutate_protocol(
        root, lambda value: value["target_contract"].__setitem__("query_research", 1.10)
    )


def _source_digest(root: Path) -> None:
    _mutate_protocol(root, lambda value: value.__setitem__("model_sha256", "0" * 64))


def _pair_order(root: Path) -> None:
    _mutate_protocol(root, lambda value: value.__setitem__("pair_orders", ["PC"]))


def _calibration_slope(root: Path) -> None:
    _mutate_profile(
        root, lambda value: value["calibration_receipt"].__setitem__("slope", 1.5)
    )


def _calibration_raw(root: Path) -> None:
    _mutate_profile(root, lambda value: value["calibration_receipt"]["triplets"].pop())


def _semantic(root: Path) -> None:
    _mutate_profile(
        root,
        lambda value: value["semantic_receipt"].__setitem__(
            "maximum_absolute_difference", 0.0
        ),
    )


def _timing(root: Path) -> None:
    _mutate_profile(root, lambda value: value.__setitem__("median_ms", 0.01))


def _topology(root: Path) -> None:
    def mutate(value: dict[str, Any]) -> None:
        topology = value["topology"]
        topology["nodes"][0]["output_shapes"] = [[6, 8, 15, 15]]
        identity = {
            key: item
            for key, item in topology.items()
            if key not in {"topology_hash", "markers"}
        }
        topology_hash = canonical_hash(identity)
        topology["topology_hash"] = topology_hash
        markers = [
            f"boundflow.r1/graph/{node['ordinal']}/{node['op_type']}/{topology_hash[:12]}"
            for node in topology["nodes"]
        ]
        topology["markers"] = markers
        value["marker_receipt"]["markers"] = markers

    _mutate_profile(root, mutate)


def _summary_verdict(root: Path) -> None:
    path = root / "summary.json"
    value = _load(path)
    value["formal_attribution_admitted"] = True
    value["status"] = "validated-r1a-attribution"
    _resign_payload(value, "summary_hash")
    artifact._write_json(path, value)


PROBES: tuple[tuple[str, Mutator], ...] = (
    ("scope_target", _target),
    ("source_digest", _source_digest),
    ("pair_order", _pair_order),
    ("calibration_slope", _calibration_slope),
    ("calibration_raw", _calibration_raw),
    ("semantic_receipt", _semantic),
    ("timing_median", _timing),
    ("production_topology", _topology),
    ("summary_verdict", _summary_verdict),
)


def probe(source: Path) -> dict[str, object]:
    artifact.replay(source)
    rows = []
    temporary = Path(tempfile.mkdtemp(prefix="cibc-r1a-tamper-"))
    try:
        for name, mutator in PROBES:
            target = temporary / name
            shutil.copytree(source, target)
            mutator(target)
            _resign_artifact(target)
            try:
                artifact.replay(target)
            except (ValueError, TypeError, KeyError) as error:
                rows.append({"name": name, "rejected": True, "error": str(error)})
            else:
                rows.append({"name": name, "rejected": False, "error": None})
    finally:
        shutil.rmtree(temporary)
    result: dict[str, object] = {
        "schema_version": "boundflow.cibc-r1-attribution-tamper/v1",
        "probe_count": len(rows),
        "rejected_count": sum(bool(row["rejected"]) for row in rows),
        "rows": rows,
    }
    result["tamper_hash"] = canonical_hash(result)
    if result["rejected_count"] != result["probe_count"]:
        raise ValueError("R1-A tamper probe admission differs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(probe(args.artifact)))


if __name__ == "__main__":
    main()
