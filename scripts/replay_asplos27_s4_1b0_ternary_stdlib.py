#!/usr/bin/env python3
"""Replay S4-1B0 formal evidence using only the Python standard library."""

# pylint: disable=missing-function-docstring,too-many-branches,too-many-locals
# pylint: disable=too-many-boolean-expressions,duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

ARTIFACT_SCHEMA = "boundflow.asplos27-s4-1b0-ternary-summary/v1"
MANIFEST_SCHEMA = "boundflow.asplos27-s4-1b0-ternary-manifest/v1"
EXPECTED_SEQUENCE = (
    *(f"positive-{ordinal:02d}" for ordinal in range(5)),
    "cache-00",
    "fault-classifier-policy",
    "fault-cache-source",
    "fault-descriptor-dlpack",
    "fault-stream-launch",
    "fault-invalid-selector-claim",
)
EXPECTED_REASONS = (
    "TERNARY_ENDPOINT_MIDPOINT_POLICY_MISMATCH",
    "TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH",
    "TERNARY_ENDPOINT_DLPACK_IDENTITY_MISMATCH",
    "TERNARY_ENDPOINT_STREAM_IDENTITY_MISMATCH",
    "TERNARY_ENDPOINT_INVALID_SELECTOR_NOT_POISONED",
)


class ReplayError(Exception):
    """Deterministic semantic replay failure."""


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayError(f"object required: {path}")
    return value


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _selected_bits(selector: int, lower: float, upper: float) -> int:
    if selector == 1:
        value = lower
    elif selector == -1:
        value = upper
    elif selector == 0:
        value = _f32(_f32(lower + upper) * 0.5)
    else:
        return 0x7FC00000
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _validate_manifest(root: Path) -> None:
    manifest = _json(root / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ReplayError("manifest schema differs")
    expected_hash = manifest.pop("manifest_hash", None)
    if expected_hash != canonical_hash(manifest):
        raise ReplayError("manifest hash differs")
    files = manifest["files"]
    actual = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if files != actual:
        raise ReplayError("manifest inventory or file hash differs")


def _validate_binary(path: Path, row: dict[str, Any]) -> None:
    payload = path.read_bytes()
    binary = row["binary"]
    if len(payload) != 313344 or binary["byte_count"] != len(payload):
        raise ReplayError("positive sidecar bytes differ")
    if hashlib.sha256(payload).hexdigest() != binary["sha256"]:
        raise ReplayError("positive sidecar hash differs")
    records: dict[str, bytes] = {}
    cursor = 0
    for record in binary["index"]:
        if record["offset"] != cursor:
            raise ReplayError("binary index offset differs")
        chunk = payload[cursor : cursor + record["byte_count"]]
        if hashlib.sha256(chunk).hexdigest() != record["sha256"]:
            raise ReplayError("binary record hash differs")
        records[record["name"]] = chunk
        cursor += record["byte_count"]
    if cursor != len(payload):
        raise ReplayError("binary index extent differs")
    coefficients = struct.iter_unpack("<f", records["coefficient"])
    selectors = struct.unpack("<18432b", records["selector"])
    lowers = struct.iter_unpack("<f", records["lower"])
    uppers = struct.iter_unpack("<f", records["upper"])
    selected_bits = struct.iter_unpack("<I", records["selected"])
    counts = {"positive": 0, "negative": 0, "zero": 0, "invalid": 0}
    for coefficient_row, selector, lower_row, upper_row, selected_row in zip(
        coefficients, selectors, lowers, uppers, selected_bits
    ):
        coefficient = coefficient_row[0]
        bits = struct.unpack("<I", struct.pack("<f", coefficient))[0]
        expected_selector = (
            -128
            if bits & 0x7F800000 == 0x7F800000
            else 1 if coefficient > 0.0 else -1 if coefficient < 0.0 else 0
        )
        if selector != expected_selector:
            raise ReplayError("selector semantic recomputation differs")
        label = (
            "positive"
            if selector == 1
            else (
                "negative" if selector == -1 else "zero" if selector == 0 else "invalid"
            )
        )
        counts[label] += 1
        expected_bits = _selected_bits(selector, lower_row[0], upper_row[0])
        if selected_row[0] != expected_bits:
            raise ReplayError("selected semantic recomputation differs")
    if counts != {"positive": 8689, "negative": 9137, "zero": 606, "invalid": 0}:
        raise ReplayError("selector class counts differ")
    if row["counts"] != counts or row["old_binary_zero_misclassified"] != 606:
        raise ReplayError("worker derived counts differ")


def validate(root: Path) -> dict[str, Any]:
    _validate_manifest(root)
    protocol = _json(root / "protocol.json")
    if (
        tuple(protocol["worker_sequence"]) != EXPECTED_SEQUENCE
        or protocol["construction_model_hash"]
        != "5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a"
        or protocol["numel"] != 18432
        or protocol["timing_recorded"] is not False
        or protocol["performance_claimed"] is not False
    ):
        raise ReplayError("protocol worker sequence differs")
    for path, digest in protocol["dependencies"].items():
        if file_sha256(Path(path)) != digest:
            raise ReplayError("protocol dependency identity differs")
    protocol_hash = file_sha256(root / "protocol.json")
    rows = [
        json.loads(line)
        for line in (root / "raw/workers.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if (
        len(rows) != 11
        or tuple(row["worker_name"] for row in rows) != EXPECTED_SEQUENCE
    ):
        raise ReplayError("worker topology differs")
    if len({row["pid"] for row in rows}) != 11:
        raise ReplayError("fresh process identity differs")
    for ordinal, row in enumerate(rows):
        payload_hash = row.pop("worker_payload_hash", None)
        if payload_hash != canonical_hash(row):
            raise ReplayError("worker payload hash differs")
        row["worker_payload_hash"] = payload_hash
        if (
            row["protocol_hash"] != protocol_hash
            or row["performance_claimed"] is not False
            or row["run_ordinal"] != ordinal
        ):
            raise ReplayError("worker protocol or claim differs")
    for ordinal, row in enumerate(rows[:5]):
        _validate_binary(root / "raw/binary" / f"positive-{ordinal:02d}.bin", row)
        if (
            len(row["descriptor_hashes"]) != 5
            or any(len(value) != 64 for value in row["descriptor_hashes"])
            or row["dlpack_pointer_exact"] != 5
        ):
            raise ReplayError("descriptor evidence differs")
        receipt = row["module_receipt"]
        if canonical_hash(receipt) != row["module_receipt_hash"]:
            raise ReplayError("module receipt hash differs")
        for name, digest in row["module_files"].items():
            if file_sha256(root / "module" / name) != digest:
                raise ReplayError("module file identity differs")
        if (
            file_sha256(root / "module/unscheduled_tir.json")
            != receipt["unscheduled_tir_hash"]
            or file_sha256(root / "module/scheduled_tir.json")
            != receipt["scheduled_tir_hash"]
            or file_sha256(root / "module/device_source.cu")
            != receipt["device_source_hash"]
            or receipt["performance_claimed"] is not False
        ):
            raise ReplayError("module receipt content identity differs")
    cache = rows[5]
    if (
        cache["events"] != ["miss", "hit"]
        or (
            cache["compile_count"],
            cache["miss_count"],
            cache["hit_count"],
            cache["entry_count"],
        )
        != (1, 1, 1, 1)
        or not cache["same_module_receipt"]
        or cache["tensor_retention_count"] != 0
    ):
        raise ReplayError("cache semantics differ")
    reasons = tuple(row["result"]["reason"] for row in rows[6:])
    if reasons != EXPECTED_REASONS:
        raise ReplayError("fault reason sequence differs")
    if any(not row["result"]["context_is_none"] for row in rows[6:]):
        raise ReplayError("fault exception context differs")
    summary = _json(root / "summary.json")
    expected_summary_hash = summary.pop("summary_hash", None)
    if expected_summary_hash != canonical_hash(summary):
        raise ReplayError("summary hash differs")
    if (
        summary["schema_version"] != ARTIFACT_SCHEMA
        or summary["worker_count"] != 11
        or summary["selector_counts"]
        != {"positive": 8689, "negative": 9137, "zero": 606, "invalid": 0}
        or summary["fault_reasons"] != list(EXPECTED_REASONS)
        or summary["performance_claimed"] is not False
        or summary["timing_recorded"] is not False
        or summary["status"] != "FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0"
    ):
        raise ReplayError("summary semantics or claim differs")
    return {
        "schema": "boundflow.asplos27-s4-1b0-ternary-replay/v1",
        "status": "PASS",
        "worker_count": 11,
        "positive_count": 5,
        "fault_count": 5,
        "summary_hash": expected_summary_hash,
        "performance_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.artifact.resolve())
    except (
        ReplayError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
