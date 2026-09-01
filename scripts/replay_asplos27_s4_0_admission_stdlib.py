#!/usr/bin/env python3
"""Replay the S4-0 five-fresh admission artifact using only stdlib."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,too-many-nested-blocks
# pylint: disable=unidiomatic-typecheck,missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

ARTIFACT_SCHEMA = "boundflow.asplos27-s4-admission-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.asplos27-s4-admission-protocol/v1"
MANIFEST_SCHEMA = "boundflow.asplos27-s4-admission-manifest/v1"
WORKER_SCHEMA = "boundflow.asplos27-s4-admission-worker/v1"
SUMMARY_SCHEMA = "boundflow.asplos27-s4-admission-summary/v1"
NEGATIVE_SCHEMA = "boundflow.asplos27-s4-admission-negative-registry/v1"
FORMAL_STATUS = "FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-0"
EXPECTED_COUNTS = {
    "slot_count": 6,
    "path_count": 12,
    "alpha_source_count": 6,
    "alpha_stored_element_count": 8496,
    "alpha_active_element_count": 4248,
    "alpha_preserved_element_count": 4248,
    "beta_slot_count": 6,
    "active_beta_slot_count": 1,
    "active_beta_element_count": 6,
    "live_tensor_count": 12,
    "live_element_count_per_pass": 8502,
    "live_bytes_per_pass": 34008,
    "live_content_capture_pass_count": 2,
    "device_to_host_validation_copy_count": 24,
    "device_to_host_validation_bytes": 68016,
}


def canonical(value: object) -> str:
    """Return the one canonical JSON representation used by the artifact."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    """Hash one canonical JSON value."""

    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object with an exact root type."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"S4 JSON root differs: {path.name}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(type(row) is not dict for row in rows):
        raise TypeError("S4 worker JSONL row differs")
    return rows


def _shape_product(value: object) -> int:
    if type(value) is not list or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise TypeError("S4 tensor shape differs")
    return math.prod(value)


def _hash_payload(value: dict[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return canonical_hash(payload)


def _validate_manifest(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or type(files) is not dict
        or manifest.get("artifact_schema") != ARTIFACT_SCHEMA
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("S4 manifest envelope differs")
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(files) != expected:
        raise ValueError("S4 manifest file inventory differs")
    for relative, expected_hash in files.items():
        if type(relative) is not str or file_sha256(root / relative) != expected_hash:
            raise ValueError(f"S4 manifest digest differs: {relative}")
    if manifest.get("manifest_hash") != _hash_payload(manifest, "manifest_hash"):
        raise ValueError("S4 manifest self hash differs")
    return manifest


def _validate_slot(slot: dict[str, Any], ordinal: int) -> tuple[str, str]:
    if slot.get("slot_ordinal") != ordinal:
        raise ValueError("S4 mutable slot order differs")
    alpha_path = slot.get("alpha_semantic_path")
    beta_path = slot.get("beta_semantic_path")
    if not isinstance(alpha_path, str) or not isinstance(beta_path, str):
        raise TypeError("S4 mutable semantic path differs")
    if slot.get("alpha_active_element_count") != _shape_product(
        slot.get("alpha_active_shape")
    ) or slot.get("alpha_preserved_element_count") != _shape_product(
        slot.get("alpha_preserved_shape")
    ):
        raise ValueError("S4 alpha slot arithmetic differs")
    beta_count = _shape_product(slot.get("beta_source_shape"))
    if slot.get("beta_element_count") != beta_count or slot.get("beta_active") is not (
        beta_count > 0
    ):
        raise ValueError("S4 beta slot arithmetic differs")
    if (
        slot.get("alpha_live_is_leaf") is not True
        or slot.get("beta_live_is_leaf") is not True
        or slot.get("alpha_live_requires_grad") is not False
        or slot.get("beta_live_requires_grad") is not True
        or slot.get("entry_content_capture_ordinal") != 1
        or slot.get("exit_content_capture_ordinal") != 2
    ):
        raise ValueError("S4 provider readiness/capture order differs")
    return alpha_path, beta_path


def _validate_receipt(receipt: dict[str, Any], ordinal: int) -> dict[str, int]:
    if receipt.get("admission_hash") != _hash_payload(receipt, "admission_hash"):
        raise ValueError("S4 admission receipt hash differs")
    if receipt.get("exact_call_identity_hash") != canonical_hash(
        {"exact_call_id": f"asplos27-s4-formal:{ordinal:03d}"}
    ):
        raise ValueError("S4 exact-call identity hash differs")
    slots = receipt.get("slots")
    if type(slots) is not list or len(slots) != EXPECTED_COUNTS["slot_count"]:
        raise ValueError("S4 mutable slot inventory differs")
    paths: list[str] = []
    for index, raw_slot in enumerate(slots):
        if type(raw_slot) is not dict:
            raise TypeError("S4 mutable slot type differs")
        paths.extend(_validate_slot(raw_slot, index))
    if len(set(paths)) != len(paths) or receipt.get(
        "mutable_path_set_hash"
    ) != canonical_hash(sorted(paths)):
        raise ValueError("S4 mutable path set differs")
    derived = {
        "slot_count": len(slots),
        "path_count": len(paths),
        "alpha_source_count": len(slots),
        "alpha_stored_element_count": sum(
            _shape_product(slot["alpha_source_shape"]) for slot in slots
        ),
        "alpha_active_element_count": sum(
            int(slot["alpha_active_element_count"]) for slot in slots
        ),
        "alpha_preserved_element_count": sum(
            int(slot["alpha_preserved_element_count"]) for slot in slots
        ),
        "beta_slot_count": len(slots),
        "active_beta_slot_count": sum(int(slot["beta_active"]) for slot in slots),
        "active_beta_element_count": sum(
            int(slot["beta_element_count"]) for slot in slots
        ),
        "live_tensor_count": len(paths),
    }
    derived["live_element_count_per_pass"] = (
        derived["alpha_stored_element_count"] + derived["active_beta_element_count"]
    )
    derived["live_bytes_per_pass"] = 4 * derived["live_element_count_per_pass"]
    for name in (
        "live_content_capture_pass_count",
        "device_to_host_validation_copy_count",
        "device_to_host_validation_bytes",
    ):
        derived[name] = int(receipt.get(name, -1))
    if derived != EXPECTED_COUNTS or any(
        receipt.get(name) != value
        for name, value in EXPECTED_COUNTS.items()
        if name not in {"slot_count", "path_count"}
    ):
        raise ValueError("S4 formal count arithmetic differs")
    if (
        receipt.get("candidate_kernel_launch_count") != 0
        or receipt.get("candidate_cuda_allocation_count") != 0
        or receipt.get("dense_materialization_observed") is not False
        or receipt.get("timing_recorded") is not False
        or receipt.get("performance_claimed") is not False
        or receipt.get("process_global_query_exclusivity_validated") is not False
    ):
        raise ValueError("S4 admission claim boundary differs")
    return derived


def _validate_provider(provider: dict[str, Any], receipt: dict[str, Any]) -> None:
    expected_structure = {
        "alpha_data": "collections.defaultdict",
        "beta_data": "builtins.dict",
    }
    structure = provider.get("structure")
    projection = provider.get("live_projection")
    if type(structure) is not dict or any(
        structure.get(name) != value for name, value in expected_structure.items()
    ):
        raise ValueError("S4 provider owner structure differs")
    rows = structure.get("slots")
    if type(rows) is not list or len(rows) != 6:
        raise ValueError("S4 provider structure slot inventory differs")
    for row in rows:
        if row != {
            "alpha_nested": "builtins.dict",
            "alpha_tensor": "torch.Tensor",
            "beta_collection": "builtins.list",
            "beta_entry": "auto_LiRPA.beta_crown.SparseBeta",
            "beta_tensor": "torch.Tensor",
        }:
            raise ValueError("S4 provider nested structure differs")
    if type(projection) is not list or len(projection) != 12:
        raise ValueError("S4 provider live projection inventory differs")
    slots = receipt["slots"]
    receipt_projection: dict[str, tuple[Any, ...]] = {}
    for slot in slots:
        receipt_projection[slot["alpha_semantic_path"]] = (
            slot["alpha_source_shape"],
            slot["alpha_source_dtype"],
            slot["alpha_source_device"],
            slot["alpha_live_stride"],
            slot["alpha_live_storage_offset"],
            slot["alpha_live_version"],
            slot["alpha_live_requires_grad"],
            slot["alpha_live_is_leaf"],
            slot["alpha_live_content_hash"],
        )
        receipt_projection[slot["beta_semantic_path"]] = (
            slot["beta_source_shape"],
            slot["beta_source_dtype"],
            slot["beta_source_device"],
            slot["beta_live_stride"],
            slot["beta_live_storage_offset"],
            slot["beta_live_version"],
            slot["beta_live_requires_grad"],
            slot["beta_live_is_leaf"],
            slot["beta_live_content_hash"],
        )
    for row in projection:
        if type(row) is not dict or row.get("python_type") != "torch.Tensor":
            raise ValueError("S4 provider tensor type differs")
        path = row.get("semantic_path")
        observed = (
            row.get("shape"),
            row.get("dtype"),
            row.get("device"),
            row.get("stride"),
            row.get("storage_offset"),
            row.get("version"),
            row.get("requires_grad"),
            row.get("is_leaf"),
            row.get("content_hash"),
        )
        if path not in receipt_projection or receipt_projection[path] != observed:
            raise ValueError("S4 provider projection/receipt binding differs")


def _validate_worker(row: dict[str, Any], ordinal: int) -> dict[str, int]:
    if (
        row.get("schema_version") != WORKER_SCHEMA
        or row.get("raw_hash") != _hash_payload(row, "raw_hash")
        or row.get("source_hash") != canonical_hash(row.get("source"))
        or row.get("protocol_hash") != canonical_hash(row.get("protocol"))
        or row.get("performance_claimed") is not False
    ):
        raise ValueError("S4 worker envelope/hash differs")
    admission = row.get("admission")
    if type(admission) is not dict or admission.get("schema_version") != WORKER_SCHEMA:
        raise ValueError("S4 admission payload envelope differs")
    if admission.get("worker_payload_hash") != _hash_payload(
        admission, "worker_payload_hash"
    ):
        raise ValueError("S4 admission worker payload hash differs")
    if admission.get("run_ordinal") != ordinal:
        raise ValueError("S4 worker run ordinal differs")
    receipt = admission.get("receipt")
    provider = admission.get("provider")
    lease = admission.get("lease")
    counters = admission.get("counters")
    if not all(type(value) is dict for value in (receipt, provider, lease, counters)):
        raise TypeError("S4 admission projection type differs")
    receipt = cast(dict[str, Any], receipt)
    provider = cast(dict[str, Any], provider)
    lease = cast(dict[str, Any], lease)
    counters = cast(dict[str, Any], counters)
    derived = _validate_receipt(receipt, ordinal)
    _validate_provider(provider, receipt)
    if lease != {
        "state_before_close": "OPEN",
        "retained_tensor_count_before_close": 12,
        "state_after_close": "CLOSED",
        "retained_tensor_count_after_close": 0,
        "single_transfer_observed": False,
        "buffer_prepare_count": 0,
    }:
        raise ValueError("S4 lease close evidence differs")
    expected_counters = {
        "provider_core_intercept_count": 1,
        "provider_core_execute_count": 0,
        "provider_compute_bounds_callback_count": 0,
        "provider_update_bounds_callback_count": 0,
        "candidate_kernel_launch_count": 0,
        "candidate_cuda_allocation_count": 0,
        "fallback_count": 0,
        "retry_count": 0,
        "mutation_count": 0,
    }
    if (
        counters != expected_counters
        or admission.get("performance_claimed") is not False
    ):
        raise ValueError("S4 pre-execution counter/claim boundary differs")
    return derived


def _derive_summary(
    rows: list[dict[str, Any]], protocol: dict[str, Any], negative: dict[str, Any]
) -> dict[str, Any]:
    counts = [_validate_worker(row, ordinal) for ordinal, row in enumerate(rows)]
    if len(rows) != 5 or any(value != EXPECTED_COUNTS for value in counts):
        raise ValueError("S4 five-fresh inventory differs")
    source_hashes = {row["source_hash"] for row in rows}
    protocol_hashes = {row["protocol_hash"] for row in rows}
    if source_hashes != {protocol["source_hash"]} or protocol_hashes != {
        protocol["worker_protocol_hash"]
    }:
        raise ValueError("S4 worker source/protocol binding differs")
    registry = negative.get("cases")
    if (
        negative.get("schema_version") != NEGATIVE_SCHEMA
        or type(registry) is not list
        or len(registry) < int(protocol["negative_case_minimum"])
        or negative.get("case_count") != len(registry)
        or len({row.get("nodeid") for row in registry}) != len(registry)
        or any(
            row.get("exact_detail_and_reason_asserted") is not True for row in registry
        )
        or negative.get("targeted_result") != "pass"
    ):
        raise ValueError("S4 negative registry differs")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": FORMAL_STATUS,
        "fresh_process_count": len(rows),
        "run_ordinals": list(range(5)),
        "distinct_raw_hash_count": len({row["raw_hash"] for row in rows}),
        "distinct_admission_hash_count": len(
            {row["admission"]["receipt"]["admission_hash"] for row in rows}
        ),
        "formal_counts": EXPECTED_COUNTS,
        "negative_case_minimum": int(protocol["negative_case_minimum"]),
        "negative_case_count": len(registry),
        "candidate_kernel_launch_count": 0,
        "candidate_cuda_allocation_count": 0,
        "provider_bound_callback_count": 0,
        "buffer_prepare_count": 0,
        "mutation_count": 0,
        "timing_recorded": False,
        "performance_claimed": False,
        "process_global_query_exclusivity_validated": False,
        "workers_jsonl_sha256": protocol["workers_jsonl_sha256"],
        "negative_registry_sha256": protocol["negative_registry_sha256"],
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def replay(root: Path) -> dict[str, Any]:
    """Validate hashes and semantically recompute the complete S4-0 artifact."""

    root = root.resolve()
    _validate_manifest(root)
    protocol = load_json(root / "protocol.json")
    summary = load_json(root / "summary.json")
    negative = load_json(root / "negative_registry.json")
    rows = _load_rows(root / "raw/workers.jsonl")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("artifact_schema") != ARTIFACT_SCHEMA
        or protocol.get("protocol_hash") != _hash_payload(protocol, "protocol_hash")
        or protocol.get("fresh_process_count") != 5
        or protocol.get("performance_claimed") is not False
        or file_sha256(root / "raw/workers.jsonl")
        != protocol.get("workers_jsonl_sha256")
        or file_sha256(root / "negative_registry.json")
        != protocol.get("negative_registry_sha256")
    ):
        raise ValueError("S4 protocol/hash binding differs")
    derived = _derive_summary(rows, protocol, negative)
    if summary != derived:
        raise ValueError("S4 summary semantic recomputation differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    summary = replay(args.artifact)
    print(
        "S4-0 admission replay PASS: "
        f"workers={summary['fresh_process_count']} "
        f"negative={summary['negative_case_count']} "
        f"status={summary['status']}"
    )


if __name__ == "__main__":
    main()
