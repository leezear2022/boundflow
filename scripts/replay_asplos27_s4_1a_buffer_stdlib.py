#!/usr/bin/env python3
"""Replay the S4-1A twelve-process artifact using only the stdlib."""

# pylint: disable=missing-function-docstring,unidiomatic-typecheck
# pylint: disable=too-many-locals,too-many-branches,too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

ARTIFACT_SCHEMA = "boundflow.asplos27-s4-1a-buffer-artifact/v1"
MANIFEST_SCHEMA = "boundflow.asplos27-s4-1a-buffer-manifest/v1"
PROTOCOL_SCHEMA = "boundflow.asplos27-s4-1a-buffer-protocol/v1"
NEGATIVE_SCHEMA = "boundflow.asplos27-s4-1a-negative-registry/v1"
SUMMARY_SCHEMA = "boundflow.asplos27-s4-1a-buffer-summary/v1"
FORMAL_STATUS = "FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1A"
EXPECTED_CONSTRUCTION_HASH = (
    "8ad25c2abf1eb98c3b1097bf7acb46aba227f7e94f0c7c03169f39e8da409a9d"
)
EXPECTED_FAULTS = (
    (
        "parameter",
        "BUFFER_PREPARE_MANIFEST_MISMATCH",
        "RECEIPT_IDENTITY_MISMATCH",
    ),
    (
        "gradient",
        "BUFFER_PREPARE_MANIFEST_MISMATCH",
        "RECEIPT_IDENTITY_MISMATCH",
    ),
    (
        "output",
        "BUFFER_PREPARE_RESOURCE_CONTEXT_RETAINED",
        "UNSAFE_ALIAS_OR_LIFETIME",
    ),
    ("view", "BASE_DLPACK_VIEW_COUNT_MISMATCH", "RECEIPT_IDENTITY_MISMATCH"),
    (
        "roundtrip",
        "BASE_DLPACK_VIEW_KEY_MISMATCH",
        "RECEIPT_IDENTITY_MISMATCH",
    ),
    (
        "receipt",
        "BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH",
        "RECEIPT_IDENTITY_MISMATCH",
    ),
    (
        "adoption",
        "BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH",
        "UNSAFE_ALIAS_OR_LIFETIME",
    ),
)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"S4-1A JSON root differs: {path.name}")
    return value


def _hash_without(value: dict[str, Any], field: str) -> str:
    return canonical_hash({key: item for key, item in value.items() if key != field})


def _validate_manifest(root: Path) -> None:
    manifest = load_json(root / "manifest.json")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("artifact_schema") != ARTIFACT_SCHEMA
        or type(files) is not dict
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("S4-1A manifest envelope differs")
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(files) != expected:
        raise ValueError("S4-1A manifest inventory differs")
    for relative, digest in files.items():
        if type(relative) is not str or file_sha256(root / relative) != digest:
            raise ValueError(f"S4-1A manifest digest differs: {relative}")
    if manifest.get("manifest_hash") != _hash_without(manifest, "manifest_hash"):
        raise ValueError("S4-1A manifest self hash differs")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(type(row) is not dict for row in rows):
        raise TypeError("S4-1A worker row differs")
    return rows


def _shape_product(shape: object) -> int:
    if type(shape) is not list or any(
        type(item) is not int or item < 0 for item in shape
    ):
        raise TypeError("S4-1A shape differs")
    return math.prod(shape)


def _validate_buffer_receipt(receipt: dict[str, Any]) -> dict[str, int]:
    if receipt.get("receipt_hash") != _hash_without(receipt, "receipt_hash"):
        raise ValueError("S4-1A receipt hash differs")
    if receipt.get("construction_model_hash") != EXPECTED_CONSTRUCTION_HASH:
        raise ValueError("S4-1A construction identity differs")
    descriptors = receipt.get("buffer_descriptors")
    tokens = receipt.get("empty_beta_tokens")
    if type(descriptors) is not list or len(descriptors) != 16:
        raise ValueError("S4-1A descriptor inventory differs")
    if type(tokens) is not list or len(tokens) != 5:
        raise ValueError("S4-1A empty beta inventory differs")
    expected_roles = (
        *("alpha_parameter" for _ in range(6)),
        "active_beta_parameter",
        *("alpha_gradient" for _ in range(6)),
        "active_beta_gradient",
        "lower_output",
        "fixed_upstream",
    )
    initialized = 0
    initialized_bytes = 0
    total_bytes = 0
    for ordinal, (descriptor, role) in enumerate(zip(descriptors, expected_roles)):
        if type(descriptor) is not dict:
            raise TypeError("S4-1A descriptor type differs")
        elements = _shape_product(descriptor.get("shape"))
        logical_bytes = 4 * elements
        if (
            descriptor.get("buffer_ordinal") != ordinal
            or descriptor.get("view_ordinal") != ordinal
            or descriptor.get("semantic_role") != role
            or descriptor.get("element_count") != elements
            or descriptor.get("logical_bytes") != logical_bytes
            or descriptor.get("dtype") != "torch.float32"
            or not str(descriptor.get("device", "")).startswith("cuda:")
            or descriptor.get("storage_offset") != 0
            or descriptor.get("is_leaf") is not True
            or descriptor.get("contiguous") is not True
            or descriptor.get("requires_grad") is not role.endswith("parameter")
        ):
            raise ValueError("S4-1A descriptor semantics differ")
        if descriptor.get("initialized_at_prepare") is True:
            initial_hash = descriptor.get("initial_content_hash_or_none")
            if type(initial_hash) is not str or len(initial_hash) != 64:
                raise ValueError("S4-1A initialized descriptor hash differs")
            initialized += 1
            initialized_bytes += logical_bytes
        elif descriptor.get("initial_content_hash_or_none") is not None:
            raise ValueError("S4-1A uninitialized descriptor was read")
        total_bytes += logical_bytes
    for ordinal, token in enumerate(tokens):
        if (
            type(token) is not dict
            or token.get("slot_ordinal") != ordinal
            or token.get("shape") != [6, 0]
            or token.get("physical_buffer_present") is not False
            or token.get("physical_view_present") is not False
            or token.get("optimizer_ordinal") != -1
        ):
            raise ValueError("S4-1A empty beta token differs")
    exact = {
        "parameter_count": 7,
        "gradient_count": 7,
        "empty_beta_token_count": 5,
        "candidate_storage_count": 16,
        "base_dlpack_view_count": 16,
        "parameter_elements": 4254,
        "parameter_bytes": 17016,
        "gradient_elements": 4254,
        "gradient_bytes": 17016,
        "candidate_logical_bytes": 34080,
        "leased_source_tensor_count": 12,
        "leased_source_elements": 8502,
        "leased_source_bytes": 34008,
        "source_d2h_copy_count": 24,
        "source_d2h_bytes": 68016,
        "initialized_candidate_d2h_copy_count": 8,
        "initialized_candidate_d2h_bytes": 17040,
        "s4_1a_d2h_copy_count": 32,
        "s4_1a_d2h_bytes": 85056,
        "prior_s4_0_d2h_copy_count": 24,
        "prior_s4_0_d2h_bytes": 68016,
        "cumulative_d2h_copy_count": 56,
        "cumulative_d2h_bytes": 153072,
        "parameter_d2d_copy_count": 7,
        "parameter_d2d_bytes": 17016,
    }
    if any(receipt.get(name) != value for name, value in exact.items()):
        raise ValueError("S4-1A receipt accounting differs")
    if total_bytes != 34080 or initialized != 8 or initialized_bytes != 17040:
        raise ValueError("S4-1A descriptor arithmetic differs")
    zero_fields = (
        "warm_dlpack_view_count",
        "full_alpha_device_copy_count",
        "dense_alpha_materialization_count",
        "dense_beta_materialization_count",
        "prepare_retry_count",
        "prepare_fallback_count",
        "empty_cache_call_count",
    )
    false_fields = (
        "provider_mapping_stability_validated",
        "process_global_exclusivity_validated",
        "crown_numeric_semantics_validated",
        "optimizer_trajectory_validated",
        "timing_recorded",
        "performance_claimed",
    )
    if any(receipt.get(name) != 0 for name in zero_fields) or any(
        receipt.get(name) is not False for name in false_fields
    ):
        raise ValueError("S4-1A claim boundary differs")
    return exact


def _validate_binary(root: Path, payload: dict[str, Any]) -> None:
    sidecar = payload.get("binary_sidecar")
    records = payload.get("binary_index")
    if type(sidecar) is not dict or type(records) is not list or len(records) != 8:
        raise ValueError("S4-1A binary sidecar inventory differs")
    relative = sidecar.get("relative_path")
    if type(relative) is not str:
        raise TypeError("S4-1A sidecar path differs")
    data = (root / relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != sidecar.get("sha256"):
        raise ValueError("S4-1A sidecar digest differs")
    for record in records:
        if type(record) is not dict:
            raise TypeError("S4-1A binary index row differs")
        count = record.get("byte_count")
        source_offset = record.get("source_offset")
        candidate_offset = record.get("candidate_offset")
        if any(
            type(value) is not int or value < 0
            for value in (count, source_offset, candidate_offset)
        ):
            raise TypeError("S4-1A binary index offset differs")
        count = cast(int, count)
        source_offset = cast(int, source_offset)
        candidate_offset = cast(int, candidate_offset)
        source = data[source_offset : source_offset + count]
        candidate = data[candidate_offset : candidate_offset + count]
        if (
            len(source) != count
            or len(candidate) != count
            or source != candidate
            or hashlib.sha256(source).hexdigest() != record.get("source_sha256")
            or hashlib.sha256(candidate).hexdigest() != record.get("candidate_sha256")
            or 4 * _shape_product(record.get("shape")) != count
            or record.get("dtype") != "torch.float32"
        ):
            raise ValueError("S4-1A source/candidate binary semantics differ")


def _validate_counters(payload: dict[str, Any]) -> None:
    counters = payload.get("counters")
    if type(counters) is not dict or counters != {
        "candidate_kernel_launch_count": 0,
        "empty_cache_count": 0,
        "fallback_count": 0,
        "mutation_count": 0,
        "provider_compute_bounds_callback_count": 0,
        "provider_core_execute_count": 0,
        "provider_core_intercept_count": 1,
        "provider_update_bounds_callback_count": 0,
        "retry_count": 0,
    }:
        raise ValueError("S4-1A execution boundary differs")
    if (
        payload.get("timing_recorded") is not False
        or payload.get("performance_claimed") is not False
    ):
        raise ValueError("S4-1A performance claim differs")


def _derive_summary(
    root: Path,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != 12 or tuple(
        row.get("admission", {}).get("mode") for row in rows
    ) != (
        *("positive" for _ in range(5)),
        *("fault" for _ in range(7)),
    ):
        raise ValueError("S4-1A worker sequence differs")
    receipt_hashes: list[str] = []
    formal_counts: dict[str, int] | None = None
    for ordinal, row in enumerate(rows):
        if row.get("raw_hash") != _hash_without(row, "raw_hash"):
            raise ValueError("S4-1A worker raw hash differs")
        if row.get("schema_version") != "boundflow.asplos27-s4-1a-buffer-worker/v1":
            raise ValueError("S4-1A worker schema differs")
        payload = row.get("admission")
        if type(payload) is not dict or payload.get("run_ordinal") != ordinal:
            raise ValueError("S4-1A worker ordinal differs")
        source = row.get("source")
        worker_protocol = row.get("protocol")
        if (
            type(source) is not dict
            or row.get("source_hash") != canonical_hash(source)
            or type(worker_protocol) is not dict
            or row.get("protocol_hash") != canonical_hash(worker_protocol)
            or row.get("performance_claimed") is not False
        ):
            raise ValueError("S4-1A worker source/protocol identity differs")
        if payload.get("worker_payload_hash") != _hash_without(
            payload, "worker_payload_hash"
        ):
            raise ValueError("S4-1A worker payload hash differs")
        _validate_counters(payload)
        fault_name = "none" if ordinal < 5 else EXPECTED_FAULTS[ordinal - 5][0]
        expected_call_hash = canonical_hash(
            {"exact_call_id": f"asplos27-s4-1a-formal:{ordinal:03d}:{fault_name}"}
        )
        if ordinal < 5:
            admission_receipt = payload.get("admission_receipt")
            receipt = payload.get("buffer_receipt")
            physical = payload.get("physical")
            allocator = payload.get("allocator")
            if (
                type(admission_receipt) is not dict
                or admission_receipt.get("admission_hash")
                != _hash_without(admission_receipt, "admission_hash")
                or admission_receipt.get("exact_call_identity_hash")
                != expected_call_hash
                or type(receipt) is not dict
                or receipt.get("admission_hash")
                != admission_receipt.get("admission_hash")
                or receipt.get("snapshot_hash")
                != admission_receipt.get("snapshot_hash")
                or receipt.get("plan_hash")
                != admission_receipt.get("production_plan_hash")
                or receipt.get("exact_call_identity_hash") != expected_call_hash
                or type(physical) is not dict
                or type(allocator) is not dict
            ):
                raise TypeError("S4-1A positive payload differs")
            formal_counts = _validate_buffer_receipt(receipt)
            _validate_binary(root, payload)
            if physical != {
                "base_dlpack_view_count": 16,
                "candidate_storage_count": 16,
                "candidate_storage_unique_count": 16,
                "empty_beta_physical_count": 0,
                "private_view_key_unique_count": 16,
                "resource_state_after_close": "CLOSED",
            }:
                raise ValueError("S4-1A physical owner evidence differs")
            if allocator.get("allocated_delta_after_close") != 0:
                raise ValueError("S4-1A positive cleanup differs")
            receipt_hashes.append(str(receipt["receipt_hash"]))
        else:
            fault, detail, reason = EXPECTED_FAULTS[ordinal - 5]
            error = payload.get("error")
            allocator = payload.get("allocator")
            if (
                payload.get("fault") != fault
                or payload.get("exact_call_identity_hash") != expected_call_hash
                or type(error) is not dict
                or error.get("detail_code") != detail
                or error.get("verification_reason") != reason
                or error.get("context_is_none") is not True
                or type(allocator) is not dict
                or allocator.get("allocated_delta") != 0
                or payload.get("admission_state") != "TRANSFERRED"
            ):
                raise ValueError("S4-1A isolated fault cleanup differs")
    if formal_counts is None:
        raise ValueError("S4-1A positive counts absent")
    if (
        registry.get("schema_version") != NEGATIVE_SCHEMA
        or int(registry.get("case_count", 0)) < 68
        or registry.get("targeted_result") != "pass"
    ):
        raise ValueError("S4-1A negative registry differs")
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "status": FORMAL_STATUS,
        "source_revision": protocol["source_revision"],
        "fresh_process_count": 12,
        "positive_process_count": 5,
        "isolated_fault_process_count": 7,
        "negative_case_count": registry["case_count"],
        "formal_counts": formal_counts,
        "receipt_hashes": receipt_hashes,
        "source_candidate_binary_pair_count": 40,
        "source_candidate_binary_exact_count": 40,
        "isolated_fault_clean_count": 7,
        "candidate_kernel_launch_count": 0,
        "fallback_count": 0,
        "retry_count": 0,
        "mutation_count": 0,
        "timing_recorded": False,
        "performance_claimed": False,
        "workers_jsonl_sha256": protocol["workers_jsonl_sha256"],
        "negative_registry_sha256": protocol["negative_registry_sha256"],
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def replay(root: Path) -> dict[str, Any]:
    root = root.resolve()
    _validate_manifest(root)
    protocol = load_json(root / "protocol.json")
    registry = load_json(root / "negative_registry.json")
    rows_path = root / "raw/workers.jsonl"
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("artifact_schema") != ARTIFACT_SCHEMA
        or protocol.get("protocol_hash") != _hash_without(protocol, "protocol_hash")
        or protocol.get("workers_jsonl_sha256") != file_sha256(rows_path)
        or protocol.get("negative_registry_sha256")
        != file_sha256(root / "negative_registry.json")
        or protocol.get("timing_recorded") is not False
        or protocol.get("performance_claimed") is not False
    ):
        raise ValueError("S4-1A protocol differs")
    derived = _derive_summary(root, _load_rows(rows_path), protocol, registry)
    stored = load_json(root / "summary.json")
    if stored != derived:
        raise ValueError("S4-1A derived summary differs")
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    summary = replay(args.artifact)
    print(
        "S4-1A stdlib replay PASS: "
        f"workers={summary['fresh_process_count']} "
        f"binary={summary['source_candidate_binary_exact_count']} "
        f"faults={summary['isolated_fault_clean_count']} "
        f"status={summary['status']}"
    )


if __name__ == "__main__":
    main()
