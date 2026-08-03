#!/usr/bin/env python3
"""Generate or replay NRIR-7 real repeated-query batching/cache evidence."""

# pylint: disable=duplicate-code,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.planner import plan_interval_ibp_v0
from boundflow.planner.representation_plan_binding import DENSE_POLICY_ID
from boundflow.runtime.abcrown_adapter import (
    bind_intermediate_bounds,
    deserialize_intermediate_bounds,
    file_sha256,
    intermediate_bounds_sha256,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_repeated_query_runtime import (
    NATIVE_REPEATED_QUERY_COMPILER_VERSION,
    REPEATED_QUERY_EXECUTION_TRACE_SCHEMA_VERSION,
    REPEATED_QUERY_LAYOUT_SCHEMA_VERSION,
    SERIAL_QUERY_TRACE_SCHEMA_VERSION,
    NativeRepeatedQueryCompilationCache,
    NativeRepeatedQuerySpec,
    RepeatedQueryCompileResult,
    compile_native_repeated_query_stream,
    execute_native_repeated_query_serial_reference,
    execute_native_repeated_query_stream,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_native_real_network_ir_artifact import (
    ABCROWN_COMMIT,
    EXPECTED_PRIMAL_OPS,
    INTERMEDIATE_BOUNDS_SHA256,
    MODEL_SHA256,
    VNNCOMP_COMMIT,
    VNNLIB_SHA256,
)
from scripts.run_native_real_network_memory_plans_artifact import (
    _load_source_artifact,
    _payload_tensors,
)

REPEATED_QUERY_ARTIFACT_SCHEMA_VERSION = (
    "boundflow.native-real-network-repeated-query-artifact/v1"
)
REPEATED_QUERY_EVIDENCE_SCHEMA_VERSION = (
    "boundflow.native-real-network-repeated-query-evidence/v1"
)
STREAM_ID = "vnncomp21-resnet2b-prop0-native-ir7-query-stream"
AVAILABLE_MEMORY_BYTES = 1 << 30
SPEC_BATCH_SIZE = 3
ATOL = 2e-4
RTOL = 2e-4
ARTIFACT_FILES = ("repeated_query.json",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model", type=Path, required=True)
        subparser.add_argument("--source-artifact-dir", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def build_repeated_query_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    """Execute nine property queries as packed-3 and serial-9 paths."""

    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-7 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    external_bounds = deserialize_intermediate_bounds(
        _mapping(payload.get("external_intermediate_bounds"), "intermediate bounds")
    )
    if intermediate_bounds_sha256(external_bounds) != INTERMEDIATE_BOUNDS_SHA256:
        raise ValueError("NRIR-7 intermediate-bound digest differs")
    linear_spec = tensors["linear_spec_c"]
    if tuple(linear_spec.shape[:2]) != (1, 9):
        raise ValueError("NRIR-7 frozen objective layout differs")

    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    legacy_module = plan_interval_ibp_v0(program)
    if tuple(op.op_type for op in legacy_module.get_entry_task().ops) != (
        EXPECTED_PRIMAL_OPS
    ):
        raise ValueError("NRIR-7 primal topology differs")
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    interval_env, local_relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    relu_pre = bind_intermediate_bounds(external_bounds, local_relu_pre)
    query_specs = tuple(
        NativeRepeatedQuerySpec(
            query_id=f"property-{index:02d}",
            linear_spec_C=linear_spec[:, index : index + 1],
        )
        for index in range(9)
    )
    workload_identity_hash = hashlib.sha256(
        f"{MODEL_SHA256}:{VNNLIB_SHA256}".encode("utf-8")
    ).hexdigest()
    state_identity_hash = hashlib.sha256(
        (
            f"{INTERMEDIATE_BOUNDS_SHA256}:"
            f"{tensor_content_hash(tensors['input_lower'])}:"
            f"{tensor_content_hash(tensors['input_upper'])}"
        ).encode("utf-8")
    ).hexdigest()
    cache = NativeRepeatedQueryCompilationCache()

    def compile_specs(
        specs: tuple[NativeRepeatedQuerySpec, ...],
        *,
        state_hash: str = state_identity_hash,
    ) -> RepeatedQueryCompileResult:
        return compile_native_repeated_query_stream(
            cache,
            legacy_module,
            input_spec,
            specs,
            interval_env=interval_env,
            relu_pre=relu_pre,
            intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
            stream_id=STREAM_ID,
            workload_identity_hash=workload_identity_hash,
            state_identity_hash=state_hash,
            available_memory_bytes=AVAILABLE_MEMORY_BYTES,
            memory_budget_bytes=AVAILABLE_MEMORY_BYTES,
            spec_slice_candidate_size=SPEC_BATCH_SIZE,
            max_spec_batch_size=SPEC_BATCH_SIZE,
        )

    first = compile_specs(query_specs)
    second = compile_specs(query_specs)
    primary_cache_counts = {
        "entries": len(cache.entries),
        "hits": cache.hit_count,
        "misses": cache.miss_count,
    }
    packed_results, miss_trace = execute_native_repeated_query_stream(
        first,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    cached_results, hit_trace = execute_native_repeated_query_stream(
        second,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    serial_results, serial_trace = execute_native_repeated_query_serial_reference(
        first.compilation,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    )
    packed_lower = torch.cat(tuple(item.result.lower for item in packed_results), dim=1)
    packed_upper = torch.cat(tuple(item.result.upper for item in packed_results), dim=1)
    cached_lower = torch.cat(tuple(item.result.lower for item in cached_results), dim=1)
    cached_upper = torch.cat(tuple(item.result.upper for item in cached_results), dim=1)
    serial_lower = torch.cat(tuple(item.result.lower for item in serial_results), dim=1)
    serial_upper = torch.cat(tuple(item.result.upper for item in serial_results), dim=1)
    expected = tensors["external_lower"].to(packed_lower)
    comparisons = {
        "packed_vs_serial_lower": _comparison(packed_lower, serial_lower),
        "packed_vs_cached_lower": _comparison(packed_lower, cached_lower),
        "packed_vs_external_lower": _comparison(packed_lower, expected),
        "serial_vs_external_lower": _comparison(serial_lower, expected),
    }
    torch.testing.assert_close(packed_upper, serial_upper, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(packed_upper, cached_upper, atol=ATOL, rtol=RTOL)
    if not all(
        item["allclose"] is True and item["sign_agreement"] == item["sign_total"] == 9
        for item in comparisons.values()
    ):
        raise ValueError("NRIR-7 packed/serial/external semantics differ")

    changed_tensor = query_specs[0].linear_spec_C.clone()
    changed_tensor[0, 0, 0] += 1.0
    changed_specs = (
        NativeRepeatedQuerySpec(query_specs[0].query_id, changed_tensor),
        *query_specs[1:],
    )
    changed = compile_specs(changed_specs)
    reordered = compile_specs(tuple(reversed(query_specs)))
    changed_state = compile_specs(query_specs, state_hash="f" * 64)
    key_probe = {
        "primary_cache_key": first.compilation.cache_key,
        "objective_tamper_cache_key": changed.compilation.cache_key,
        "reordered_cache_key": reordered.compilation.cache_key,
        "state_tamper_cache_key": changed_state.compilation.cache_key,
        "all_distinct": len(
            {
                first.compilation.cache_key,
                changed.compilation.cache_key,
                reordered.compilation.cache_key,
                changed_state.compilation.cache_key,
            }
        )
        == 4,
    }
    query_contract = [item.to_dict() for item in query_specs]
    layout = first.compilation.layout_trace.to_dict()
    packed_result_payload = [item.to_dict() for item in packed_results]
    serial_result_payload = [item.to_dict() for item in serial_results]
    per_query_match = all(
        packed_item.query_id == serial_item.query_id
        and bool(
            torch.allclose(
                packed_item.result.lower,
                serial_item.result.lower,
                atol=ATOL,
                rtol=RTOL,
            )
        )
        and bool(
            torch.allclose(
                packed_item.result.upper,
                serial_item.result.upper,
                atol=ATOL,
                rtol=RTOL,
            )
        )
        for packed_item, serial_item in zip(packed_results, serial_results)
    )
    gates = {
        "nine_distinct_property_queries_are_explicit": bool(
            len(query_contract) == 9
            and len({item["query_id"] for item in query_contract}) == 9
            and all(item["spec_count"] == 1 for item in query_contract)
        ),
        "packed_three_children_replace_nine_serial_children": bool(
            miss_trace.packed_child_stack_count == 3
            and serial_trace.serial_child_stack_count == 9
        ),
        "first_compile_misses_second_exact_compile_hits": bool(
            first.cache_hit is False
            and second.cache_hit is True
            and primary_cache_counts == {"entries": 1, "hits": 1, "misses": 1}
            and second.compilation is first.compilation
        ),
        "cache_key_tracks_objective_order_and_state": bool(
            key_probe["all_distinct"]
            and changed.cache_hit is False
            and reordered.cache_hit is False
            and changed_state.cache_hit is False
        ),
        "nine_results_restore_exact_query_lineage": bool(
            len(packed_result_payload) == len(serial_result_payload) == 9
            and per_query_match
        ),
        "packed_cached_serial_and_external_semantics_match": all(
            item["allclose"] is True for item in comparisons.values()
        ),
        "packed_and_serial_inherit_one_source_policy": bool(
            layout["representation_policy_id"] == DENSE_POLICY_ID
            and serial_trace.representation_policy_id == DENSE_POLICY_ID
            and layout["storage_candidate_id"] == serial_trace.storage_candidate_id
        ),
        "mechanism_counts_do_not_claim_performance": True,
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-7 gates failed: {gates}")

    evidence: dict[str, object] = {
        "schema_version": REPEATED_QUERY_EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "compiler_version": NATIVE_REPEATED_QUERY_COMPILER_VERSION,
        "claim_boundary": (
            "real-network repeated-query formation, packed spec execution, exact "
            "cache, lineage, and serial-reference correctness only; child-count "
            "reduction is not a latency, memory, CUDA, OOM, Pareto, or speedup claim"
        ),
        "source": {
            "native_ir_artifact_schema": source_manifest["schema_version"],
            "native_ir_manifest_sha256": file_sha256(
                source_artifact_dir / "manifest.json"
            ),
            "native_ir_payload_sha256": file_sha256(source_artifact_dir / "payload.pt"),
            "model_sha256": MODEL_SHA256,
            "vnnlib_sha256": VNNLIB_SHA256,
            "vnncomp_commit": VNNCOMP_COMMIT,
            "abcrown_commit": ABCROWN_COMMIT,
            "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
            "workload_identity_hash": workload_identity_hash,
            "state_identity_hash": state_identity_hash,
        },
        "query_contract": query_contract,
        "layout_trace": layout,
        "cache": {
            "primary": primary_cache_counts,
            "first_cache_hit": first.cache_hit,
            "second_cache_hit": second.cache_hit,
            "key_probe": key_probe,
        },
        "packed": {
            "execution_trace_miss": miss_trace.to_dict(),
            "execution_trace_hit": hit_trace.to_dict(),
            "results": packed_result_payload,
            "lower_hash": tensor_content_hash(packed_lower),
            "upper_hash": tensor_content_hash(packed_upper),
        },
        "serial": {
            "trace": serial_trace.to_dict(),
            "results": serial_result_payload,
            "lower_hash": tensor_content_hash(serial_lower),
            "upper_hash": tensor_content_hash(serial_upper),
        },
        "semantics": comparisons,
        "gates": gates,
        "limitations": [
            "same input domain with nine distinct property objectives",
            "BaB parent/child domain state validity remains pending",
            "cache is exact in-process compilation reuse, not a disk/code cache result",
            "child-count reduction is a mechanism count, not a timing result",
            "no memory, latency, CUDA, allocator, OOM, Pareto, or speedup claim",
        ],
    }
    validate_repeated_query_evidence(evidence)
    return evidence


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    difference = (actual - expected).abs()
    sign_match = (actual >= 0) == (expected >= 0)
    return {
        "allclose": bool(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
        "max_abs_diff": float(difference.max().item()),
        "sign_agreement": int(sign_match.sum().item()),
        "sign_total": int(sign_match.numel()),
        "atol": ATOL,
        "rtol": RTOL,
    }


def validate_repeated_query_evidence(evidence: Mapping[str, Any]) -> None:
    """Reject incomplete, relinked, or claim-inflated NRIR-7 evidence."""

    if (
        evidence.get("schema_version") != REPEATED_QUERY_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("compiler_version") != NATIVE_REPEATED_QUERY_COMPILER_VERSION
    ):
        raise ValueError("NRIR-7 evidence contract differs")
    claim = str(evidence.get("claim_boundary", ""))
    if any(
        phrase not in claim
        for phrase in ("repeated-query", "cache", "not a latency", "speedup")
    ):
        raise ValueError("NRIR-7 claim boundary omits hard limitations")
    gates = _mapping(evidence.get("gates"), "NRIR-7 gates")
    expected_gates = {
        "nine_distinct_property_queries_are_explicit",
        "packed_three_children_replace_nine_serial_children",
        "first_compile_misses_second_exact_compile_hits",
        "cache_key_tracks_objective_order_and_state",
        "nine_results_restore_exact_query_lineage",
        "packed_cached_serial_and_external_semantics_match",
        "packed_and_serial_inherit_one_source_policy",
        "mechanism_counts_do_not_claim_performance",
    }
    if set(gates) != expected_gates or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("NRIR-7 gates are incomplete or failed")
    queries = evidence.get("query_contract")
    if not isinstance(queries, list) or len(queries) != 9:
        raise ValueError("NRIR-7 query contract count differs")
    expected_query_ids = [f"property-{index:02d}" for index in range(9)]
    for index, payload in enumerate(queries):
        query = _mapping(payload, "NRIR-7 query contract")
        if (
            query.get("query_id") != expected_query_ids[index]
            or query.get("spec_count") != 1
            or query.get("shape") != [1, 1, 10]
            or len(str(query.get("objective_hash", ""))) != 64
        ):
            raise ValueError("NRIR-7 query identity/objective differs")
    layout = _mapping(evidence.get("layout_trace"), "NRIR-7 layout")
    ranges = layout.get("query_ranges")
    if (
        layout.get("schema_version") != REPEATED_QUERY_LAYOUT_SCHEMA_VERSION
        or layout.get("stream_id") != STREAM_ID
        or layout.get("representation_policy_id") != DENSE_POLICY_ID
        or layout.get("storage_candidate_id") != "storage:native-retain-all-v1"
        or layout.get("batch_candidate_id") != "batch:native-spec-sliced-v1:0003"
        or layout.get("spec_slice_candidate_size") != 3
        or layout.get("max_spec_batch_size") != 3
        or not isinstance(ranges, list)
        or len(ranges) != 9
    ):
        raise ValueError("NRIR-7 layout policy/accounting differs")
    for index, payload in enumerate(ranges):
        item = _mapping(payload, "NRIR-7 query range")
        if (
            item.get("query_id") != expected_query_ids[index]
            or item.get("start_index") != index
            or item.get("stop_index") != index + 1
            or item.get("objective_hash") != queries[index]["objective_hash"]
        ):
            raise ValueError("NRIR-7 query range lineage differs")
    cache = _mapping(evidence.get("cache"), "NRIR-7 cache")
    primary = _mapping(cache.get("primary"), "NRIR-7 primary cache")
    probe = _mapping(cache.get("key_probe"), "NRIR-7 cache probe")
    probe_keys = (
        probe.get("primary_cache_key"),
        probe.get("objective_tamper_cache_key"),
        probe.get("reordered_cache_key"),
        probe.get("state_tamper_cache_key"),
    )
    if (
        primary != {"entries": 1, "hits": 1, "misses": 1}
        or cache.get("first_cache_hit") is not False
        or cache.get("second_cache_hit") is not True
        or probe.get("all_distinct") is not True
        or len(set(probe_keys)) != 4
        or any(len(str(value)) != 64 for value in probe_keys)
    ):
        raise ValueError("NRIR-7 cache identity/accounting differs")
    packed = _mapping(evidence.get("packed"), "NRIR-7 packed path")
    serial = _mapping(evidence.get("serial"), "NRIR-7 serial path")
    miss_trace = _mapping(packed.get("execution_trace_miss"), "NRIR-7 miss execution")
    hit_trace = _mapping(packed.get("execution_trace_hit"), "NRIR-7 hit execution")
    serial_trace = _mapping(serial.get("trace"), "NRIR-7 serial trace")
    expected_result_ids = expected_query_ids
    packed_results = packed.get("results")
    serial_results = serial.get("results")
    if (
        miss_trace.get("schema_version")
        != REPEATED_QUERY_EXECUTION_TRACE_SCHEMA_VERSION
        or hit_trace.get("schema_version")
        != REPEATED_QUERY_EXECUTION_TRACE_SCHEMA_VERSION
        or miss_trace.get("cache_hit") is not False
        or hit_trace.get("cache_hit") is not True
        or miss_trace.get("cache_key") != probe_keys[0]
        or hit_trace.get("cache_key") != probe_keys[0]
        or miss_trace.get("packed_child_stack_count") != 3
        or hit_trace.get("packed_child_stack_count") != 3
        or serial_trace.get("schema_version") != SERIAL_QUERY_TRACE_SCHEMA_VERSION
        or serial_trace.get("serial_child_stack_count") != 9
        or serial_trace.get("representation_policy_id") != DENSE_POLICY_ID
        or not isinstance(packed_results, list)
        or not isinstance(serial_results, list)
        or len(packed_results) != 9
        or len(serial_results) != 9
    ):
        raise ValueError("NRIR-7 packed/serial trace accounting differs")
    for index, (packed_payload, serial_payload) in enumerate(
        zip(packed_results, serial_results)
    ):
        packed_result = _mapping(packed_payload, "NRIR-7 packed result")
        serial_result = _mapping(serial_payload, "NRIR-7 serial result")
        if (
            packed_result.get("query_id") != expected_result_ids[index]
            or serial_result.get("query_id") != expected_result_ids[index]
            or packed_result.get("shape") != [1, 1]
            or serial_result.get("shape") != [1, 1]
            or len(str(packed_result.get("lower_hash", ""))) != 64
            or len(str(packed_result.get("upper_hash", ""))) != 64
        ):
            raise ValueError("NRIR-7 restored query result differs")
    if miss_trace.get("query_results") != packed_results:
        raise ValueError("NRIR-7 packed execution/result linkage differs")
    if serial_trace.get("query_results") != serial_results:
        raise ValueError("NRIR-7 serial execution/result linkage differs")
    semantics = _mapping(evidence.get("semantics"), "NRIR-7 semantics")
    expected_comparisons = {
        "packed_vs_serial_lower",
        "packed_vs_cached_lower",
        "packed_vs_external_lower",
        "serial_vs_external_lower",
    }
    if set(semantics) != expected_comparisons:
        raise ValueError("NRIR-7 semantic comparison set differs")
    for payload in semantics.values():
        comparison = _mapping(payload, "NRIR-7 semantic comparison")
        if (
            comparison.get("allclose") is not True
            or comparison.get("sign_agreement") != 9
            or comparison.get("sign_total") != 9
        ):
            raise ValueError("NRIR-7 semantic comparison differs")


def generate_artifact(
    artifact_dir: Path, *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_repeated_query_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    (artifact_dir / "repeated_query.json").write_text(
        _canonical_json(evidence, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": REPEATED_QUERY_ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "files": {name: file_sha256(artifact_dir / name) for name in ARTIFACT_FILES},
    }
    (artifact_dir / "manifest.json").write_text(
        _canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def replay_artifact(
    artifact_dir: Path, *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    manifest = _load_json(artifact_dir / "manifest.json")
    if (
        manifest.get("schema_version") != REPEATED_QUERY_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-7 manifest contract differs")
    files = _mapping(manifest.get("files"), "NRIR-7 manifest files")
    if set(files) != set(ARTIFACT_FILES):
        raise ValueError("NRIR-7 manifest file set differs")
    for name in ARTIFACT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError(f"NRIR-7 artifact digest differs: {name}")
    stored = _load_json(artifact_dir / "repeated_query.json")
    validate_repeated_query_evidence(stored)
    actual = build_repeated_query_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    if _canonical_json(stored) != _canonical_json(actual):
        raise ValueError("NRIR-7 semantic replay differs")
    return actual


def _main() -> None:
    args = _parse_args()
    model = args.model.expanduser().resolve()
    source_dir = args.source_artifact_dir.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    if args.command == "generate":
        evidence = generate_artifact(
            artifact_dir, model=model, source_artifact_dir=source_dir
        )
    else:
        evidence = replay_artifact(
            artifact_dir, model=model, source_artifact_dir=source_dir
        )
    print(
        _canonical_json(
            {
                "status": "ok",
                "mode": args.command,
                "artifact_dir": str(artifact_dir),
                "gates": evidence["gates"],
            }
        )
    )


if __name__ == "__main__":
    _main()
