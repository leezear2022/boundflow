#!/usr/bin/env python3
"""Generate or replay NRIR-8 real parent/child domain-batching evidence."""

# pylint: disable=duplicate-code,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.planner import plan_interval_ibp_v0
from boundflow.planner.representation_plan_binding import DENSE_POLICY_ID
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.native_domain_batch_runtime import (
    DOMAIN_BATCH_BINDING_SCHEMA_VERSION,
    DOMAIN_BATCH_EXECUTION_TRACE_SCHEMA_VERSION,
    DOMAIN_QUERY_STATE_SCHEMA_VERSION,
    NATIVE_DOMAIN_BATCH_COMPILER_VERSION,
    NativeDomainQueryResult,
    PARENT_STATE_VALIDITY,
    SERIAL_DOMAIN_TRACE_SCHEMA_VERSION,
    build_deterministic_box_domain_queries,
    compile_native_domain_batch_query,
    execute_native_domain_batch_query,
    execute_native_domain_serial_reference,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_native_real_network_ir_artifact import (
    ABCROWN_COMMIT,
    EXPECTED_PRIMAL_OPS,
    MODEL_SHA256,
    VNNCOMP_COMMIT,
    VNNLIB_SHA256,
)
from scripts.run_native_real_network_memory_plans_artifact import (
    _load_source_artifact,
    _payload_tensors,
)

DOMAIN_BATCH_ARTIFACT_SCHEMA_VERSION = (
    "boundflow.native-real-network-domain-batch-artifact/v1"
)
DOMAIN_BATCH_EVIDENCE_SCHEMA_VERSION = (
    "boundflow.native-real-network-domain-batch-evidence/v1"
)
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir8-domain-batch"
ROOT_QUERY_ID = "vnncomp21-resnet2b-prop0-bab-root"
AVAILABLE_MEMORY_BYTES = 1 << 30
DOMAIN_BATCH_SIZE = 4
SPLIT_DEPTH = 3
ATOL = 2e-4
RTOL = 2e-4
ARTIFACT_FILES = ("domain_batch.json",)


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


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _stack_results(
    results: tuple[NativeDomainQueryResult, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.cat(tuple(item.result.lower for item in results), dim=0)
    upper = torch.cat(tuple(item.result.upper for item in results), dim=0)
    return lower, upper


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    difference = (actual - expected).abs()
    return {
        "allclose": bool(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
        "max_abs_diff": float(difference.max().item()),
        "actual_hash": tensor_content_hash(actual),
        "expected_hash": tensor_content_hash(expected),
        "shape": list(actual.shape),
        "atol": ATOL,
        "rtol": RTOL,
    }


def build_domain_batch_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    """Execute eight leaf boxes through packed-4, full-8, and serial-8 paths."""

    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-8 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    linear_spec = tensors["linear_spec_c"]
    if tuple(linear_spec.shape) != (1, 9, 10):
        raise ValueError("NRIR-8 frozen objective layout differs")
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    legacy_module = plan_interval_ibp_v0(program)
    if tuple(op.op_type for op in legacy_module.get_entry_task().ops) != (
        EXPECTED_PRIMAL_OPS
    ):
        raise ValueError("NRIR-8 primal topology differs")
    root_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    query_specs = build_deterministic_box_domain_queries(
        root_spec,
        root_query_id=ROOT_QUERY_ID,
        split_depth=SPLIT_DEPTH,
    )
    objective = linear_spec[:, 0:1].contiguous()
    packed = compile_native_domain_batch_query(
        legacy_module,
        query_specs,
        linear_spec_C=objective,
        query_id=QUERY_ID,
        available_memory_bytes=AVAILABLE_MEMORY_BYTES,
        memory_budget_bytes=AVAILABLE_MEMORY_BYTES,
        domain_batch_candidate_size=DOMAIN_BATCH_SIZE,
        max_domain_batch_size=DOMAIN_BATCH_SIZE,
    )
    full = compile_native_domain_batch_query(
        legacy_module,
        query_specs,
        linear_spec_C=objective,
        query_id=QUERY_ID,
        available_memory_bytes=AVAILABLE_MEMORY_BYTES,
        memory_budget_bytes=AVAILABLE_MEMORY_BYTES,
        domain_batch_candidate_size=DOMAIN_BATCH_SIZE,
        max_domain_batch_size=len(query_specs),
    )
    packed_results, packed_trace = execute_native_domain_batch_query(
        packed, legacy_task_module=legacy_module
    )
    full_results, full_trace = execute_native_domain_batch_query(
        full, legacy_task_module=legacy_module
    )
    serial_results, serial_trace = execute_native_domain_serial_reference(
        packed,
        legacy_task_module=legacy_module,
        available_memory_bytes=AVAILABLE_MEMORY_BYTES,
    )
    packed_lower, packed_upper = _stack_results(packed_results)
    full_lower, full_upper = _stack_results(full_results)
    serial_lower, serial_upper = _stack_results(serial_results)
    comparisons = {
        "packed_vs_full_lower": _comparison(packed_lower, full_lower),
        "packed_vs_full_upper": _comparison(packed_upper, full_upper),
        "packed_vs_serial_lower": _comparison(packed_lower, serial_lower),
        "packed_vs_serial_upper": _comparison(packed_upper, serial_upper),
    }
    if not all(item["allclose"] is True for item in comparisons.values()):
        raise ValueError("NRIR-8 packed/full/serial semantics differ")
    packed_binding = packed.binding_trace.to_dict()
    full_binding = full.binding_trace.to_dict()
    query_state_payload = [item.to_dict() for item in packed.binding_trace.query_states]
    packed_result_payload = [item.to_dict() for item in packed_results]
    full_result_payload = [item.to_dict() for item in full_results]
    serial_result_payload = [item.to_dict() for item in serial_results]
    source_hashes_same = (
        packed.hashes()["source_bound_module_hash"]
        == full.hashes()["source_bound_module_hash"]
        and packed.hashes()["source_plan_template_hash"]
        == full.hashes()["source_plan_template_hash"]
    )
    decisions_differ = (
        packed.hashes()["source_plan_instance_hash"]
        != full.hashes()["source_plan_instance_hash"]
        and packed.hashes()["source_schedule_hash"]
        != full.hashes()["source_schedule_hash"]
    )
    expected_ids = [f"{ROOT_QUERY_ID}:d03:n{index:04d}" for index in range(8)]
    expected_parents = [f"{ROOT_QUERY_ID}:d02:n{index // 2:04d}" for index in range(8)]
    gates = {
        "eight_distinct_leaf_boxes_have_exact_tree_lineage": bool(
            len(query_state_payload) == 8
            and [item["query_id"] for item in query_state_payload] == expected_ids
            and [item["parent_query_id"] for item in query_state_payload]
            == expected_parents
            and len({item["input_lower_hash"] for item in query_state_payload}) == 8
            and len({item["input_upper_hash"] for item in query_state_payload}) == 8
        ),
        "every_leaf_recomputes_distinct_exact_state": bool(
            len({item["exact_state_hash"] for item in query_state_payload}) == 8
            and all(
                item["exact_state_hash"] != item["parent_state_hash"]
                for item in query_state_payload
            )
        ),
        "parent_state_is_warm_start_only_and_never_exact": bool(
            all(
                item["parent_state_validity"] == PARENT_STATE_VALIDITY
                and item["parent_state_consumed_as_exact"] is False
                for item in query_state_payload
            )
            and packed_trace.parent_state_consumed_as_exact is False
            and full_trace.parent_state_consumed_as_exact is False
            and serial_trace.parent_state_consumed_as_exact is False
        ),
        "source_plan_selects_full_or_packed_domain_candidate": bool(
            source_hashes_same
            and decisions_differ
            and packed_binding["selected_batch_candidate_id"]
            == "batch:native-domain-sliced-v1:0004"
            and full_binding["selected_batch_candidate_id"] == "batch:full-query"
        ),
        "packed_two_children_replace_eight_serial_children": bool(
            packed_trace.packed_child_stack_count == 2
            and full_trace.packed_child_stack_count == 1
            and serial_trace.serial_child_stack_count == 8
        ),
        "packed_full_and_serial_restore_identical_results": bool(
            all(item["allclose"] is True for item in comparisons.values())
            and [item["query_id"] for item in packed_result_payload] == expected_ids
            and [item["query_id"] for item in full_result_payload] == expected_ids
            and [item["query_id"] for item in serial_result_payload] == expected_ids
        ),
        "all_paths_preserve_one_representation_and_storage_policy": bool(
            packed_binding["selected_representation_policy_id"] == DENSE_POLICY_ID
            and full_binding["selected_representation_policy_id"] == DENSE_POLICY_ID
            and serial_trace.representation_policy_id == DENSE_POLICY_ID
            and packed_binding["selected_storage_candidate_id"]
            == full_binding["selected_storage_candidate_id"]
            == serial_trace.storage_candidate_id
        ),
        "mechanism_counts_do_not_claim_performance_or_full_bab": True,
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-8 gates failed: {gates}")
    evidence: dict[str, object] = {
        "schema_version": DOMAIN_BATCH_EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "compiler_version": NATIVE_DOMAIN_BATCH_COMPILER_VERSION,
        "claim_boundary": (
            "real-network input-box leaf formation, exact child-state recomputation, "
            "domain-axis Plan/Schedule execution, lineage, and serial-reference "
            "correctness only; not full BaB, pruning, ReLU/β branching, latency, "
            "memory, CUDA, OOM, Pareto, or speedup evidence"
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
            "root_input_lower_hash": tensor_content_hash(tensors["input_lower"]),
            "root_input_upper_hash": tensor_content_hash(tensors["input_upper"]),
            "objective_hash": tensor_content_hash(objective),
            "objective_index": 0,
        },
        "domain_tree": {
            "root_query_id": ROOT_QUERY_ID,
            "split_depth": SPLIT_DEPTH,
            "leaf_count": len(query_specs),
            "query_specs": [item.to_dict() for item in query_specs],
            "query_states": query_state_payload,
        },
        "packed": {
            "ir_hashes": packed.hashes(),
            "binding_trace": packed_binding,
            "execution_trace": packed_trace.to_dict(),
            "results": packed_result_payload,
        },
        "full": {
            "ir_hashes": full.hashes(),
            "binding_trace": full_binding,
            "execution_trace": full_trace.to_dict(),
            "results": full_result_payload,
        },
        "serial": {
            "trace": serial_trace.to_dict(),
            "results": serial_result_payload,
        },
        "semantics": comparisons,
        "gates": gates,
        "limitations": [
            "input-box branching only; no ReLU split or beta state",
            "no BaB queue, bound-based pruning, termination, or verified property claim",
            "each child recomputes IBP state; parent state is warm-start-only metadata",
            "two versus eight child stacks is a mechanism count, not timing evidence",
            "CPU reference correctness only; no memory, CUDA, OOM, Pareto, or speedup claim",
        ],
    }
    validate_domain_batch_evidence(evidence)
    return evidence


def validate_domain_batch_evidence(evidence: Mapping[str, Any]) -> None:
    """Reject incomplete, relinked, or claim-inflated NRIR-8 evidence."""

    if (
        evidence.get("schema_version") != DOMAIN_BATCH_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("compiler_version") != NATIVE_DOMAIN_BATCH_COMPILER_VERSION
    ):
        raise ValueError("NRIR-8 evidence contract differs")
    claim = str(evidence.get("claim_boundary", ""))
    if any(
        phrase not in claim
        for phrase in (
            "input-box",
            "exact child-state",
            "not full BaB",
            "not full BaB, pruning",
            "speedup",
        )
    ):
        raise ValueError("NRIR-8 claim boundary omits hard limitations")
    gates = _mapping(evidence.get("gates"), "NRIR-8 gates")
    expected_gates = {
        "eight_distinct_leaf_boxes_have_exact_tree_lineage",
        "every_leaf_recomputes_distinct_exact_state",
        "parent_state_is_warm_start_only_and_never_exact",
        "source_plan_selects_full_or_packed_domain_candidate",
        "packed_two_children_replace_eight_serial_children",
        "packed_full_and_serial_restore_identical_results",
        "all_paths_preserve_one_representation_and_storage_policy",
        "mechanism_counts_do_not_claim_performance_or_full_bab",
    }
    if set(gates) != expected_gates or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("NRIR-8 gates are incomplete or failed")
    source = _mapping(evidence.get("source"), "NRIR-8 source")
    if (
        source.get("model_sha256") != MODEL_SHA256
        or source.get("vnnlib_sha256") != VNNLIB_SHA256
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("abcrown_commit") != ABCROWN_COMMIT
        or source.get("objective_index") != 0
        or any(
            len(str(source.get(name, ""))) != 64
            for name in (
                "native_ir_manifest_sha256",
                "native_ir_payload_sha256",
                "root_input_lower_hash",
                "root_input_upper_hash",
                "objective_hash",
            )
        )
    ):
        raise ValueError("NRIR-8 source identity differs")
    tree = _mapping(evidence.get("domain_tree"), "NRIR-8 domain tree")
    specs = _list(tree.get("query_specs"), "NRIR-8 query specs")
    states = _list(tree.get("query_states"), "NRIR-8 query states")
    if (
        tree.get("root_query_id") != ROOT_QUERY_ID
        or tree.get("split_depth") != 3
        or tree.get("leaf_count") != 8
        or len(specs) != 8
        or len(states) != 8
    ):
        raise ValueError("NRIR-8 domain tree size differs")
    expected_ids = [f"{ROOT_QUERY_ID}:d03:n{index:04d}" for index in range(8)]
    expected_parents = [f"{ROOT_QUERY_ID}:d02:n{index // 2:04d}" for index in range(8)]
    for index, (spec_value, state_value) in enumerate(zip(specs, states)):
        spec = _mapping(spec_value, "NRIR-8 query spec")
        state = _mapping(state_value, "NRIR-8 query state")
        if (
            spec.get("query_id") != expected_ids[index]
            or state.get("query_id") != expected_ids[index]
            or spec.get("parent_query_id") != expected_parents[index]
            or state.get("parent_query_id") != expected_parents[index]
            or spec.get("depth") != 3
            or spec.get("branch_ordinal") != index % 2
            or state.get("schema_version") != DOMAIN_QUERY_STATE_SCHEMA_VERSION
            or state.get("parent_state_validity") != PARENT_STATE_VALIDITY
            or state.get("parent_state_consumed_as_exact") is not False
            or state.get("exact_state_hash") == state.get("parent_state_hash")
            or any(
                len(str(state.get(name, ""))) != 64
                for name in (
                    "input_lower_hash",
                    "input_upper_hash",
                    "exact_state_hash",
                    "parent_state_hash",
                )
            )
        ):
            raise ValueError("NRIR-8 query lineage/state differs")
    packed = _mapping(evidence.get("packed"), "NRIR-8 packed path")
    full = _mapping(evidence.get("full"), "NRIR-8 full path")
    serial = _mapping(evidence.get("serial"), "NRIR-8 serial path")
    packed_binding = _mapping(packed.get("binding_trace"), "NRIR-8 packed binding")
    full_binding = _mapping(full.get("binding_trace"), "NRIR-8 full binding")
    packed_trace = _mapping(packed.get("execution_trace"), "NRIR-8 packed trace")
    full_trace = _mapping(full.get("execution_trace"), "NRIR-8 full trace")
    serial_trace = _mapping(serial.get("trace"), "NRIR-8 serial trace")
    if (
        packed_binding.get("schema_version") != DOMAIN_BATCH_BINDING_SCHEMA_VERSION
        or full_binding.get("schema_version") != DOMAIN_BATCH_BINDING_SCHEMA_VERSION
        or packed_binding.get("selected_batch_candidate_id")
        != "batch:native-domain-sliced-v1:0004"
        or packed_binding.get("selected_domain_batch_size") != 4
        or full_binding.get("selected_batch_candidate_id") != "batch:full-query"
        or full_binding.get("selected_domain_batch_size") != 8
        or packed_binding.get("total_domain_count") != 8
        or full_binding.get("total_domain_count") != 8
        or packed_binding.get("query_states") != states
        or full_binding.get("query_states") != states
        or packed_binding.get("selected_representation_policy_id") != DENSE_POLICY_ID
        or full_binding.get("selected_representation_policy_id") != DENSE_POLICY_ID
    ):
        raise ValueError("NRIR-8 Plan/binding policy differs")
    packed_slices = _list(packed_binding.get("slices"), "NRIR-8 packed slices")
    full_slices = _list(full_binding.get("slices"), "NRIR-8 full slices")
    if (
        len(packed_slices) != 2
        or len(full_slices) != 1
        or [
            (item.get("start_index"), item.get("stop_index"))
            for item in packed_slices
            if isinstance(item, Mapping)
        ]
        != [(0, 4), (4, 8)]
        or [
            (item.get("start_index"), item.get("stop_index"))
            for item in full_slices
            if isinstance(item, Mapping)
        ]
        != [(0, 8)]
    ):
        raise ValueError("NRIR-8 domain Schedule ranges differ")
    packed_results = _list(packed.get("results"), "NRIR-8 packed results")
    full_results = _list(full.get("results"), "NRIR-8 full results")
    serial_results = _list(serial.get("results"), "NRIR-8 serial results")
    if (
        packed_trace.get("schema_version")
        != DOMAIN_BATCH_EXECUTION_TRACE_SCHEMA_VERSION
        or full_trace.get("schema_version")
        != DOMAIN_BATCH_EXECUTION_TRACE_SCHEMA_VERSION
        or serial_trace.get("schema_version") != SERIAL_DOMAIN_TRACE_SCHEMA_VERSION
        or packed_trace.get("packed_child_stack_count") != 2
        or full_trace.get("packed_child_stack_count") != 1
        or serial_trace.get("serial_child_stack_count") != 8
        or packed_trace.get("parent_state_consumed_as_exact") is not False
        or full_trace.get("parent_state_consumed_as_exact") is not False
        or serial_trace.get("parent_state_consumed_as_exact") is not False
        or any(
            len(value) != 8 for value in (packed_results, full_results, serial_results)
        )
    ):
        raise ValueError("NRIR-8 execution accounting differs")
    for result_set in (packed_results, full_results, serial_results):
        for index, value in enumerate(result_set):
            result = _mapping(value, "NRIR-8 domain result")
            if (
                result.get("query_id") != expected_ids[index]
                or result.get("parent_query_id") != expected_parents[index]
                or result.get("shape") != [1, 1]
                or len(str(result.get("lower_hash", ""))) != 64
                or len(str(result.get("upper_hash", ""))) != 64
            ):
                raise ValueError("NRIR-8 restored result lineage differs")
    if (
        packed_trace.get("query_results") != packed_results
        or full_trace.get("query_results") != full_results
        or serial_trace.get("query_results") != serial_results
    ):
        raise ValueError("NRIR-8 execution/result linkage differs")
    semantics = _mapping(evidence.get("semantics"), "NRIR-8 semantics")
    if set(semantics) != {
        "packed_vs_full_lower",
        "packed_vs_full_upper",
        "packed_vs_serial_lower",
        "packed_vs_serial_upper",
    }:
        raise ValueError("NRIR-8 semantic comparison set differs")
    for value in semantics.values():
        comparison = _mapping(value, "NRIR-8 semantic comparison")
        if (
            comparison.get("allclose") is not True
            or comparison.get("shape") != [8, 1]
            or comparison.get("atol") != ATOL
            or comparison.get("rtol") != RTOL
        ):
            raise ValueError("NRIR-8 semantic comparison differs")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or not all(
        any(phrase in str(item) for item in limitations)
        for phrase in ("no ReLU split", "no BaB queue", "warm-start-only", "not timing")
    ):
        raise ValueError("NRIR-8 limitations are incomplete")


def generate_artifact(
    artifact_dir: Path, *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_domain_batch_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    (artifact_dir / "domain_batch.json").write_text(
        _canonical_json(evidence, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": DOMAIN_BATCH_ARTIFACT_SCHEMA_VERSION,
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
        manifest.get("schema_version") != DOMAIN_BATCH_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-8 manifest contract differs")
    files = _mapping(manifest.get("files"), "NRIR-8 manifest files")
    if set(files) != set(ARTIFACT_FILES):
        raise ValueError("NRIR-8 manifest file set differs")
    for name in ARTIFACT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError(f"NRIR-8 artifact digest differs: {name}")
    stored = _load_json(artifact_dir / "domain_batch.json")
    validate_domain_batch_evidence(stored)
    actual = build_domain_batch_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    if _canonical_json(stored) != _canonical_json(actual):
        raise ValueError("NRIR-8 semantic replay differs")
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
