#!/usr/bin/env python3
"""Generate or replay NRIR-5 real-network spec-sliced execution evidence."""

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
from boundflow.ir.schedule import BatchLoopAction, EmitResultAction, LaunchAction
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.abcrown_adapter import (
    bind_intermediate_bounds,
    deserialize_intermediate_bounds,
    file_sha256,
    intermediate_bounds_sha256,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_sliced_batch_integration import (
    NATIVE_SLICED_BATCH_COMPILER_VERSION,
    SPEC_BATCH_BINDING_SCHEMA_VERSION,
    SPEC_BATCH_EXECUTION_TRACE_SCHEMA_VERSION,
    compile_native_plain_crown_sliced_batch_query,
    execute_native_plain_crown_sliced_batch_query,
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

SLICED_BATCH_ARTIFACT_SCHEMA_VERSION = (
    "boundflow.native-real-network-sliced-batch-artifact/v1"
)
SLICED_BATCH_EVIDENCE_SCHEMA_VERSION = (
    "boundflow.native-real-network-sliced-batch-evidence/v1"
)
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir5-spec-batch"
AVAILABLE_MEMORY_BYTES = 1 << 30
SPEC_SLICE_SIZE = 3
ATOL = 2e-4
RTOL = 2e-4
ARTIFACT_FILES = ("sliced_batch.json",)


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


def build_sliced_batch_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    """Compile and execute full and 3-way spec-sliced native ResNet paths."""

    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-5 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    external_bounds = deserialize_intermediate_bounds(
        _mapping(payload.get("external_intermediate_bounds"), "intermediate bounds")
    )
    if intermediate_bounds_sha256(external_bounds) != INTERMEDIATE_BOUNDS_SHA256:
        raise ValueError("NRIR-5 intermediate-bound digest differs")
    linear_spec = tensors["linear_spec_c"]
    total_specs = int(linear_spec.shape[-2])
    if total_specs != 9:
        raise ValueError("NRIR-5 frozen objective count differs")

    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    legacy_module = plan_interval_ibp_v0(program)
    primal_ops = tuple(op.op_type for op in legacy_module.get_entry_task().ops)
    if primal_ops != EXPECTED_PRIMAL_OPS:
        raise ValueError("NRIR-5 primal topology differs")
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    interval_env, local_relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    relu_pre = bind_intermediate_bounds(external_bounds, local_relu_pre)

    def compile_with_limit(max_spec_batch_size: int):
        return compile_native_plain_crown_sliced_batch_query(
            legacy_module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
            intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
            query_id=QUERY_ID,
            available_memory_bytes=AVAILABLE_MEMORY_BYTES,
            memory_budget_bytes=AVAILABLE_MEMORY_BYTES,
            spec_slice_candidate_size=SPEC_SLICE_SIZE,
            max_spec_batch_size=max_spec_batch_size,
        )

    full = compile_with_limit(total_specs)
    sliced = compile_with_limit(SPEC_SLICE_SIZE)
    full_result, full_execution_trace = execute_native_plain_crown_sliced_batch_query(
        full,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
    )
    sliced_result, sliced_execution_trace = (
        execute_native_plain_crown_sliced_batch_query(
            sliced,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
        )
    )
    expected = tensors["external_lower"].to(full_result.lower)
    full_vs_sliced = _comparison(full_result.lower, sliced_result.lower)
    full_vs_external = _comparison(full_result.lower, expected)
    sliced_vs_external = _comparison(sliced_result.lower, expected)
    if not all(
        comparison["allclose"] is True
        and comparison["sign_agreement"] == comparison["sign_total"] == 9
        for comparison in (
            full_vs_sliced,
            full_vs_external,
            sliced_vs_external,
        )
    ):
        raise ValueError("NRIR-5 full/sliced semantics differ")
    torch.testing.assert_close(
        full_result.upper, sliced_result.upper, atol=ATOL, rtol=RTOL
    )

    full_loop = _batch_loop(full.source_schedule)
    sliced_loop = _batch_loop(sliced.source_schedule)
    full_hashes = full.hashes()
    sliced_hashes = sliced.hashes()
    slice_ranges = [
        [item.start_index, item.stop_index] for item in sliced.binding_trace.slices
    ]
    sliced_child_task_count = sum(
        len(child.task_module.tasks) for child in sliced.child_compilations
    )
    sliced_child_launch_count = sum(
        sum(isinstance(action, LaunchAction) for action in child.schedule.actions)
        for child in sliced.child_compilations
    )
    gates = {
        "same_source_bound_and_template": bool(
            full_hashes["source_bound_module_hash"]
            == sliced_hashes["source_bound_module_hash"]
            and full_hashes["source_plan_template_hash"]
            == sliced_hashes["source_plan_template_hash"]
        ),
        "runtime_limit_switches_batch_decision": bool(
            full.source_instance.batch_decision.candidate_id == "batch:full-query"
            and sliced.source_instance.batch_decision.candidate_id
            == "batch:native-spec-sliced-v1:0003"
        ),
        "source_schedule_owns_exact_spec_ranges": bool(
            full_loop.axis == "domain"
            and sliced_loop.axis == "spec"
            and slice_ranges == [[0, 3], [3, 6], [6, 9]]
            and all(item.query_ids == (QUERY_ID,) for item in sliced_loop.slices)
        ),
        "each_slice_has_distinct_native_ir_stack": bool(
            len(sliced.child_compilations) == 3
            and len(
                {
                    child.bound_module.stable_hash()
                    for child in sliced.child_compilations
                }
            )
            == 3
        ),
        "all_child_bound_ops_are_tasked_and_launched": bool(
            sliced_child_task_count == sliced_child_launch_count == 63
            and all(
                len(child.task_module.tasks) == len(child.bound_module.graph.ops) == 21
                for child in sliced.child_compilations
            )
        ),
        "source_and_child_query_accounting_is_exact": bool(
            _emitted_query_ids(sliced.source_schedule) == (QUERY_ID,)
            and sliced_execution_trace.child_query_ids
            == tuple(item.child_query_id for item in sliced.binding_trace.slices)
            and len(set(sliced_execution_trace.child_query_ids)) == 3
        ),
        "full_and_sliced_match_external_semantics": bool(
            full_vs_sliced["allclose"]
            and full_vs_external["allclose"]
            and sliced_vs_external["allclose"]
        ),
        "controller_storage_is_not_claimed_as_slice_memory": bool(
            full.source_instance.cost_summary.predicted_peak_bytes
            == sliced.source_instance.cost_summary.predicted_peak_bytes
            and full.source_instance.storage_decision.candidate_id
            == sliced.source_instance.storage_decision.candidate_id
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-5 gates failed: {gates}")

    evidence: dict[str, object] = {
        "schema_version": SLICED_BATCH_EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "compiler_version": NATIVE_SLICED_BATCH_COMPILER_VERSION,
        "claim_boundary": (
            "real-network spec-axis Plan/Schedule slicing, child compiler-stack "
            "ownership, aggregation, and CPU correctness only; source controller "
            "storage is unchanged and supports no memory, latency, CUDA, OOM, "
            "Pareto, or speedup claim"
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
            "linear_spec_sha256": tensor_content_hash(linear_spec),
        },
        "full_policy": _policy_summary(full, full_execution_trace.to_dict()),
        "sliced_policy": {
            **_policy_summary(sliced, sliced_execution_trace.to_dict()),
            "slice_ranges": slice_ranges,
            "binding_trace": sliced.binding_trace.to_dict(),
            "child_task_count": sliced_child_task_count,
            "child_launch_count": sliced_child_launch_count,
        },
        "semantics": {
            "full_vs_sliced_lower": full_vs_sliced,
            "full_vs_external_lower": full_vs_external,
            "sliced_vs_external_lower": sliced_vs_external,
            "full_lower_sha256": tensor_content_hash(full_result.lower),
            "sliced_lower_sha256": tensor_content_hash(sliced_result.lower),
            "full_upper_sha256": tensor_content_hash(full_result.upper),
            "sliced_upper_sha256": tensor_content_hash(sliced_result.upper),
        },
        "gates": gates,
        "limitations": [
            "CPU correctness and compiler/schedule ownership evidence only",
            "v1 slices the spec axis; domain and sample axes remain independent pending work",
            "each slice is a static child compiler stack executed sequentially",
            "source controller storage remains the full-query ledger for both policies",
            "no memory reduction, latency, CUDA, allocator, OOM, Pareto, or speedup claim",
        ],
    }
    validate_sliced_batch_evidence(evidence)
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


def _batch_loop(schedule):
    loops = tuple(
        action for action in schedule.actions if isinstance(action, BatchLoopAction)
    )
    if len(loops) != 1:
        raise ValueError("NRIR-5 source Schedule batch-loop count differs")
    return loops[0]


def _emitted_query_ids(schedule) -> tuple[str, ...]:
    emits = tuple(
        action for action in schedule.actions if isinstance(action, EmitResultAction)
    )
    if len(emits) != 1:
        raise ValueError("NRIR-5 source Schedule emit count differs")
    return emits[0].query_ids


def _policy_summary(compilation, execution_trace: Mapping[str, Any]):
    loop = _batch_loop(compilation.source_schedule)
    return {
        "batch_candidate_id": compilation.source_instance.batch_decision.candidate_id,
        "selected_spec_batch_size": compilation.binding_trace.selected_spec_batch_size,
        "source_ir_hashes": compilation.hashes(),
        "source_schedule_axis": loop.axis,
        "source_schedule_slice_count": len(loop.slices),
        "child_stack_count": len(compilation.child_compilations),
        "child_ir_hashes": [child.hashes() for child in compilation.child_compilations],
        "execution_trace": dict(execution_trace),
        "controller_storage_candidate_id": (
            compilation.source_instance.storage_decision.candidate_id
        ),
        "controller_peak_bytes": (
            compilation.source_instance.cost_summary.predicted_peak_bytes
        ),
    }


def validate_sliced_batch_evidence(evidence: Mapping[str, Any]) -> None:
    """Reject incomplete, relinked, or claim-inflated NRIR-5 evidence."""

    if (
        evidence.get("schema_version") != SLICED_BATCH_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("compiler_version") != NATIVE_SLICED_BATCH_COMPILER_VERSION
    ):
        raise ValueError("NRIR-5 evidence contract differs")
    claim = str(evidence.get("claim_boundary", ""))
    if any(
        phrase not in claim for phrase in ("spec-axis", "CPU correctness", "no memory")
    ):
        raise ValueError("NRIR-5 claim boundary omits hard limitations")
    gates = _mapping(evidence.get("gates"), "NRIR-5 gates")
    expected_gates = {
        "same_source_bound_and_template",
        "runtime_limit_switches_batch_decision",
        "source_schedule_owns_exact_spec_ranges",
        "each_slice_has_distinct_native_ir_stack",
        "all_child_bound_ops_are_tasked_and_launched",
        "source_and_child_query_accounting_is_exact",
        "full_and_sliced_match_external_semantics",
        "controller_storage_is_not_claimed_as_slice_memory",
    }
    if set(gates) != expected_gates or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("NRIR-5 gates are incomplete or failed")
    full = _mapping(evidence.get("full_policy"), "NRIR-5 full policy")
    sliced = _mapping(evidence.get("sliced_policy"), "NRIR-5 sliced policy")
    if (
        full.get("batch_candidate_id") != "batch:full-query"
        or sliced.get("batch_candidate_id") != "batch:native-spec-sliced-v1:0003"
        or full.get("source_schedule_axis") != "domain"
        or sliced.get("source_schedule_axis") != "spec"
        or int(full.get("child_stack_count", 0)) != 1
        or int(sliced.get("child_stack_count", 0)) != 3
        or sliced.get("slice_ranges") != [[0, 3], [3, 6], [6, 9]]
        or full.get("controller_storage_candidate_id")
        != sliced.get("controller_storage_candidate_id")
        or full.get("controller_peak_bytes") != sliced.get("controller_peak_bytes")
    ):
        raise ValueError("NRIR-5 policy selection/accounting differs")
    full_hashes = _validate_source_hashes(
        _mapping(full.get("source_ir_hashes"), "NRIR-5 full hashes")
    )
    sliced_hashes = _validate_source_hashes(
        _mapping(sliced.get("source_ir_hashes"), "NRIR-5 sliced hashes")
    )
    if (
        full_hashes["source_bound_module_hash"]
        != sliced_hashes["source_bound_module_hash"]
        or full_hashes["source_plan_template_hash"]
        != sliced_hashes["source_plan_template_hash"]
        or full_hashes["source_plan_instance_hash"]
        == sliced_hashes["source_plan_instance_hash"]
        or full_hashes["source_schedule_hash"] == sliced_hashes["source_schedule_hash"]
    ):
        raise ValueError("NRIR-5 source IR linkage differs")
    binding = _mapping(sliced.get("binding_trace"), "NRIR-5 binding trace")
    slices = binding.get("slices")
    if (
        binding.get("source_bound_module_hash")
        != sliced_hashes["source_bound_module_hash"]
        or binding.get("source_plan_template_hash")
        != sliced_hashes["source_plan_template_hash"]
        or binding.get("source_plan_instance_hash")
        != sliced_hashes["source_plan_instance_hash"]
        or binding.get("source_schedule_hash") != sliced_hashes["source_schedule_hash"]
        or not isinstance(slices, list)
        or len(slices) != 3
        or binding.get("schema_version") != SPEC_BATCH_BINDING_SCHEMA_VERSION
        or binding.get("selected_batch_candidate_id")
        != sliced.get("batch_candidate_id")
        or binding.get("selected_spec_batch_size") != 3
        or binding.get("total_spec_count") != 9
        or hashlib.sha256(_canonical_json(binding).encode("utf-8")).hexdigest()
        != sliced_hashes["spec_batch_binding_hash"]
    ):
        raise ValueError("NRIR-5 binding trace linkage differs")
    child_hashes = sliced.get("child_ir_hashes")
    if not isinstance(child_hashes, list) or len(child_hashes) != 3:
        raise ValueError("NRIR-5 child compiler stacks are incomplete")
    expected_ranges = ((0, 3), (3, 6), (6, 9))
    expected_query_ids: list[str] = []
    for position, (item, hashes, expected_range) in enumerate(
        zip(slices, child_hashes, expected_ranges)
    ):
        binding_slice = _mapping(item, "NRIR-5 binding slice")
        child = _mapping(hashes, "NRIR-5 child hashes")
        expected_start, expected_stop = expected_range
        expected_query_id = f"{QUERY_ID}:spec:{expected_start:04d}:{expected_stop:04d}"
        expected_query_ids.append(expected_query_id)
        if (
            binding_slice.get("slice_id")
            != f"spec-slice:{expected_start:04d}:{expected_stop:04d}"
            or binding_slice.get("start_index") != expected_start
            or binding_slice.get("stop_index") != expected_stop
            or binding_slice.get("child_query_id") != expected_query_id
            or position != expected_start // SPEC_SLICE_SIZE
        ):
            raise ValueError("NRIR-5 binding slice ranges/query ownership differs")
        if set(child) != {
            "bound_module_hash",
            "plan_template_hash",
            "plan_instance_hash",
            "task_module_hash",
            "schedule_hash",
        } or any(len(str(value)) != 64 for value in child.values()):
            raise ValueError("NRIR-5 child IR hash set differs")
        for binding_key, child_key in (
            ("child_bound_module_hash", "bound_module_hash"),
            ("child_plan_template_hash", "plan_template_hash"),
            ("child_plan_instance_hash", "plan_instance_hash"),
            ("child_task_module_hash", "task_module_hash"),
            ("child_schedule_hash", "schedule_hash"),
        ):
            if binding_slice.get(binding_key) != child.get(child_key):
                raise ValueError("NRIR-5 binding/child IR hash differs")
    if (
        int(sliced.get("child_task_count", 0)) != 63
        or int(sliced.get("child_launch_count", 0)) != 63
    ):
        raise ValueError("NRIR-5 child Task/Launch ownership differs")
    semantics = _mapping(evidence.get("semantics"), "NRIR-5 semantics")
    full_trace = _mapping(full.get("execution_trace"), "NRIR-5 full execution trace")
    sliced_trace = _mapping(
        sliced.get("execution_trace"), "NRIR-5 sliced execution trace"
    )
    _validate_execution_trace(
        full_trace,
        expected_binding_hash=full_hashes["spec_batch_binding_hash"],
        expected_query_ids=(f"{QUERY_ID}:spec:0000:0009",),
        expected_lower_hash=str(semantics.get("full_lower_sha256", "")),
        expected_upper_hash=str(semantics.get("full_upper_sha256", "")),
    )
    _validate_execution_trace(
        sliced_trace,
        expected_binding_hash=sliced_hashes["spec_batch_binding_hash"],
        expected_query_ids=tuple(expected_query_ids),
        expected_lower_hash=str(semantics.get("sliced_lower_sha256", "")),
        expected_upper_hash=str(semantics.get("sliced_upper_sha256", "")),
    )
    for name in (
        "full_vs_sliced_lower",
        "full_vs_external_lower",
        "sliced_vs_external_lower",
    ):
        comparison = _mapping(semantics.get(name), name)
        if (
            comparison.get("allclose") is not True
            or comparison.get("sign_agreement") != 9
            or comparison.get("sign_total") != 9
        ):
            raise ValueError("NRIR-5 semantic comparison differs")


def _validate_execution_trace(
    trace: Mapping[str, Any],
    *,
    expected_binding_hash: str,
    expected_query_ids: tuple[str, ...],
    expected_lower_hash: str,
    expected_upper_hash: str,
) -> None:
    task_hashes = trace.get("child_task_trace_hashes")
    if (
        trace.get("schema_version") != SPEC_BATCH_EXECUTION_TRACE_SCHEMA_VERSION
        or trace.get("binding_hash") != expected_binding_hash
        or trace.get("child_query_ids") != list(expected_query_ids)
        or not isinstance(task_hashes, list)
        or len(task_hashes) != len(expected_query_ids)
        or any(len(str(value)) != 64 for value in task_hashes)
        or trace.get("result_lower_hash") != expected_lower_hash
        or trace.get("result_upper_hash") != expected_upper_hash
        or len(expected_lower_hash) != 64
        or len(expected_upper_hash) != 64
    ):
        raise ValueError("NRIR-5 execution trace linkage differs")


def _validate_source_hashes(values: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "source_bound_module_hash",
        "source_plan_template_hash",
        "source_plan_instance_hash",
        "source_schedule_hash",
        "spec_batch_binding_hash",
    }
    if set(values) != expected or any(
        len(str(value)) != 64 for value in values.values()
    ):
        raise ValueError("NRIR-5 source hash set differs")
    return values


def generate_artifact(
    artifact_dir: Path, *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_sliced_batch_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    (artifact_dir / "sliced_batch.json").write_text(
        _canonical_json(evidence, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SLICED_BATCH_ARTIFACT_SCHEMA_VERSION,
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
        manifest.get("schema_version") != SLICED_BATCH_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-5 manifest contract differs")
    files = _mapping(manifest.get("files"), "NRIR-5 manifest files")
    if set(files) != set(ARTIFACT_FILES):
        raise ValueError("NRIR-5 manifest file set differs")
    for name in ARTIFACT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError(f"NRIR-5 artifact digest differs: {name}")
    stored = _load_json(artifact_dir / "sliced_batch.json")
    validate_sliced_batch_evidence(stored)
    actual = build_sliced_batch_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    if _canonical_json(stored) != _canonical_json(actual):
        raise ValueError("NRIR-5 semantic replay differs")
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
