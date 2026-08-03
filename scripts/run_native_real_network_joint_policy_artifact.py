#!/usr/bin/env python3
"""Generate or replay NRIR-6 representation × spec-batch evidence."""

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
from boundflow.ir.schedule import BatchLoopAction, LaunchAction
from boundflow.planner import plan_interval_ibp_v0
from boundflow.planner.representation_plan_binding import (
    DENSE_POLICY_ID,
    STRUCTURED_AFFINE_POLICY_ID,
)
from boundflow.runtime.abcrown_adapter import (
    bind_intermediate_bounds,
    deserialize_intermediate_bounds,
    file_sha256,
    intermediate_bounds_sha256,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_joint_policy_integration import (
    JOINT_POLICY_BINDING_SCHEMA_VERSION,
    JOINT_POLICY_EXECUTION_TRACE_SCHEMA_VERSION,
    NATIVE_JOINT_POLICY_COMPILER_VERSION,
    NativePlainCrownJointPolicyCompilation,
    compile_native_plain_crown_joint_policy_query,
    execute_native_plain_crown_joint_policy_query,
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

JOINT_POLICY_ARTIFACT_SCHEMA_VERSION = (
    "boundflow.native-real-network-joint-policy-artifact/v1"
)
JOINT_POLICY_EVIDENCE_SCHEMA_VERSION = (
    "boundflow.native-real-network-joint-policy-evidence/v1"
)
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir6-joint-policy"
AVAILABLE_MEMORY_BYTES = 1 << 30
SPEC_SLICE_SIZE = 3
ATOL = 2e-4
RTOL = 2e-4
ARTIFACT_FILES = ("joint_policy.json",)
POLICY_ORDER = (
    "dense_full",
    "dense_sliced",
    "structured_full",
    "structured_sliced",
)


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


def build_joint_policy_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    """Compile and execute the four real-network joint policy combinations."""

    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-6 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    external_bounds = deserialize_intermediate_bounds(
        _mapping(payload.get("external_intermediate_bounds"), "intermediate bounds")
    )
    if intermediate_bounds_sha256(external_bounds) != INTERMEDIATE_BOUNDS_SHA256:
        raise ValueError("NRIR-6 intermediate-bound digest differs")
    linear_spec = tensors["linear_spec_c"]
    total_specs = int(linear_spec.shape[-2])
    if total_specs != 9:
        raise ValueError("NRIR-6 frozen objective count differs")

    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    legacy_module = plan_interval_ibp_v0(program)
    if tuple(op.op_type for op in legacy_module.get_entry_task().ops) != (
        EXPECTED_PRIMAL_OPS
    ):
        raise ValueError("NRIR-6 primal topology differs")
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    interval_env, local_relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    relu_pre = bind_intermediate_bounds(external_bounds, local_relu_pre)

    def compile_policy(
        memory_budget_bytes: int, max_spec_batch_size: int
    ) -> NativePlainCrownJointPolicyCompilation:
        return compile_native_plain_crown_joint_policy_query(
            legacy_module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
            intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
            query_id=QUERY_ID,
            available_memory_bytes=AVAILABLE_MEMORY_BYTES,
            memory_budget_bytes=memory_budget_bytes,
            spec_slice_candidate_size=SPEC_SLICE_SIZE,
            max_spec_batch_size=max_spec_batch_size,
        )

    dense_full = compile_policy(AVAILABLE_MEMORY_BYTES, total_specs)
    storage = {
        item.candidate_id: item
        for item in dense_full.source_template.storage_candidates
    }
    reuse_budget = storage["storage:native-lifetime-reuse-v1"].cost.predicted_peak_bytes
    compilations = {
        "dense_full": dense_full,
        "dense_sliced": compile_policy(AVAILABLE_MEMORY_BYTES, SPEC_SLICE_SIZE),
        "structured_full": compile_policy(reuse_budget, total_specs),
        "structured_sliced": compile_policy(reuse_budget, SPEC_SLICE_SIZE),
    }
    results = {}
    execution_traces = {}
    for name in POLICY_ORDER:
        result, execution_trace = execute_native_plain_crown_joint_policy_query(
            compilations[name],
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
        )
        results[name] = result
        execution_traces[name] = execution_trace

    expected = tensors["external_lower"].to(results["dense_full"].lower)
    semantics: dict[str, object] = {}
    for name in POLICY_ORDER:
        semantics[f"{name}_vs_external_lower"] = _comparison(
            results[name].lower, expected
        )
        semantics[f"{name}_lower_sha256"] = tensor_content_hash(results[name].lower)
        semantics[f"{name}_upper_sha256"] = tensor_content_hash(results[name].upper)
        if name != "dense_full":
            semantics[f"dense_full_vs_{name}_lower"] = _comparison(
                results["dense_full"].lower, results[name].lower
            )
            torch.testing.assert_close(
                results["dense_full"].upper,
                results[name].upper,
                atol=ATOL,
                rtol=RTOL,
            )

    summaries = {
        name: _policy_summary(compilations[name], execution_traces[name].to_dict())
        for name in POLICY_ORDER
    }
    source_hashes = [
        compilations[name].hashes()["source_bound_module_hash"] for name in POLICY_ORDER
    ]
    template_hashes = [
        compilations[name].hashes()["source_plan_template_hash"]
        for name in POLICY_ORDER
    ]
    instance_hashes = [
        compilations[name].hashes()["source_plan_instance_hash"]
        for name in POLICY_ORDER
    ]
    schedule_hashes = [
        compilations[name].hashes()["source_schedule_hash"] for name in POLICY_ORDER
    ]
    expected_decisions = {
        "dense_full": (DENSE_POLICY_ID, "batch:full-query", 1),
        "dense_sliced": (
            DENSE_POLICY_ID,
            "batch:native-spec-sliced-v1:0003",
            3,
        ),
        "structured_full": (
            STRUCTURED_AFFINE_POLICY_ID,
            "batch:full-query",
            1,
        ),
        "structured_sliced": (
            STRUCTURED_AFFINE_POLICY_ID,
            "batch:native-spec-sliced-v1:0003",
            3,
        ),
    }
    gates = {
        "one_source_bound_and_template": bool(
            len(set(source_hashes)) == len(set(template_hashes)) == 1
        ),
        "four_joint_plan_and_schedule_identities": bool(
            len(set(instance_hashes)) == len(set(schedule_hashes)) == 4
        ),
        "budget_and_spec_limit_select_exact_cross_product": all(
            (
                summaries[name]["representation_policy_id"],
                summaries[name]["batch_candidate_id"],
                summaries[name]["child_stack_count"],
            )
            == expected_decisions[name]
            for name in POLICY_ORDER
        ),
        "children_inherit_source_representation_policy": all(
            summaries[name]["child_representation_policy_ids"]
            == [summaries[name]["representation_policy_id"]]
            for name in POLICY_ORDER
        ),
        "sliced_schedules_own_exact_spec_ranges": bool(
            summaries["dense_sliced"]["slice_ranges"]
            == summaries["structured_sliced"]["slice_ranges"]
            == [[0, 3], [3, 6], [6, 9]]
        ),
        "structured_transitions_remain_execution_owned": bool(
            summaries["structured_full"]["source_transition_count"]
            == summaries["structured_sliced"]["source_transition_count"]
            == 28
            and summaries["structured_full"]["source_execution_op_count"]
            == summaries["structured_sliced"]["source_execution_op_count"]
            == 49
        ),
        "all_child_ops_are_tasked_and_launched": bool(
            [summaries[name]["child_op_count"] for name in POLICY_ORDER]
            == [21, 63, 49, 147]
            and all(
                summaries[name]["child_op_count"]
                == summaries[name]["child_task_count"]
                == summaries[name]["child_launch_count"]
                for name in POLICY_ORDER
            )
        ),
        "four_paths_match_external_semantics": all(
            _mapping(
                semantics[f"{name}_vs_external_lower"],
                f"{name} semantics",
            ).get("allclose")
            is True
            for name in POLICY_ORDER
        ),
        "no_joint_performance_or_memory_claim": bool(
            summaries["dense_full"]["controller_peak_bytes"]
            == summaries["dense_sliced"]["controller_peak_bytes"]
            == 1_860_912
            and summaries["structured_full"]["controller_peak_bytes"]
            == summaries["structured_sliced"]["controller_peak_bytes"]
            == 442_656
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-6 gates failed: {gates}")

    evidence: dict[str, object] = {
        "schema_version": JOINT_POLICY_EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "compiler_version": NATIVE_JOINT_POLICY_COMPILER_VERSION,
        "claim_boundary": (
            "real-network representation × spec-batch joint Plan selection and "
            "execution ownership under CPU floating-point semantics only; no "
            "memory, latency, CUDA, OOM, Pareto, or speedup claim"
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
        "reuse_budget_bytes": reuse_budget,
        "policies": summaries,
        "semantics": semantics,
        "gates": gates,
        "limitations": [
            "CPU correctness and cross-axis compiler ownership only",
            "spec slices execute sequentially; this is not a throughput result",
            "structured operators and storage remain dense-equivalent",
            "domain/sample and cross-query physical batching remain pending",
            "no memory, latency, CUDA, allocator, OOM, Pareto, or speedup claim",
        ],
    }
    validate_joint_policy_evidence(evidence)
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


def _policy_summary(
    compilation: NativePlainCrownJointPolicyCompilation,
    execution_trace: Mapping[str, Any],
) -> dict[str, object]:
    loop = next(
        action
        for action in compilation.source_schedule.actions
        if isinstance(action, BatchLoopAction)
    )
    slice_ranges = [
        [item.start_index, item.stop_index] for item in compilation.binding_trace.slices
    ]
    child_ops = sum(
        len(child.bound_module.graph.ops) for child in compilation.child_compilations
    )
    child_tasks = sum(
        len(child.task_module.tasks) for child in compilation.child_compilations
    )
    child_launches = sum(
        sum(isinstance(action, LaunchAction) for action in child.schedule.actions)
        for child in compilation.child_compilations
    )
    return {
        "representation_policy_id": (
            compilation.binding_trace.selected_representation_policy_id
        ),
        "storage_candidate_id": (
            compilation.source_instance.storage_decision.candidate_id
        ),
        "batch_candidate_id": compilation.source_instance.batch_decision.candidate_id,
        "selected_spec_batch_size": (
            compilation.binding_trace.selected_spec_batch_size
        ),
        "source_ir_hashes": compilation.hashes(),
        "source_schedule_axis": loop.axis,
        "slice_ranges": slice_ranges,
        "source_transition_count": len(
            compilation.source_representation_binding.trace.events
        ),
        "source_execution_op_count": len(
            compilation.source_representation_binding.execution_bound_module.graph.ops
        ),
        "child_stack_count": len(compilation.child_compilations),
        "child_op_count": child_ops,
        "child_task_count": child_tasks,
        "child_launch_count": child_launches,
        "child_representation_policy_ids": sorted(
            {child.binding.trace.policy_id for child in compilation.child_compilations}
        ),
        "child_ir_hashes": [child.hashes() for child in compilation.child_compilations],
        "binding_trace": compilation.binding_trace.to_dict(),
        "execution_trace": dict(execution_trace),
        "controller_peak_bytes": (
            compilation.source_instance.cost_summary.predicted_peak_bytes
        ),
    }


def validate_joint_policy_evidence(evidence: Mapping[str, Any]) -> None:
    """Reject incomplete, relinked, or claim-inflated NRIR-6 evidence."""

    if (
        evidence.get("schema_version") != JOINT_POLICY_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("compiler_version") != NATIVE_JOINT_POLICY_COMPILER_VERSION
    ):
        raise ValueError("NRIR-6 evidence contract differs")
    claim = str(evidence.get("claim_boundary", ""))
    if any(
        phrase not in claim for phrase in ("joint Plan", "CPU", "no memory", "speedup")
    ):
        raise ValueError("NRIR-6 claim boundary omits hard limitations")
    gates = _mapping(evidence.get("gates"), "NRIR-6 gates")
    expected_gates = {
        "one_source_bound_and_template",
        "four_joint_plan_and_schedule_identities",
        "budget_and_spec_limit_select_exact_cross_product",
        "children_inherit_source_representation_policy",
        "sliced_schedules_own_exact_spec_ranges",
        "structured_transitions_remain_execution_owned",
        "all_child_ops_are_tasked_and_launched",
        "four_paths_match_external_semantics",
        "no_joint_performance_or_memory_claim",
    }
    if set(gates) != expected_gates or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("NRIR-6 gates are incomplete or failed")
    if evidence.get("reuse_budget_bytes") != 442_656:
        raise ValueError("NRIR-6 reuse budget differs")
    policies = _mapping(evidence.get("policies"), "NRIR-6 policies")
    if tuple(policies) != POLICY_ORDER:
        raise ValueError("NRIR-6 policy order/set differs")
    expected = {
        "dense_full": {
            "representation": DENSE_POLICY_ID,
            "storage": "storage:native-retain-all-v1",
            "batch": "batch:full-query",
            "axis": "domain",
            "ranges": [[0, 9]],
            "children": 1,
            "transitions": 0,
            "source_ops": 21,
            "child_ops": 21,
            "peak": 1_860_912,
        },
        "dense_sliced": {
            "representation": DENSE_POLICY_ID,
            "storage": "storage:native-retain-all-v1",
            "batch": "batch:native-spec-sliced-v1:0003",
            "axis": "spec",
            "ranges": [[0, 3], [3, 6], [6, 9]],
            "children": 3,
            "transitions": 0,
            "source_ops": 21,
            "child_ops": 63,
            "peak": 1_860_912,
        },
        "structured_full": {
            "representation": STRUCTURED_AFFINE_POLICY_ID,
            "storage": "storage:native-lifetime-reuse-v1",
            "batch": "batch:full-query",
            "axis": "domain",
            "ranges": [[0, 9]],
            "children": 1,
            "transitions": 28,
            "source_ops": 49,
            "child_ops": 49,
            "peak": 442_656,
        },
        "structured_sliced": {
            "representation": STRUCTURED_AFFINE_POLICY_ID,
            "storage": "storage:native-lifetime-reuse-v1",
            "batch": "batch:native-spec-sliced-v1:0003",
            "axis": "spec",
            "ranges": [[0, 3], [3, 6], [6, 9]],
            "children": 3,
            "transitions": 28,
            "source_ops": 49,
            "child_ops": 147,
            "peak": 442_656,
        },
    }
    source_bound_hashes: set[str] = set()
    source_template_hashes: set[str] = set()
    source_instance_hashes: set[str] = set()
    source_schedule_hashes: set[str] = set()
    semantics = _mapping(evidence.get("semantics"), "NRIR-6 semantics")
    for name in POLICY_ORDER:
        policy = _mapping(policies[name], f"NRIR-6 {name}")
        spec = expected[name]
        if (
            policy.get("representation_policy_id") != spec["representation"]
            or policy.get("storage_candidate_id") != spec["storage"]
            or policy.get("batch_candidate_id") != spec["batch"]
            or policy.get("source_schedule_axis") != spec["axis"]
            or policy.get("slice_ranges") != spec["ranges"]
            or policy.get("child_stack_count") != spec["children"]
            or policy.get("source_transition_count") != spec["transitions"]
            or policy.get("source_execution_op_count") != spec["source_ops"]
            or policy.get("child_op_count") != spec["child_ops"]
            or policy.get("child_task_count") != spec["child_ops"]
            or policy.get("child_launch_count") != spec["child_ops"]
            or policy.get("controller_peak_bytes") != spec["peak"]
            or policy.get("child_representation_policy_ids") != [spec["representation"]]
        ):
            raise ValueError("NRIR-6 joint policy decision/accounting differs")
        hashes = _mapping(policy.get("source_ir_hashes"), f"NRIR-6 {name} hashes")
        expected_hash_keys = {
            "source_bound_module_hash",
            "source_plan_template_hash",
            "source_plan_instance_hash",
            "source_schedule_hash",
            "source_representation_binding_hash",
            "joint_policy_binding_hash",
        }
        if set(hashes) != expected_hash_keys or any(
            len(str(value)) != 64 for value in hashes.values()
        ):
            raise ValueError("NRIR-6 source hash set differs")
        source_bound_hashes.add(str(hashes["source_bound_module_hash"]))
        source_template_hashes.add(str(hashes["source_plan_template_hash"]))
        source_instance_hashes.add(str(hashes["source_plan_instance_hash"]))
        source_schedule_hashes.add(str(hashes["source_schedule_hash"]))
        binding = _mapping(policy.get("binding_trace"), f"NRIR-6 {name} binding")
        if (
            binding.get("schema_version") != JOINT_POLICY_BINDING_SCHEMA_VERSION
            or binding.get("source_bound_module_hash")
            != hashes["source_bound_module_hash"]
            or binding.get("source_plan_template_hash")
            != hashes["source_plan_template_hash"]
            or binding.get("source_plan_instance_hash")
            != hashes["source_plan_instance_hash"]
            or binding.get("source_schedule_hash") != hashes["source_schedule_hash"]
            or binding.get("selected_representation_policy_id")
            != spec["representation"]
            or binding.get("selected_storage_candidate_id") != spec["storage"]
            or binding.get("selected_batch_candidate_id") != spec["batch"]
            or hashlib.sha256(_canonical_json(binding).encode("utf-8")).hexdigest()
            != hashes["joint_policy_binding_hash"]
        ):
            raise ValueError("NRIR-6 joint binding linkage differs")
        slices = binding.get("slices")
        child_hashes = policy.get("child_ir_hashes")
        expected_ranges = spec["ranges"]
        if (
            not isinstance(slices, list)
            or not isinstance(child_hashes, list)
            or not isinstance(expected_ranges, list)
            or len(slices) != spec["children"]
            or len(child_hashes) != spec["children"]
        ):
            raise ValueError("NRIR-6 child stack set differs")
        expected_query_ids: list[str] = []
        for slice_payload, child_payload, expected_range in zip(
            slices, child_hashes, expected_ranges
        ):
            joint_slice = _mapping(slice_payload, "NRIR-6 joint slice")
            child = _mapping(child_payload, "NRIR-6 child hashes")
            start, stop = expected_range
            query_id = f"{QUERY_ID}:spec:{start:04d}:{stop:04d}"
            expected_query_ids.append(query_id)
            if (
                joint_slice.get("start_index") != start
                or joint_slice.get("stop_index") != stop
                or joint_slice.get("child_query_id") != query_id
                or joint_slice.get("representation_policy_id") != spec["representation"]
                or joint_slice.get("child_ir_hashes") != child
                or len(child) != 10
                or any(len(str(value)) != 64 for value in child.values())
            ):
                raise ValueError("NRIR-6 child binding/range differs")
        execution = _mapping(policy.get("execution_trace"), f"NRIR-6 {name} execution")
        if (
            execution.get("schema_version")
            != JOINT_POLICY_EXECUTION_TRACE_SCHEMA_VERSION
            or execution.get("binding_hash") != hashes["joint_policy_binding_hash"]
            or execution.get("representation_policy_id") != spec["representation"]
            or execution.get("child_query_ids") != expected_query_ids
            or len(execution.get("child_task_trace_hashes", [])) != spec["children"]
            or len(execution.get("child_representation_binding_hashes", []))
            != spec["children"]
            or execution.get("result_lower_hash")
            != semantics.get(f"{name}_lower_sha256")
            or execution.get("result_upper_hash")
            != semantics.get(f"{name}_upper_sha256")
        ):
            raise ValueError("NRIR-6 execution trace linkage differs")
        comparison = _mapping(
            semantics.get(f"{name}_vs_external_lower"),
            f"NRIR-6 {name} semantics",
        )
        if (
            comparison.get("allclose") is not True
            or comparison.get("sign_agreement") != 9
            or comparison.get("sign_total") != 9
        ):
            raise ValueError("NRIR-6 semantic comparison differs")
    if (
        len(source_bound_hashes) != 1
        or len(source_template_hashes) != 1
        or len(source_instance_hashes) != 4
        or len(source_schedule_hashes) != 4
    ):
        raise ValueError("NRIR-6 source cross-product identity differs")


def generate_artifact(
    artifact_dir: Path, *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_joint_policy_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    (artifact_dir / "joint_policy.json").write_text(
        _canonical_json(evidence, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": JOINT_POLICY_ARTIFACT_SCHEMA_VERSION,
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
        manifest.get("schema_version") != JOINT_POLICY_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-6 manifest contract differs")
    files = _mapping(manifest.get("files"), "NRIR-6 manifest files")
    if set(files) != set(ARTIFACT_FILES):
        raise ValueError("NRIR-6 manifest file set differs")
    for name in ARTIFACT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError(f"NRIR-6 artifact digest differs: {name}")
    stored = _load_json(artifact_dir / "joint_policy.json")
    validate_joint_policy_evidence(stored)
    actual = build_joint_policy_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    if _canonical_json(stored) != _canonical_json(actual):
        raise ValueError("NRIR-6 semantic replay differs")
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
