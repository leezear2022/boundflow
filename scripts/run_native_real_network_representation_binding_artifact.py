#!/usr/bin/env python3
"""Generate or replay NRIR-4 representation-binding evidence on frozen ResNet."""

# pylint: disable=duplicate-code,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.ir.bound import BoundOpKind, BoundRepresentation
from boundflow.ir.schedule import LaunchAction, MaterializeAction
from boundflow.planner import plan_interval_ibp_v0
from boundflow.planner.plan_ir_selector import NoFeasiblePlanError
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
from boundflow.runtime.native_verifier_ir_integration import (
    NATIVE_PLAIN_CROWN_REPRESENTATION_COMPILER_VERSION,
    NativePlainCrownRepresentationCompilation,
    compile_native_plain_crown_representation_query,
    execute_native_plain_crown_representation_query,
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

REPRESENTATION_ARTIFACT_SCHEMA_VERSION = (
    "boundflow.native-real-network-representation-artifact/v1"
)
REPRESENTATION_EVIDENCE_SCHEMA_VERSION = (
    "boundflow.native-real-network-representation-evidence/v1"
)
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir4-representation"
AVAILABLE_MEMORY_BYTES = 1 << 30
ATOL = 2e-4
RTOL = 2e-4
ARTIFACT_FILES = ("representation.json",)


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


def build_representation_binding_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    """Rebuild, bind, and execute both representation policies from source data."""

    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-4 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    external_bounds = deserialize_intermediate_bounds(
        _mapping(payload.get("external_intermediate_bounds"), "intermediate bounds")
    )
    if intermediate_bounds_sha256(external_bounds) != INTERMEDIATE_BOUNDS_SHA256:
        raise ValueError("NRIR-4 intermediate-bound digest differs")

    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    legacy_module = plan_interval_ibp_v0(program)
    primal_ops = tuple(op.op_type for op in legacy_module.get_entry_task().ops)
    if primal_ops != EXPECTED_PRIMAL_OPS:
        raise ValueError("NRIR-4 primal topology differs")
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    interval_env, local_relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    relu_pre = bind_intermediate_bounds(external_bounds, local_relu_pre)

    def compile_with_budget(
        memory_budget_bytes: int,
    ) -> NativePlainCrownRepresentationCompilation:
        return compile_native_plain_crown_representation_query(
            legacy_module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=tensors["linear_spec_c"],
            intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
            query_id=QUERY_ID,
            available_memory_bytes=AVAILABLE_MEMORY_BYTES,
            memory_budget_bytes=memory_budget_bytes,
        )

    dense = compile_with_budget(AVAILABLE_MEMORY_BYTES)
    storage_candidates = {
        candidate.candidate_id: candidate
        for candidate in dense.source_template.storage_candidates
    }
    retain = storage_candidates["storage:native-retain-all-v1"]
    reuse = storage_candidates["storage:native-lifetime-reuse-v1"]
    structured = compile_with_budget(reuse.cost.predicted_peak_bytes)
    below_minimum_failures: list[dict[str, object]] = []
    try:
        compile_with_budget(reuse.cost.predicted_peak_bytes - 1)
    except NoFeasiblePlanError as error:
        below_minimum_failures = [
            {"reason": failure.reason, "count": failure.count}
            for failure in error.failures
        ]
    if not below_minimum_failures:
        raise ValueError("NRIR-4 accepted a budget below both source policies")

    dense_result, dense_task_trace = execute_native_plain_crown_representation_query(
        dense,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=tensors["linear_spec_c"],
    )
    structured_result, structured_task_trace = (
        execute_native_plain_crown_representation_query(
            structured,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=tensors["linear_spec_c"],
        )
    )
    expected = tensors["external_lower"].to(dense_result.lower)
    dense_vs_structured = _comparison(dense_result.lower, structured_result.lower)
    dense_vs_external = _comparison(dense_result.lower, expected)
    structured_vs_external = _comparison(structured_result.lower, expected)
    if not all(
        comparison["allclose"] is True
        and comparison["sign_agreement"] == comparison["sign_total"] == 9
        for comparison in (
            dense_vs_structured,
            dense_vs_external,
            structured_vs_external,
        )
    ):
        raise ValueError("NRIR-4 representation executions differ semantically")
    torch.testing.assert_close(
        dense_result.upper, structured_result.upper, atol=ATOL, rtol=RTOL
    )

    transition_ops = tuple(
        op
        for op in structured.bound_module.graph.ops
        if op.kind in {BoundOpKind.REPRESENTATION_CAST, BoundOpKind.MATERIALIZE}
    )
    transition_actions = tuple(
        action
        for action in structured.source_schedule.actions
        if isinstance(action, MaterializeAction)
    )
    event_op_ids = {event.execution_op_id for event in structured.binding.trace.events}
    task_op_ids = {
        op_id for event in structured_task_trace.events for op_id in event.op_ids
    }
    execution_values = {
        value.value_id: value for value in structured.bound_module.graph.values
    }
    execution_storage = structured.execution_template.storage_candidates[0]
    structured_bindings = tuple(
        binding
        for binding in execution_storage.bindings
        if execution_values[binding.value_id].representation
        == BoundRepresentation.STRUCTURED
    )
    dense_hashes = dense.hashes()
    structured_hashes = structured.hashes()
    gates = {
        "same_source_bound_and_template": bool(
            dense_hashes["source_bound_module_hash"]
            == structured_hashes["source_bound_module_hash"]
            and dense_hashes["source_plan_template_hash"]
            == structured_hashes["source_plan_template_hash"]
        ),
        "budget_switches_global_policy": bool(
            dense.binding.trace.policy_id == DENSE_POLICY_ID
            and structured.binding.trace.policy_id == STRUCTURED_AFFINE_POLICY_ID
            and dense.source_instance.storage_decision.candidate_id
            == retain.candidate_id
            and structured.source_instance.storage_decision.candidate_id
            == reuse.candidate_id
        ),
        "dense_execution_is_source_bound": bool(
            dense_hashes["source_bound_module_hash"]
            == dense_hashes["execution_bound_module_hash"]
        ),
        "structured_execution_is_rewritten_bound": bool(
            structured_hashes["source_bound_module_hash"]
            != structured_hashes["execution_bound_module_hash"]
        ),
        "transitions_bind_plan_schedule_bound_one_to_one": bool(
            transition_ops
            and len(transition_ops)
            == len(transition_actions)
            == len(structured.binding.trace.events)
            and event_op_ids == {op.op_id for op in transition_ops}
        ),
        "transition_ops_are_tasked_and_launched": bool(
            event_op_ids <= task_op_ids
            and sum(
                isinstance(action, LaunchAction)
                for action in structured.schedule.actions
            )
            == len(structured.bound_module.graph.ops)
        ),
        "structured_storage_is_dense_equivalent_not_compressed": bool(
            structured_bindings
            and all(
                binding.representation == BoundRepresentation.STRUCTURED
                and binding.size_bytes >= binding.logical_size_bytes
                for binding in structured_bindings
            )
        ),
        "both_policies_match_semantics": bool(
            dense_vs_structured["allclose"]
            and dense_vs_external["allclose"]
            and structured_vs_external["allclose"]
        ),
        "below_minimum_budget_fails_closed": any(
            item["reason"] == "memory_budget_exceeded"
            for item in below_minimum_failures
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-4 gates failed: {gates}")

    transition_kinds = Counter(op.kind.value for op in transition_ops)
    evidence: dict[str, object] = {
        "schema_version": REPRESENTATION_EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "compiler_version": NATIVE_PLAIN_CROWN_REPRESENTATION_COMPILER_VERSION,
        "claim_boundary": (
            "real-network representation Plan-to-Bound semantic binding and CPU "
            "correctness only; structured values retain dense-equivalent physical "
            "size and support no compression, memory, latency, CUDA, or Pareto claim"
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
        },
        "source_plan": {
            "bound_op_count": len(dense.source_bound_module.graph.ops),
            "representation_candidate_count": len(
                dense.source_template.representation_candidates
            ),
            "transition_candidate_count": len(
                dense.source_template.materialization_candidates
            ),
            "retain_all_peak_bytes": retain.cost.predicted_peak_bytes,
            "lifetime_reuse_peak_bytes": reuse.cost.predicted_peak_bytes,
            "below_minimum_budget": reuse.cost.predicted_peak_bytes - 1,
            "below_minimum_failures": below_minimum_failures,
        },
        "dense_policy": _policy_summary(dense, dense_task_trace.stable_hash()),
        "structured_affine_policy": {
            **_policy_summary(structured, structured_task_trace.stable_hash()),
            "binding_trace": structured.binding.trace.to_dict(),
            "transition_op_count": len(transition_ops),
            "transition_op_kinds": dict(sorted(transition_kinds.items())),
            "source_schedule_materialization_count": len(transition_actions),
            "structured_storage_binding_count": len(structured_bindings),
            "structured_storage_dense_equivalent": True,
        },
        "semantics": {
            "dense_vs_structured_lower": dense_vs_structured,
            "dense_vs_external_lower": dense_vs_external,
            "structured_vs_external_lower": structured_vs_external,
            "dense_lower_sha256": tensor_content_hash(dense_result.lower),
            "structured_lower_sha256": tensor_content_hash(structured_result.lower),
            "dense_upper_sha256": tensor_content_hash(dense_result.upper),
            "structured_upper_sha256": tensor_content_hash(structured_result.upper),
        },
        "gates": gates,
        "limitations": [
            "CPU correctness and compiler-ownership evidence only",
            "the source selector admits exactly two global representation policies",
            "arbitrary mixed per-region representation policies remain unsupported",
            (
                "DenseLinearOperator stores a dense tensor and every structured "
                "binding reserves at least its dense logical byte size"
            ),
            (
                "the source policy is coupled to NRIR-2 retain/reuse storage only "
                "to make deterministic budget selection explicit"
            ),
            "no compression, memory reduction, latency, CUDA, allocator, OOM, or Pareto claim",
        ],
    }
    validate_representation_binding_evidence(evidence)
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


def _policy_summary(compilation, task_trace_hash: str) -> dict[str, object]:
    return {
        "policy_id": compilation.binding.trace.policy_id,
        "memory_budget_bytes": compilation.source_instance.memory_budget_bytes,
        "storage_candidate_id": (
            compilation.source_instance.storage_decision.candidate_id
        ),
        "ir_hashes": compilation.hashes(),
        "source_schedule_materialization_count": sum(
            isinstance(action, MaterializeAction)
            for action in compilation.source_schedule.actions
        ),
        "execution_bound_op_count": len(compilation.bound_module.graph.ops),
        "execution_task_count": len(compilation.task_module.tasks),
        "execution_launch_count": sum(
            isinstance(action, LaunchAction) for action in compilation.schedule.actions
        ),
        "task_trace_hash": task_trace_hash,
    }


def validate_representation_binding_evidence(evidence: Mapping[str, Any]) -> None:
    """Reject incomplete, unlinked, or claim-inflated NRIR-4 evidence."""

    if (
        evidence.get("schema_version") != REPRESENTATION_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("compiler_version")
        != NATIVE_PLAIN_CROWN_REPRESENTATION_COMPILER_VERSION
    ):
        raise ValueError("NRIR-4 evidence contract differs")
    claim = str(evidence.get("claim_boundary", ""))
    if any(
        phrase not in claim
        for phrase in ("semantic binding", "CPU correctness", "no compression")
    ) or "no compression" not in str(evidence.get("limitations", ())):
        raise ValueError("NRIR-4 claim boundary omits representation limitations")
    gates = _mapping(evidence.get("gates"), "NRIR-4 gates")
    expected_gates = {
        "same_source_bound_and_template",
        "budget_switches_global_policy",
        "dense_execution_is_source_bound",
        "structured_execution_is_rewritten_bound",
        "transitions_bind_plan_schedule_bound_one_to_one",
        "transition_ops_are_tasked_and_launched",
        "structured_storage_is_dense_equivalent_not_compressed",
        "both_policies_match_semantics",
        "below_minimum_budget_fails_closed",
    }
    if set(gates) != expected_gates or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("NRIR-4 gates are incomplete or failed")
    source_plan = _mapping(evidence.get("source_plan"), "NRIR-4 source plan")
    if (
        int(source_plan.get("bound_op_count", 0)) <= 1
        or int(source_plan.get("transition_candidate_count", 0)) <= 0
        or int(source_plan.get("retain_all_peak_bytes", 0))
        <= int(source_plan.get("lifetime_reuse_peak_bytes", 0))
        or int(source_plan.get("below_minimum_budget", 0))
        != int(source_plan.get("lifetime_reuse_peak_bytes", 0)) - 1
    ):
        raise ValueError("NRIR-4 source Plan contract differs")
    dense = _mapping(evidence.get("dense_policy"), "NRIR-4 dense policy")
    structured = _mapping(
        evidence.get("structured_affine_policy"), "NRIR-4 structured policy"
    )
    if (
        dense.get("policy_id") != DENSE_POLICY_ID
        or structured.get("policy_id") != STRUCTURED_AFFINE_POLICY_ID
        or dense.get("storage_candidate_id") != "storage:native-retain-all-v1"
        or structured.get("storage_candidate_id") != "storage:native-lifetime-reuse-v1"
        or int(dense.get("source_schedule_materialization_count", -1)) != 0
    ):
        raise ValueError("NRIR-4 selected policy identity differs")
    transition_count = int(structured.get("transition_op_count", 0))
    transition_kinds = _mapping(
        structured.get("transition_op_kinds"), "NRIR-4 transition kinds"
    )
    if (
        transition_count <= 0
        or transition_count
        != int(structured.get("source_schedule_materialization_count", 0))
        or set(transition_kinds) != {"representation_cast", "materialize"}
        or sum(int(value) for value in transition_kinds.values()) != transition_count
        or structured.get("structured_storage_dense_equivalent") is not True
        or int(structured.get("structured_storage_binding_count", 0)) <= 0
    ):
        raise ValueError("NRIR-4 transition/storage evidence differs")
    dense_hashes = _validate_hashes(_mapping(dense.get("ir_hashes"), "dense hashes"))
    structured_hashes = _validate_hashes(
        _mapping(structured.get("ir_hashes"), "structured hashes")
    )
    if (
        dense_hashes["source_bound_module_hash"]
        != structured_hashes["source_bound_module_hash"]
        or dense_hashes["source_plan_template_hash"]
        != structured_hashes["source_plan_template_hash"]
        or dense_hashes["source_bound_module_hash"]
        != dense_hashes["execution_bound_module_hash"]
        or structured_hashes["source_bound_module_hash"]
        == structured_hashes["execution_bound_module_hash"]
    ):
        raise ValueError("NRIR-4 source/execution hash linkage differs")
    binding = _mapping(structured.get("binding_trace"), "NRIR-4 binding trace")
    events = binding.get("events")
    selected_transitions = binding.get("selected_transition_candidate_ids")
    selected_representations = binding.get("selected_representation_candidate_ids")
    event_mappings = (
        [_mapping(event, "binding event") for event in events]
        if isinstance(events, list)
        else []
    )
    if (
        binding.get("policy_id") != STRUCTURED_AFFINE_POLICY_ID
        or binding.get("source_bound_module_hash")
        != structured_hashes["source_bound_module_hash"]
        or binding.get("source_plan_template_hash")
        != structured_hashes["source_plan_template_hash"]
        or binding.get("source_plan_instance_hash")
        != structured_hashes["source_plan_instance_hash"]
        or binding.get("source_schedule_hash")
        != structured_hashes["source_schedule_hash"]
        or binding.get("execution_bound_module_hash")
        != structured_hashes["execution_bound_module_hash"]
        or not isinstance(events, list)
        or len(events) != transition_count
        or not isinstance(selected_transitions, list)
        or selected_transitions
        != [event.get("transition_candidate_id") for event in event_mappings]
        or not isinstance(selected_representations, list)
        or len(selected_representations) != int(source_plan["bound_op_count"])
        or any(
            not str(candidate_id).startswith(
                f"representation:{STRUCTURED_AFFINE_POLICY_ID}:"
            )
            for candidate_id in selected_representations
        )
        or len({str(event.get("transition_candidate_id")) for event in event_mappings})
        != transition_count
        or any(
            len({str(event.get(field)) for event in event_mappings}) != transition_count
            for field in (
                "schedule_action_id",
                "execution_op_id",
                "execution_output_value_id",
            )
        )
        or hashlib.sha256(_canonical_json(binding).encode("utf-8")).hexdigest()
        != structured_hashes["representation_binding_hash"]
    ):
        raise ValueError("NRIR-4 binding trace linkage differs")
    event_kinds = Counter(
        (
            "representation_cast"
            if event.get("transition_kind") == "cast"
            else str(event.get("transition_kind"))
        )
        for event in event_mappings
    )
    if dict(event_kinds) != {
        str(key): int(value) for key, value in transition_kinds.items()
    } or any(
        (
            event.get("transition_kind") == "materialize"
            and event.get("target_representation") != "dense"
        )
        or event.get("source_representation") == event.get("target_representation")
        for event in event_mappings
    ):
        raise ValueError("NRIR-4 binding event semantics differ")
    dense_op_count = int(dense.get("execution_bound_op_count", 0))
    structured_op_count = int(structured.get("execution_bound_op_count", 0))
    if (
        dense_op_count != int(source_plan["bound_op_count"])
        or structured_op_count != dense_op_count + transition_count
        or int(dense.get("execution_task_count", 0)) != dense_op_count
        or int(dense.get("execution_launch_count", 0)) != dense_op_count
        or int(structured.get("execution_task_count", 0)) != structured_op_count
        or int(structured.get("execution_launch_count", 0)) != structured_op_count
    ):
        raise ValueError("NRIR-4 execution ownership counts differ")
    semantics = _mapping(evidence.get("semantics"), "NRIR-4 semantics")
    for name in (
        "dense_vs_structured_lower",
        "dense_vs_external_lower",
        "structured_vs_external_lower",
    ):
        comparison = _mapping(semantics.get(name), name)
        if (
            comparison.get("allclose") is not True
            or comparison.get("sign_agreement") != 9
            or comparison.get("sign_total") != 9
        ):
            raise ValueError("NRIR-4 semantic comparison differs")


def _validate_hashes(values: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "source_bound_module_hash",
        "source_plan_template_hash",
        "source_plan_instance_hash",
        "source_schedule_hash",
        "representation_binding_hash",
        "execution_bound_module_hash",
        "execution_plan_template_hash",
        "execution_plan_instance_hash",
        "task_module_hash",
        "schedule_hash",
    }
    if set(values) != expected or any(
        len(str(value)) != 64 for value in values.values()
    ):
        raise ValueError("NRIR-4 IR hash set differs")
    return values


def generate_artifact(
    artifact_dir: Path, *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    """Write immutable NRIR-4 evidence without duplicating the source payload."""

    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_representation_binding_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    (artifact_dir / "representation.json").write_text(
        _canonical_json(evidence, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": REPRESENTATION_ARTIFACT_SCHEMA_VERSION,
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
    """Verify digests, then independently rebuild both semantic executions."""

    manifest = _load_json(artifact_dir / "manifest.json")
    if (
        manifest.get("schema_version") != REPRESENTATION_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-4 manifest contract differs")
    files = _mapping(manifest.get("files"), "NRIR-4 manifest files")
    if set(files) != set(ARTIFACT_FILES):
        raise ValueError("NRIR-4 manifest file set differs")
    for name in ARTIFACT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError(f"NRIR-4 artifact digest differs: {name}")
    stored = _load_json(artifact_dir / "representation.json")
    validate_representation_binding_evidence(stored)
    actual = build_representation_binding_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    if _canonical_json(stored) != _canonical_json(actual):
        raise ValueError("NRIR-4 semantic replay differs")
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
