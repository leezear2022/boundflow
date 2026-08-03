#!/usr/bin/env python3
"""Generate or replay NRIR-10 fixed-ResNet alpha/beta state evidence."""

# pylint: disable=too-many-locals,too-many-statements,duplicate-code
# pylint: disable=too-many-boolean-expressions,missing-function-docstring

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.ir.bound import OptimizedReluRelaxationAttrs
from boundflow.ir.schedule import LaunchAction
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NATIVE_ALPHA_BETA_STATE_SCHEMA_VERSION,
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    classify_native_alpha_beta_warm_start,
    compile_native_alpha_beta_state_query,
    execute_native_alpha_beta_state_query,
    optimize_native_alpha_beta_state,
)
from boundflow.runtime.native_relu_split_bab_runtime import _select_branch
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

ARTIFACT_SCHEMA_VERSION = "boundflow.native-alpha-beta-state-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.native-alpha-beta-state-evidence/v1"
ARTIFACT_FILE = "alpha_beta_state.json"
MANIFEST_FILE = "manifest.json"
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir10-alpha-beta-state"
POLICY = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.05, alpha_init=0.5, beta_init=0.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model", type=Path, required=True)
        subparser.add_argument("--source-artifact-dir", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def hashlib_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def build_native_alpha_beta_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-10 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    linear_spec = tensors["linear_spec_c"]
    if tuple(linear_spec.shape) != (1, 9, 10):
        raise ValueError("NRIR-10 frozen objective layout differs")
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    if tuple(op.op_type for op in module.get_entry_task().ops) != EXPECTED_PRIMAL_OPS:
        raise ValueError("NRIR-10 primal topology differs")
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    objective = linear_spec[:, 0:1].contiguous()
    _root_env, root_pre = _forward_ibp_trace_mlp(module, input_spec)
    parent_split = {
        name: torch.zeros_like(pre.lower, dtype=torch.int8)
        for name, pre in sorted(root_pre.items())
    }
    branch = _select_branch(root_pre, relu_split_state=parent_split)
    if branch is None:
        raise ValueError("NRIR-10 fixed ResNet has no ambiguous ReLU branch")
    parent = optimize_native_alpha_beta_state(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_split_state=parent_split,
        policy=POLICY,
    )
    exact_decision = classify_native_alpha_beta_warm_start(
        parent.state,
        target_scope=parent.state.scope,
        target_split_state=parent_split,
    )
    child_split = {name: value.clone() for name, value in parent_split.items()}
    child_split[branch.relu_input].reshape(-1)[branch.neuron_index] = 1
    child = optimize_native_alpha_beta_state(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_split_state=child_split,
        policy=POLICY,
        warm_start=parent.state,
    )
    if child.warm_start_decision is None:
        raise AssertionError("NRIR-10 child lacks warm-start decision")
    compilation = compile_native_alpha_beta_state_query(
        module,
        input_spec,
        linear_spec_C=objective,
        optimization=child,
        query_id=QUERY_ID,
    )
    native_bounds, task_trace = execute_native_alpha_beta_state_query(
        compilation,
        module,
        input_spec,
        linear_spec_C=objective,
        optimization=child,
    )

    zero_beta_state = NativeAlphaBetaOptimizationState(
        scope=child.state.scope,
        split_by_relu_input=child.state.split_by_relu_input,
        alpha_by_relu_input=child.state.alpha_by_relu_input,
        beta_by_relu_input=tuple(
            (name, torch.zeros_like(value))
            for name, value in child.state.beta_by_relu_input
        ),
    )
    zero_beta = replace(child, state=zero_beta_state)
    zero_compilation = compile_native_alpha_beta_state_query(
        module,
        input_spec,
        linear_spec_C=objective,
        optimization=zero_beta,
        query_id=f"{QUERY_ID}:zero-beta",
    )
    zero_bounds, _zero_trace = execute_native_alpha_beta_state_query(
        zero_compilation,
        module,
        input_spec,
        linear_spec_C=objective,
        optimization=zero_beta,
    )

    source_module = compilation.build.module
    execution_module = compilation.bound_module
    optimized_source_ops = sum(
        isinstance(op.attrs, OptimizedReluRelaxationAttrs)
        for op in source_module.graph.ops
    )
    optimized_execution_ops = sum(
        isinstance(op.attrs, OptimizedReluRelaxationAttrs)
        for op in execution_module.graph.ops
    )
    launches = sum(
        isinstance(action, LaunchAction) for action in compilation.schedule.actions
    )
    beta_sum = sum(float(value.sum().item()) for value in child.state.betas.values())
    lower_max_diff = float(
        (native_bounds.lower - child.bounds.lower).abs().max().item()
    )
    upper_max_diff = float(
        (native_bounds.upper - child.bounds.upper).abs().max().item()
    )
    beta_lower_improvement = float(
        (native_bounds.lower - zero_bounds.lower).min().item()
    )
    ir_hashes = compilation.hashes()
    zero_ir_hashes = zero_compilation.hashes()
    gates = {
        "fixed_resnet_source_is_digest_bound": True,
        "six_relu_split_alpha_beta_inputs_are_first_class": bool(
            len(child.state.splits) == 6
            and len(source_module.graph.inputs) == 19
            and optimized_source_ops == 6
            and optimized_execution_ops == 6
        ),
        "warm_start_is_monotonic_refinement_initialization_only": bool(
            exact_decision.kind == "exact"
            and exact_decision.exact_state_reuse_allowed
            and child.warm_start_decision.kind == "monotonic_split_refinement"
            and child.warm_start_decision.alpha_initialization_allowed
            and child.warm_start_decision.beta_initialization_allowed
            and not child.warm_start_decision.exact_state_reuse_allowed
        ),
        "native_bounds_match_legacy_alpha_beta_oracle": bool(
            lower_max_diff == 0.0 and upper_max_diff == 0.0
        ),
        "nonzero_beta_changes_native_lower_dual_result": bool(
            beta_sum > 0.0 and beta_lower_improvement > 0.0
        ),
        "plan_task_schedule_consume_optimized_state": bool(
            compilation.source_template.workload.alpha_enabled
            and compilation.source_template.workload.beta_enabled
            and compilation.execution_template.workload.alpha_enabled
            and compilation.execution_template.workload.beta_enabled
            and launches == len(compilation.task_module.tasks) == len(task_trace.events)
            and len(task_trace.events) == len(execution_module.graph.ops)
        ),
        "all_compiler_hashes_change_with_beta_payload": bool(
            set(ir_hashes) == set(zero_ir_hashes)
            and all(value != zero_ir_hashes[name] for name, value in ir_hashes.items())
        ),
        "claims_remain_correctness_only": True,
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-10 gates failed: {gates}")

    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "property_status": "not_claimed",
        "claim_boundary": (
            "fixed-ResNet frozen alpha/beta state ownership, monotonic-refinement "
            "warm-start validity, native beta lower-dual execution, and full "
            "Bound/Plan/Task/Schedule correctness only; runtime Adam loop is not "
            "compiled and there is no complete verifier, latency, memory, CUDA, or "
            "speedup claim"
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
            "input_lower_hash": tensor_content_hash(tensors["input_lower"]),
            "input_upper_hash": tensor_content_hash(tensors["input_upper"]),
            "objective_hash": tensor_content_hash(objective),
            "objective_index": 0,
        },
        "optimizer_policy": POLICY.to_dict(),
        "branch": branch.to_dict(),
        "parent_state": parent.state.to_dict(),
        "child_state": child.state.to_dict(),
        "exact_warm_start": exact_decision.to_dict(),
        "child_warm_start": child.warm_start_decision.to_dict(),
        "execution": {
            "legacy_lower_hash": tensor_content_hash(child.bounds.lower),
            "legacy_upper_hash": tensor_content_hash(child.bounds.upper),
            "native_lower_hash": tensor_content_hash(native_bounds.lower),
            "native_upper_hash": tensor_content_hash(native_bounds.upper),
            "zero_beta_lower_hash": tensor_content_hash(zero_bounds.lower),
            "lower_max_abs_diff": lower_max_diff,
            "upper_max_abs_diff": upper_max_diff,
            "beta_sum": beta_sum,
            "beta_lower_improvement": beta_lower_improvement,
            "source_graph_input_count": len(source_module.graph.inputs),
            "source_optimized_relu_op_count": optimized_source_ops,
            "execution_optimized_relu_op_count": optimized_execution_ops,
            "task_count": len(compilation.task_module.tasks),
            "schedule_launch_count": launches,
            "task_trace_event_count": len(task_trace.events),
        },
        "ir_hashes": ir_hashes,
        "zero_beta_ir_hashes": zero_ir_hashes,
        "gates": gates,
        "limitations": [
            "the Adam optimization loop remains runtime-owned and is not a Task/Schedule program",
            "one parent and one monotonic child are evaluated; this is not a complete BaB search",
            "property_status remains not_claimed",
            "CPU correctness only; no latency, memory, CUDA, or speedup evidence",
        ],
    }
    validate_native_alpha_beta_evidence(evidence)
    return evidence


def validate_native_alpha_beta_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("property_status") != "not_claimed"
        or "runtime Adam loop is not compiled"
        not in str(evidence.get("claim_boundary", ""))
    ):
        raise ValueError("NRIR-10 evidence header/claim boundary differs")
    source = _mapping(evidence.get("source"), "NRIR-10 source")
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
                "input_lower_hash",
                "input_upper_hash",
                "objective_hash",
            )
        )
    ):
        raise ValueError("NRIR-10 source identity differs")
    policy = _mapping(evidence.get("optimizer_policy"), "NRIR-10 policy")
    if policy != POLICY.to_dict():
        raise ValueError("NRIR-10 optimizer policy differs")
    branch = _mapping(evidence.get("branch"), "NRIR-10 branch")
    if (
        not branch.get("relu_input")
        or not isinstance(branch.get("neuron_index"), int)
        or float(branch.get("lower", 0.0)) >= 0.0
        or float(branch.get("upper", 0.0)) <= 0.0
    ):
        raise ValueError("NRIR-10 branch identity differs")
    parent = _mapping(evidence.get("parent_state"), "NRIR-10 parent state")
    child = _mapping(evidence.get("child_state"), "NRIR-10 child state")
    for name, state in (("parent", parent), ("child", child)):
        relu_states = _mapping(state.get("relu_states"), f"NRIR-10 {name} ReLUs")
        if (
            state.get("schema_version") != NATIVE_ALPHA_BETA_STATE_SCHEMA_VERSION
            or len(str(state.get("payload_hash", ""))) != 64
            or len(str(state.get("state_hash", ""))) != 64
            or len(relu_states) != 6
            or any(
                len(str(item.get(hash_name, ""))) != 64
                for item_value in relu_states.values()
                for item in (_mapping(item_value, "NRIR-10 ReLU state"),)
                for hash_name in ("split_hash", "alpha_hash", "beta_hash")
            )
        ):
            raise ValueError(f"NRIR-10 {name} state identity differs")
    exact = _mapping(evidence.get("exact_warm_start"), "NRIR-10 exact warm start")
    warm = _mapping(evidence.get("child_warm_start"), "NRIR-10 child warm start")
    if (
        exact.get("kind") != "exact"
        or exact.get("exact_state_reuse_allowed") is not True
        or warm.get("kind") != "monotonic_split_refinement"
        or warm.get("alpha_initialization_allowed") is not True
        or warm.get("beta_initialization_allowed") is not True
        or warm.get("exact_state_reuse_allowed") is not False
    ):
        raise ValueError("NRIR-10 warm-start validity differs")
    execution = _mapping(evidence.get("execution"), "NRIR-10 execution")
    if (
        execution.get("legacy_lower_hash") != execution.get("native_lower_hash")
        or execution.get("legacy_upper_hash") != execution.get("native_upper_hash")
        or execution.get("lower_max_abs_diff") != 0.0
        or execution.get("upper_max_abs_diff") != 0.0
        or float(execution.get("beta_sum", 0.0)) <= 0.0
        or float(execution.get("beta_lower_improvement", 0.0)) <= 0.0
        or execution.get("source_graph_input_count") != 19
        or execution.get("source_optimized_relu_op_count") != 6
        or execution.get("execution_optimized_relu_op_count") != 6
        or execution.get("task_count") != 21
        or execution.get("schedule_launch_count") != 21
        or execution.get("task_trace_event_count") != 21
    ):
        raise ValueError("NRIR-10 execution/beta contract differs")
    ir_hashes = _mapping(evidence.get("ir_hashes"), "NRIR-10 IR hashes")
    zero_hashes = _mapping(
        evidence.get("zero_beta_ir_hashes"), "NRIR-10 zero-beta IR hashes"
    )
    if (
        set(ir_hashes) != set(zero_hashes)
        or len(ir_hashes) != 10
        or any(len(str(value)) != 64 for value in ir_hashes.values())
        or any(ir_hashes[name] == zero_hashes[name] for name in ir_hashes)
    ):
        raise ValueError("NRIR-10 compiler hash/state binding differs")
    gates = _mapping(evidence.get("gates"), "NRIR-10 gates")
    expected_gates = {
        "fixed_resnet_source_is_digest_bound",
        "six_relu_split_alpha_beta_inputs_are_first_class",
        "warm_start_is_monotonic_refinement_initialization_only",
        "native_bounds_match_legacy_alpha_beta_oracle",
        "nonzero_beta_changes_native_lower_dual_result",
        "plan_task_schedule_consume_optimized_state",
        "all_compiler_hashes_change_with_beta_payload",
        "claims_remain_correctness_only",
    }
    if set(gates) != expected_gates or any(
        value is not True for value in gates.values()
    ):
        raise ValueError("NRIR-10 gates differ")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 4:
        raise ValueError("NRIR-10 limitations differ")


def _generate(args: argparse.Namespace) -> None:
    evidence = build_native_alpha_beta_evidence(
        model=args.model, source_artifact_dir=args.source_artifact_dir
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "evidence": evidence,
    }
    artifact_path = args.artifact_dir / ARTIFACT_FILE
    artifact_path.write_text(
        _canonical_json(artifact, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "files": {ARTIFACT_FILE: file_sha256(artifact_path)},
        "evidence_hash": hashlib_sha256(evidence),
    }
    (args.artifact_dir / MANIFEST_FILE).write_text(
        _canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def _replay(args: argparse.Namespace) -> None:
    manifest = json.loads(
        (args.artifact_dir / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    artifact_path = args.artifact_dir / ARTIFACT_FILE
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != {ARTIFACT_FILE: file_sha256(artifact_path)}
        or artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or artifact.get("status") != "ok"
    ):
        raise ValueError("NRIR-10 artifact manifest/header differs")
    stored = _mapping(artifact.get("evidence"), "NRIR-10 stored evidence")
    validate_native_alpha_beta_evidence(stored)
    if manifest.get("evidence_hash") != hashlib_sha256(stored):
        raise ValueError("NRIR-10 stored evidence hash differs")
    actual = build_native_alpha_beta_evidence(
        model=args.model, source_artifact_dir=args.source_artifact_dir
    )
    if stored != actual:
        raise ValueError("NRIR-10 replay differs from frozen evidence")
    print(_canonical_json({"status": "ok", "evidence_hash": hashlib_sha256(actual)}))


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
