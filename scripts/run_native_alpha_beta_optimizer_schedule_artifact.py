#!/usr/bin/env python3
"""Generate or replay NRIR-11 fixed-ResNet optimizer Schedule evidence."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,duplicate-code
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
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NATIVE_ALPHA_BETA_STATE_SCHEMA_VERSION,
    NativeAlphaBetaOptimizationResult,
    NativeAlphaBetaOptimizerPolicy,
    compile_native_alpha_beta_state_query,
    execute_native_alpha_beta_state_query,
    optimize_native_alpha_beta_state,
)
from boundflow.runtime.native_alpha_beta_optimizer_schedule import (
    NATIVE_OPTIMIZER_EXECUTION_TRACE_SCHEMA_VERSION,
    compile_native_alpha_beta_optimizer_program,
    execute_native_alpha_beta_optimizer_program,
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

ARTIFACT_SCHEMA_VERSION = "boundflow.native-alpha-beta-optimizer-schedule-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.native-alpha-beta-optimizer-schedule-evidence/v1"
ARTIFACT_FILE = "optimizer_schedule.json"
MANIFEST_FILE = "manifest.json"
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir11-optimizer-schedule"
POLICY = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.05, alpha_init=0.5, beta_init=0.0)
EXPECTED_TASK_KINDS = (
    "evaluate_bound",
    "reduce_metric",
    "backward",
    "adam_update",
    "project_state",
    "evaluate_bound",
    "reduce_metric",
    "select_best",
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


def _sequence(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_state_summary(value: object, label: str) -> Mapping[str, Any]:
    state = _mapping(value, label)
    relu_states = _mapping(state.get("relu_states"), f"{label} ReLU states")
    if (
        state.get("schema_version") != NATIVE_ALPHA_BETA_STATE_SCHEMA_VERSION
        or not _sha256(state.get("payload_hash"))
        or not _sha256(state.get("state_hash"))
        or len(relu_states) != 6
    ):
        raise ValueError(f"{label} identity differs")
    return state


def _validate_program_and_trace(evidence: Mapping[str, Any]) -> None:
    program = _mapping(evidence.get("optimizer_program"), "NRIR-11 optimizer program")
    plan = _mapping(program.get("plan"), "NRIR-11 optimizer Plan")
    task_module = _mapping(program.get("task_module"), "NRIR-11 optimizer Task module")
    schedule = _mapping(program.get("schedule"), "NRIR-11 optimizer Schedule")
    hashes = _mapping(program.get("hashes"), "NRIR-11 optimizer hashes")
    source_hashes = _mapping(
        program.get("source_compiler_hashes"), "NRIR-11 source compiler hashes"
    )
    initial_state = _validate_state_summary(
        evidence.get("initial_state"), "NRIR-11 initial state"
    )
    selected_state = _validate_state_summary(
        evidence.get("selected_state"), "NRIR-11 selected state"
    )
    policy = _mapping(evidence.get("optimizer_policy"), "NRIR-11 optimizer policy")

    if (
        plan.get("schema_version") != "boundflow.optimizer_plan_ir/v1"
        or plan.get("steps") != 1
        or plan.get("warm_start_kind") != "monotonic_split_refinement"
        or plan.get("performance_claimed") is not False
        or plan.get("initial_state_hash") != initial_state.get("state_hash")
        or plan.get("optimizer_policy_hash") != hashlib_sha256(policy)
        or plan.get("source_ir_hashes") != source_hashes
        or len(source_hashes) != 10
        or any(not _sha256(value) for value in source_hashes.values())
        or hashes.get("optimizer_plan_hash") != hashlib_sha256(plan)
    ):
        raise ValueError("NRIR-11 optimizer Plan/source identity differs")

    tasks = _sequence(task_module.get("tasks"), "NRIR-11 optimizer tasks")
    actions = _sequence(schedule.get("actions"), "NRIR-11 optimizer actions")
    if (
        task_module.get("schema_version") != "boundflow.optimizer_task_ir/v1"
        or task_module.get("optimizer_plan_hash") != hashes.get("optimizer_plan_hash")
        or hashes.get("optimizer_task_module_hash") != hashlib_sha256(task_module)
        or schedule.get("schema_version") != "boundflow.optimizer_schedule_ir/v1"
        or schedule.get("optimizer_plan_hash") != hashes.get("optimizer_plan_hash")
        or schedule.get("optimizer_task_module_hash")
        != hashes.get("optimizer_task_module_hash")
        or hashes.get("optimizer_schedule_hash") != hashlib_sha256(schedule)
        or len(tasks) != 8
        or len(actions) != 8
        or tuple(_mapping(task, "NRIR-11 task").get("kind") for task in tasks)
        != EXPECTED_TASK_KINDS
    ):
        raise ValueError("NRIR-11 optimizer Task/Schedule identity differs")

    for sequence, (task_value, action_value) in enumerate(zip(tasks, actions)):
        task = _mapping(task_value, "NRIR-11 task")
        action = _mapping(action_value, "NRIR-11 action")
        if (
            action.get("sequence") != sequence
            or action.get("task_id") != task.get("task_id")
            or action.get("input_value_ids") != task.get("input_value_ids")
            or action.get("output_value_ids") != task.get("output_value_ids")
        ):
            raise ValueError("NRIR-11 Schedule/Task linkage differs")

    trace = _mapping(evidence.get("execution_trace"), "NRIR-11 execution trace")
    trace_actions = _sequence(trace.get("actions"), "NRIR-11 trace actions")
    evaluations = _sequence(trace.get("evaluations"), "NRIR-11 evaluations")
    if (
        trace.get("schema_version") != NATIVE_OPTIMIZER_EXECUTION_TRACE_SCHEMA_VERSION
        or trace.get("performance_claimed") is not False
        or trace.get("plan_hash") != hashes.get("optimizer_plan_hash")
        or trace.get("task_module_hash") != hashes.get("optimizer_task_module_hash")
        or trace.get("schedule_hash") != hashes.get("optimizer_schedule_hash")
        or trace.get("initial_state_hash") != initial_state.get("state_hash")
        or trace.get("selected_state_hash") != selected_state.get("state_hash")
        or program.get("execution_trace_hash") != hashlib_sha256(trace)
        or len(trace_actions) != len(actions)
        or len(evaluations) != 2
        or trace.get("best_iteration_by_domain") != [1]
    ):
        raise ValueError("NRIR-11 execution trace header differs")

    evaluation_by_iteration = {
        int(_mapping(value, "NRIR-11 evaluation")["iteration"]): _mapping(
            value, "NRIR-11 evaluation"
        )
        for value in evaluations
    }
    runtime_hashes: dict[str, object] = {
        "optimizer.state.s000": initial_state.get("state_hash")
    }
    for sequence, (task_value, action_value, trace_value) in enumerate(
        zip(tasks, actions, trace_actions)
    ):
        task = _mapping(task_value, "NRIR-11 task")
        action = _mapping(action_value, "NRIR-11 action")
        action_trace = _mapping(trace_value, "NRIR-11 action trace")
        input_ids = _sequence(task.get("input_value_ids"), "NRIR-11 task inputs")
        output_ids = _sequence(task.get("output_value_ids"), "NRIR-11 task outputs")
        input_hashes = _mapping(
            action_trace.get("input_hashes"), "NRIR-11 trace input hashes"
        )
        output_hashes = _mapping(
            action_trace.get("output_hashes"), "NRIR-11 trace output hashes"
        )
        if (
            action_trace.get("sequence") != sequence
            or action_trace.get("action_id") != action.get("action_id")
            or action_trace.get("task_id") != task.get("task_id")
            or action_trace.get("kind") != task.get("kind")
            or set(input_hashes) != set(input_ids)
            or set(output_hashes) != set(output_ids)
            or any(input_hashes[name] != runtime_hashes.get(name) for name in input_ids)
            or any(not _sha256(output_hashes[name]) for name in output_ids)
            or any(name in runtime_hashes for name in output_ids)
        ):
            raise ValueError("NRIR-11 execution transition chain differs")
        runtime_hashes.update(output_hashes)
        iteration = action_trace.get("iteration")
        kind = task.get("kind")
        if kind == "evaluate_bound":
            if not isinstance(iteration, int):
                raise ValueError("NRIR-11 evaluation iteration differs")
            evaluation = evaluation_by_iteration.get(iteration)
            if (
                evaluation is None
                or input_hashes[input_ids[0]] != evaluation.get("state_hash")
                or output_hashes[output_ids[0]]
                != hashlib_sha256(
                    {
                        "lower": evaluation.get("lower_hash"),
                        "upper": evaluation.get("upper_hash"),
                    }
                )
            ):
                raise ValueError("NRIR-11 evaluated state/bound transition differs")
        elif kind == "reduce_metric":
            if not isinstance(iteration, int):
                raise ValueError("NRIR-11 reduction iteration differs")
            evaluation = evaluation_by_iteration.get(iteration)
            if evaluation is None or output_hashes[output_ids[0]] != evaluation.get(
                "metric_hash"
            ):
                raise ValueError("NRIR-11 metric transition differs")
        elif kind == "backward":
            if (
                float(action_trace.get("alpha_gradient_l1", 0.0)) <= 0.0
                or float(action_trace.get("beta_gradient_l1", 0.0)) <= 0.0
            ):
                raise ValueError("NRIR-11 backward gradient evidence differs")
        elif kind == "project_state":
            if action_trace.get("projection_applied") is not True:
                raise ValueError("NRIR-11 projection evidence differs")
        elif kind == "select_best" and output_hashes[
            output_ids[0]
        ] != selected_state.get("state_hash"):
            raise ValueError("NRIR-11 selected state transition differs")


def build_native_optimizer_schedule_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-11 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    program_value = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program_value)
    if tuple(op.op_type for op in module.get_entry_task().ops) != EXPECTED_PRIMAL_OPS:
        raise ValueError("NRIR-11 primal topology differs")
    input_spec = InputSpec.box(
        value_name=program_value.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    objective = tensors["linear_spec_c"][:, 0:1].contiguous()
    _root_env, root_pre = _forward_ibp_trace_mlp(module, input_spec)
    parent_split = {
        name: torch.zeros_like(pre.lower, dtype=torch.int8)
        for name, pre in sorted(root_pre.items())
    }
    branch = _select_branch(root_pre, relu_split_state=parent_split)
    if branch is None:
        raise ValueError("NRIR-11 fixed ResNet has no ambiguous ReLU branch")
    parent = optimize_native_alpha_beta_state(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_split_state=parent_split,
        policy=POLICY,
    )
    child_split = {name: value.clone() for name, value in parent_split.items()}
    child_split[branch.relu_input].reshape(-1)[branch.neuron_index] = 1
    optimizer_program = compile_native_alpha_beta_optimizer_program(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_split_state=child_split,
        policy=POLICY,
        program_id=QUERY_ID,
        warm_start=parent.state,
    )
    scheduled = execute_native_alpha_beta_optimizer_program(
        optimizer_program, module, input_spec, linear_spec_C=objective
    )
    legacy = optimize_native_alpha_beta_state(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_split_state=child_split,
        policy=POLICY,
        warm_start=parent.state,
    )
    selected = NativeAlphaBetaOptimizationResult(
        bounds=scheduled.bounds,
        state=scheduled.state,
        interval_env=optimizer_program.interval_env,
        relu_pre=optimizer_program.relu_pre,
        warm_start_decision=optimizer_program.warm_start_decision,
    )
    selected_compilation = compile_native_alpha_beta_state_query(
        module,
        input_spec,
        linear_spec_C=objective,
        optimization=selected,
        query_id=f"{QUERY_ID}:selected",
    )
    native_bounds, native_trace = execute_native_alpha_beta_state_query(
        selected_compilation,
        module,
        input_spec,
        linear_spec_C=objective,
        optimization=selected,
    )

    lower_legacy_diff = float(
        (scheduled.bounds.lower - legacy.bounds.lower).abs().max().item()
    )
    upper_legacy_diff = float(
        (scheduled.bounds.upper - legacy.bounds.upper).abs().max().item()
    )
    lower_native_diff = float(
        (scheduled.bounds.lower - native_bounds.lower).abs().max().item()
    )
    upper_native_diff = float(
        (scheduled.bounds.upper - native_bounds.upper).abs().max().item()
    )
    backward = [
        action for action in scheduled.trace.actions if action.kind.value == "backward"
    ]
    projections = [
        action
        for action in scheduled.trace.actions
        if action.kind.value == "project_state"
    ]
    program_hashes = optimizer_program.hashes()
    trace_payload = scheduled.trace.to_dict(program=optimizer_program)
    gates = {
        "fixed_resnet_source_is_digest_bound": True,
        "optimizer_plan_binds_nrir10_source_compiler_stack": bool(
            len(optimizer_program.source_compilation.hashes()) == 10
            and optimizer_program.plan.source_ir_hashes
            == tuple(sorted(optimizer_program.source_compilation.hashes().items()))
        ),
        "fixed_step_control_is_first_class_plan_task_schedule": bool(
            len(optimizer_program.task_module.tasks)
            == len(optimizer_program.schedule.actions)
            == len(scheduled.trace.actions)
            == 8
        ),
        "backward_update_projection_execute_in_schedule_order": bool(
            len(backward) == len(projections) == 1
            and backward[0].alpha_gradient_l1 is not None
            and backward[0].alpha_gradient_l1 > 0.0
            and backward[0].beta_gradient_l1 is not None
            and backward[0].beta_gradient_l1 > 0.0
            and projections[0].projection_applied
        ),
        "schedule_selected_state_matches_legacy_optimizer": bool(
            lower_legacy_diff == upper_legacy_diff == 0.0
            and scheduled.state.stable_hash() == legacy.state.stable_hash()
        ),
        "selected_state_reexecutes_through_native_compiler_stack": bool(
            lower_native_diff == upper_native_diff == 0.0
            and len(native_trace.events) == len(selected_compilation.task_module.tasks)
        ),
        "best_state_is_selected_from_evaluated_iterations": bool(
            len(scheduled.trace.evaluations) == 2
            and scheduled.trace.best_iteration_by_domain == (1,)
            and scheduled.state.stable_hash()
            == scheduled.trace.evaluations[1].state_hash
        ),
        "claims_remain_correctness_only": True,
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-11 gates failed: {gates}")

    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "property_status": "not_claimed",
        "claim_boundary": (
            "fixed-step alpha/beta optimizer Plan/Task/Schedule ownership, exact "
            "action/state-transition trace, fixed-ResNet legacy equivalence, and "
            "selected-state native compiler re-execution only; dynamic early stop, "
            "multi-node BaB integration, complete verdict, CUDA, latency, memory, "
            "and speedup remain unclaimed"
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
            "objective_hash": tensor_content_hash(objective),
            "objective_index": 0,
        },
        "optimizer_policy": POLICY.to_dict(),
        "branch": branch.to_dict(),
        "warm_start": (
            optimizer_program.warm_start_decision.to_dict()
            if optimizer_program.warm_start_decision is not None
            else None
        ),
        "initial_state": optimizer_program.initial_state.to_dict(),
        "selected_state": scheduled.state.to_dict(),
        "optimizer_program": {
            "plan": optimizer_program.plan.to_dict(),
            "task_module": optimizer_program.task_module.to_dict(
                plan=optimizer_program.plan
            ),
            "schedule": optimizer_program.schedule.to_dict(
                plan=optimizer_program.plan,
                task_module=optimizer_program.task_module,
            ),
            "hashes": program_hashes,
            "source_compiler_hashes": optimizer_program.source_compilation.hashes(),
            "selected_compiler_hashes": selected_compilation.hashes(),
            "execution_trace_hash": hashlib_sha256(trace_payload),
        },
        "execution_trace": trace_payload,
        "execution": {
            "scheduled_lower_hash": tensor_content_hash(scheduled.bounds.lower),
            "scheduled_upper_hash": tensor_content_hash(scheduled.bounds.upper),
            "legacy_lower_hash": tensor_content_hash(legacy.bounds.lower),
            "legacy_upper_hash": tensor_content_hash(legacy.bounds.upper),
            "native_lower_hash": tensor_content_hash(native_bounds.lower),
            "native_upper_hash": tensor_content_hash(native_bounds.upper),
            "legacy_lower_max_abs_diff": lower_legacy_diff,
            "legacy_upper_max_abs_diff": upper_legacy_diff,
            "native_lower_max_abs_diff": lower_native_diff,
            "native_upper_max_abs_diff": upper_native_diff,
            "optimizer_task_count": len(optimizer_program.task_module.tasks),
            "optimizer_schedule_action_count": len(optimizer_program.schedule.actions),
            "optimizer_trace_action_count": len(scheduled.trace.actions),
            "evaluated_iteration_count": len(scheduled.trace.evaluations),
            "backward_action_count": len(backward),
            "projection_action_count": len(projections),
            "alpha_gradient_l1": backward[0].alpha_gradient_l1,
            "beta_gradient_l1": backward[0].beta_gradient_l1,
            "best_iteration_by_domain": list(scheduled.trace.best_iteration_by_domain),
            "native_task_trace_event_count": len(native_trace.events),
        },
        "gates": gates,
        "limitations": [
            "the optimizer schedule is a fixed-step static unroll; dynamic early stop is pending",
            (
                "one parent and one monotonic child are evaluated; multi-node BaB "
                "integration is pending"
            ),
            "property_status remains not_claimed and this is not a complete verifier verdict",
            "CPU correctness only; no CUDA, latency, memory, or speedup evidence",
        ],
    }
    validate_native_optimizer_schedule_evidence(evidence)
    return evidence


def validate_native_optimizer_schedule_evidence(
    evidence: Mapping[str, Any],
) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("property_status") != "not_claimed"
        or "dynamic early stop" not in str(evidence.get("claim_boundary", ""))
        or "multi-node BaB integration" not in str(evidence.get("claim_boundary", ""))
    ):
        raise ValueError("NRIR-11 evidence header/claim boundary differs")
    source = _mapping(evidence.get("source"), "NRIR-11 source")
    if (
        source.get("model_sha256") != MODEL_SHA256
        or source.get("vnnlib_sha256") != VNNLIB_SHA256
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("abcrown_commit") != ABCROWN_COMMIT
        or source.get("objective_index") != 0
        or any(
            not _sha256(source.get(name))
            for name in (
                "native_ir_manifest_sha256",
                "native_ir_payload_sha256",
                "objective_hash",
            )
        )
    ):
        raise ValueError("NRIR-11 source identity differs")
    if _mapping(evidence.get("optimizer_policy"), "NRIR-11 policy") != POLICY.to_dict():
        raise ValueError("NRIR-11 optimizer policy differs")
    warm = _mapping(evidence.get("warm_start"), "NRIR-11 warm start")
    if (
        warm.get("kind") != "monotonic_split_refinement"
        or warm.get("alpha_initialization_allowed") is not True
        or warm.get("beta_initialization_allowed") is not True
        or warm.get("exact_state_reuse_allowed") is not False
    ):
        raise ValueError("NRIR-11 warm-start validity differs")
    _validate_program_and_trace(evidence)
    execution = _mapping(evidence.get("execution"), "NRIR-11 execution")
    if (
        execution.get("scheduled_lower_hash") != execution.get("legacy_lower_hash")
        or execution.get("scheduled_upper_hash") != execution.get("legacy_upper_hash")
        or execution.get("scheduled_lower_hash") != execution.get("native_lower_hash")
        or execution.get("scheduled_upper_hash") != execution.get("native_upper_hash")
        or execution.get("legacy_lower_max_abs_diff") != 0.0
        or execution.get("legacy_upper_max_abs_diff") != 0.0
        or execution.get("native_lower_max_abs_diff") != 0.0
        or execution.get("native_upper_max_abs_diff") != 0.0
        or execution.get("optimizer_task_count") != 8
        or execution.get("optimizer_schedule_action_count") != 8
        or execution.get("optimizer_trace_action_count") != 8
        or execution.get("evaluated_iteration_count") != 2
        or execution.get("backward_action_count") != 1
        or execution.get("projection_action_count") != 1
        or float(execution.get("alpha_gradient_l1", 0.0)) <= 0.0
        or float(execution.get("beta_gradient_l1", 0.0)) <= 0.0
        or execution.get("best_iteration_by_domain") != [1]
        or execution.get("native_task_trace_event_count") != 21
    ):
        raise ValueError("NRIR-11 execution equivalence/control evidence differs")
    gates = _mapping(evidence.get("gates"), "NRIR-11 gates")
    expected_gates = {
        "fixed_resnet_source_is_digest_bound",
        "optimizer_plan_binds_nrir10_source_compiler_stack",
        "fixed_step_control_is_first_class_plan_task_schedule",
        "backward_update_projection_execute_in_schedule_order",
        "schedule_selected_state_matches_legacy_optimizer",
        "selected_state_reexecutes_through_native_compiler_stack",
        "best_state_is_selected_from_evaluated_iterations",
        "claims_remain_correctness_only",
    }
    if set(gates) != expected_gates or any(
        value is not True for value in gates.values()
    ):
        raise ValueError("NRIR-11 gates differ")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 4:
        raise ValueError("NRIR-11 limitations differ")


def _generate(args: argparse.Namespace) -> None:
    evidence = build_native_optimizer_schedule_evidence(
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
        raise ValueError("NRIR-11 artifact manifest/header differs")
    stored = _mapping(artifact.get("evidence"), "NRIR-11 stored evidence")
    validate_native_optimizer_schedule_evidence(stored)
    if manifest.get("evidence_hash") != hashlib_sha256(stored):
        raise ValueError("NRIR-11 stored evidence hash differs")
    actual = build_native_optimizer_schedule_evidence(
        model=args.model, source_artifact_dir=args.source_artifact_dir
    )
    if stored != actual:
        raise ValueError("NRIR-11 replay differs from frozen evidence")
    print(_canonical_json({"status": "ok", "evidence_hash": hashlib_sha256(actual)}))


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
