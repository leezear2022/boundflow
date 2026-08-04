#!/usr/bin/env python3
"""Generate or replay NRIR-12 optimized ReLU-split BaB evidence."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,duplicate-code
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NATIVE_OPTIMIZED_RELU_SPLIT_BAB_COMPILER_VERSION,
    NATIVE_OPTIMIZED_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION,
    NATIVE_REEXECUTION_ATOL,
    compare_native_optimized_bab_states,
    execute_native_optimized_relu_split_bab,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
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

ARTIFACT_SCHEMA_VERSION = "boundflow.native-optimized-relu-split-bab-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.native-optimized-relu-split-bab-evidence/v1"
ARTIFACT_FILE = "optimized_bab.json"
MANIFEST_FILE = "manifest.json"
RUN_ID = "vnncomp21-resnet2b-prop0-native-ir12-optimized-bab"
MAX_NODES = 7
MAX_DEPTH = 8
EXPANSION_BATCH_SIZE = 2
PACKED_EVAL_BATCH_SIZE = 4
AVAILABLE_MEMORY_BYTES = 1 << 30
BOUNDS_ATOL = 2e-4
BOUNDS_RTOL = 2e-4
STATE_ATOL = 2e-6
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


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _trace_comparison(packed: Any, serial: Any) -> dict[str, object]:
    packed_by_id = {item.node.node_id: item for item in packed.trace.evaluations}
    serial_by_id = {item.node.node_id: item for item in serial.trace.evaluations}
    if tuple(packed_by_id) != tuple(serial_by_id):
        raise ValueError("NRIR-12 packed/serial node order differs")
    lower_diff = max(
        abs(packed_by_id[node_id].lower - serial_by_id[node_id].lower)
        for node_id in packed_by_id
    )
    upper_diff = max(
        abs(packed_by_id[node_id].upper - serial_by_id[node_id].upper)
        for node_id in packed_by_id
    )
    lower_scale = max(
        1.0,
        max(abs(item.lower) for item in packed_by_id.values()),
        max(abs(item.lower) for item in serial_by_id.values()),
    )
    upper_scale = max(
        1.0,
        max(abs(item.upper) for item in packed_by_id.values()),
        max(abs(item.upper) for item in serial_by_id.values()),
    )
    state = compare_native_optimized_bab_states(packed, serial)
    return {
        "node_ids_same": True,
        "logical_queue_signature_same": (
            packed.trace.logical_queue_signature()
            == serial.trace.logical_queue_signature()
        ),
        "split_state_hashes_same": all(
            packed_by_id[node_id].node.split_state_hash
            == serial_by_id[node_id].node.split_state_hash
            for node_id in packed_by_id
        ),
        "lower_max_abs_diff": lower_diff,
        "upper_max_abs_diff": upper_diff,
        "lower_allclose": lower_diff <= BOUNDS_ATOL + BOUNDS_RTOL * lower_scale,
        "upper_allclose": upper_diff <= BOUNDS_ATOL + BOUNDS_RTOL * upper_scale,
        "state": state,
    }


def build_native_optimized_bab_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-12 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    if tuple(op.op_type for op in module.get_entry_task().ops) != EXPECTED_PRIMAL_OPS:
        raise ValueError("NRIR-12 primal topology differs")
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    objective = tensors["linear_spec_c"][:, 0:1].contiguous()

    def execute(batch_size: int):
        return execute_native_optimized_relu_split_bab(
            module,
            input_spec,
            linear_spec_C=objective,
            run_id=RUN_ID,
            config=NativeReluSplitBabConfig(
                max_nodes=MAX_NODES,
                max_depth=MAX_DEPTH,
                expansion_batch_size=EXPANSION_BATCH_SIZE,
                max_eval_batch_size=batch_size,
                threshold=0.0,
                available_memory_bytes=AVAILABLE_MEMORY_BYTES,
                memory_budget_bytes=AVAILABLE_MEMORY_BYTES,
            ),
            optimizer_policy=POLICY,
        )

    packed = execute(PACKED_EVAL_BATCH_SIZE)
    serial = execute(1)
    comparison = _trace_comparison(packed, serial)
    packed_trace = packed.trace
    serial_trace = serial.trace
    packed_child_stacks = packed_trace.native_stacks[1:]
    serial_child_stacks = serial_trace.native_stacks[1:]
    packed_max_lower_native_diff = max(
        stack.selected_native_lower_max_abs_diff for stack in packed_trace.native_stacks
    )
    packed_max_upper_native_diff = max(
        stack.selected_native_upper_max_abs_diff for stack in packed_trace.native_stacks
    )
    gates = {
        "fixed_resnet_and_property_source_are_digest_bound": True,
        "every_node_executes_optimizer_and_selected_native_ir_stacks": bool(
            all(
                stack.optimizer_action_count == 8
                and stack.optimizer_evaluation_count == 2
                and stack.optimizer_backward_count == 1
                and stack.optimizer_projection_count == 1
                and stack.native_task_count
                == stack.native_schedule_launch_count
                == stack.native_task_trace_event_count
                == 21
                for stack in (*packed_trace.native_stacks, *serial_trace.native_stacks)
            )
        ),
        "every_child_uses_monotonic_parent_initialization_only": bool(
            all(
                item.warm_start_kind == "monotonic_split_refinement"
                and item.parent_selected_state_hash is not None
                and not item.parent_state_consumed_as_exact
                for item in (
                    *packed_trace.evaluations[1:],
                    *serial_trace.evaluations[1:],
                )
            )
            and all(
                stack.warm_start_kind == "monotonic_split_refinement"
                and stack.warm_source_state_hash is not None
                for stack in (*packed_child_stacks, *serial_child_stacks)
            )
        ),
        "active_split_batches_execute_nonzero_beta_gradients": bool(
            all(stack.beta_gradient_l1 > 0.0 for stack in packed_child_stacks)
            and all(stack.beta_gradient_l1 > 0.0 for stack in serial_child_stacks)
        ),
        "selected_states_reexecute_through_native_compiler": bool(
            packed_max_lower_native_diff <= NATIVE_REEXECUTION_ATOL
            and packed_max_upper_native_diff <= NATIVE_REEXECUTION_ATOL
            and all(
                stack.selected_native_lower_max_abs_diff <= NATIVE_REEXECUTION_ATOL
                and stack.selected_native_upper_max_abs_diff <= NATIVE_REEXECUTION_ATOL
                for stack in serial_trace.native_stacks
            )
        ),
        "best_first_queue_forms_three_generations_and_four_frontier_nodes": bool(
            len(packed_trace.evaluations) == 7
            and len(packed_trace.decisions) == 3
            and len(packed_trace.final_frontier_node_ids) == 4
            and max(item.node.depth for item in packed_trace.evaluations) == 2
        ),
        "packed_three_stacks_replace_seven_serial_stacks": bool(
            packed_trace.native_stack_count == 3
            and serial_trace.native_stack_count == 7
        ),
        "packed_serial_queue_bounds_and_state_tensors_match": bool(
            comparison["logical_queue_signature_same"] is True
            and comparison["split_state_hashes_same"] is True
            and comparison["lower_allclose"] is True
            and comparison["upper_allclose"] is True
            and _mapping(comparison["state"], "NRIR-12 state comparison").get(
                "split_tensors_exact"
            )
            is True
            and float(
                _mapping(comparison["state"], "NRIR-12 state comparison").get(
                    "alpha_max_abs_diff", math.inf
                )
            )
            <= STATE_ATOL
            and float(
                _mapping(comparison["state"], "NRIR-12 state comparison").get(
                    "beta_max_abs_diff", math.inf
                )
            )
            <= STATE_ATOL
        ),
        "bounded_run_remains_unknown_and_correctness_only": bool(
            packed_trace.status == serial_trace.status == "budget_exhausted"
            and packed_trace.property_status
            == serial_trace.property_status
            == "not_claimed"
            and not packed_trace.performance_claimed
            and not serial_trace.performance_claimed
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-12 gates failed: {gates}; comparison={comparison}")

    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "property_status": "not_claimed",
        "compiler_version": NATIVE_OPTIMIZED_RELU_SPLIT_BAB_COMPILER_VERSION,
        "claim_boundary": (
            "fixed-ResNet ReLU-split best-first bounded queue whose every node batch "
            "executes fixed-step alpha/beta optimizer Plan/Task/Schedule and selected-"
            "state native Bound/Plan/Task/Schedule, with monotonic parent warm "
            "initialization and packed/serial correctness only; not dynamic early stop, "
            "complete property verdict, latency, memory, CUDA, or speedup evidence"
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
            "objective_hash": tensor_content_hash(objective[0]),
            "objective_index": 0,
        },
        "optimizer_policy": POLICY.to_dict(),
        "packed": {
            "trace": packed_trace.to_dict(),
            "trace_hash": packed_trace.stable_hash(),
        },
        "serial": {
            "trace": serial_trace.to_dict(),
            "trace_hash": serial_trace.stable_hash(),
        },
        "comparison": comparison,
        "gates": gates,
        "limitations": [
            "fixed-step optimizer schedule only; dynamic early stop is not implemented",
            "seven-node bounded run ends budget_exhausted with property_status=not_claimed",
            "parent selected state is warm initialization only and never child exact state",
            "CPU packed/serial state tensors use tolerance; exact batch-layout hashes may differ",
            "three versus seven stacks is a mechanism count, not latency or speedup evidence",
        ],
    }
    validate_native_optimized_bab_evidence(evidence)
    return evidence


def _validate_trace(
    trace: Mapping[str, Any],
    *,
    mode: str,
    expected_stack_count: int,
    expected_batch_size: int,
) -> None:
    evaluations = _list(trace.get("evaluations"), f"NRIR-12 {mode} evaluations")
    decisions = _list(trace.get("decisions"), f"NRIR-12 {mode} decisions")
    frontier = _list(trace.get("final_frontier_node_ids"), f"NRIR-12 {mode} frontier")
    stacks = _list(trace.get("native_stacks"), f"NRIR-12 {mode} stacks")
    config = _mapping(trace.get("config"), f"NRIR-12 {mode} config")
    if (
        trace.get("schema_version")
        != NATIVE_OPTIMIZED_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION
        or trace.get("compiler_version")
        != NATIVE_OPTIMIZED_RELU_SPLIT_BAB_COMPILER_VERSION
        or trace.get("run_id") != RUN_ID
        or trace.get("status") != "budget_exhausted"
        or trace.get("termination_reason") != "node_budget_exhausted"
        or trace.get("performance_claimed") is not False
        or trace.get("property_status") != "not_claimed"
        or trace.get("optimizer_policy") != POLICY.to_dict()
        or trace.get("native_stack_count") != expected_stack_count
        or len(stacks) != expected_stack_count
        or len(evaluations) != 7
        or len(decisions) != 3
        or len(frontier) != 4
        or config.get("max_nodes") != MAX_NODES
        or config.get("max_depth") != MAX_DEPTH
        or config.get("expansion_batch_size") != EXPANSION_BATCH_SIZE
        or config.get("max_eval_batch_size") != expected_batch_size
    ):
        raise ValueError(f"NRIR-12 {mode} bounded queue shape differs")
    by_id: dict[str, Mapping[str, Any]] = {}
    batches: dict[str, list[tuple[int, str]]] = {}
    for position, value in enumerate(evaluations):
        evaluation = _mapping(value, f"NRIR-12 {mode} evaluation")
        node = _mapping(evaluation.get("node"), f"NRIR-12 {mode} node")
        node_id = str(node.get("node_id", ""))
        parent_id = node.get("parent_node_id")
        depth = node.get("depth")
        if (
            not node_id
            or node_id in by_id
            or not isinstance(depth, int)
            or depth < 0
            or not _sha256(node.get("split_state_hash"))
            or not _sha256(evaluation.get("selected_state_hash"))
            or not _sha256(evaluation.get("optimizer_execution_trace_hash"))
            or evaluation.get("parent_state_consumed_as_exact") is not False
            or not all(
                isinstance(evaluation.get(name), (int, float))
                and math.isfinite(float(evaluation[name]))
                for name in ("lower", "upper", "priority")
            )
        ):
            raise ValueError(f"NRIR-12 {mode} evaluation identity differs")
        if position == 0:
            if (
                depth != 0
                or parent_id is not None
                or evaluation.get("warm_start_kind") != "none"
                or evaluation.get("parent_selected_state_hash") is not None
            ):
                raise ValueError(f"NRIR-12 {mode} root warm identity differs")
        else:
            parent = by_id.get(str(parent_id))
            if (
                parent is None
                or depth != int(_mapping(parent["node"], "parent")["depth"]) + 1
                or evaluation.get("warm_start_kind") != "monotonic_split_refinement"
                or evaluation.get("parent_selected_state_hash")
                != parent.get("selected_state_hash")
            ):
                raise ValueError(f"NRIR-12 {mode} parent warm-state link differs")
        batch_id = str(evaluation.get("eval_batch_id", ""))
        batch_position = evaluation.get("eval_batch_position")
        if not batch_id or not isinstance(batch_position, int):
            raise ValueError(f"NRIR-12 {mode} batch identity differs")
        by_id[node_id] = evaluation
        batches.setdefault(batch_id, []).append((batch_position, node_id))
    if any(
        [position for position, _node_id in values] != list(range(len(values)))
        for values in batches.values()
    ):
        raise ValueError(f"NRIR-12 {mode} batch positions differ")

    stack_ids: set[str] = set()
    for value in stacks:
        stack = _mapping(value, f"NRIR-12 {mode} stack")
        stack_id = str(stack.get("stack_id", ""))
        node_ids = tuple(
            str(item)
            for item in _list(stack.get("node_ids"), f"NRIR-12 {mode} stack nodes")
        )
        expected_nodes = tuple(
            node_id for _position, node_id in batches.get(stack_id, [])
        )
        optimizer_hashes = _mapping(
            stack.get("optimizer_ir_hashes"), "NRIR-12 optimizer hashes"
        )
        native_hashes = _mapping(stack.get("native_ir_hashes"), "NRIR-12 native hashes")
        if (
            not stack_id
            or stack_id in stack_ids
            or node_ids != expected_nodes
            or stack.get("domain_batch_size") != len(node_ids)
            or stack.get("optimizer_action_count") != 8
            or stack.get("optimizer_evaluation_count") != 2
            or stack.get("optimizer_backward_count") != 1
            or stack.get("optimizer_projection_count") != 1
            or stack.get("native_task_count") != 21
            or stack.get("native_schedule_launch_count") != 21
            or stack.get("native_task_trace_event_count") != 21
            or float(stack.get("selected_native_lower_max_abs_diff", math.inf))
            > NATIVE_REEXECUTION_ATOL
            or float(stack.get("selected_native_upper_max_abs_diff", math.inf))
            > NATIVE_REEXECUTION_ATOL
            or set(optimizer_hashes)
            != {
                "optimizer_plan_hash",
                "optimizer_task_module_hash",
                "optimizer_schedule_hash",
            }
            or set(native_hashes)
            != {
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
            or any(not _sha256(item) for item in optimizer_hashes.values())
            or any(not _sha256(item) for item in native_hashes.values())
        ):
            raise ValueError(f"NRIR-12 {mode} optimizer/native stack differs")
        stack_ids.add(stack_id)
        for node_id in node_ids:
            evaluation = by_id[node_id]
            if (
                evaluation.get("optimizer_ir_hashes") != optimizer_hashes
                or evaluation.get("native_ir_hashes") != native_hashes
                or evaluation.get("optimizer_execution_trace_hash")
                != stack.get("optimizer_execution_trace_hash")
            ):
                raise ValueError(f"NRIR-12 {mode} stack/node hashes differ")
    if stack_ids != set(batches):
        raise ValueError(f"NRIR-12 {mode} stack coverage differs")

    decision_nodes: set[str] = set()
    children: set[str] = set()
    for index, value in enumerate(decisions):
        decision = _mapping(value, f"NRIR-12 {mode} decision")
        node_id = str(decision.get("node_id", ""))
        child_ids = tuple(
            str(item)
            for item in _list(
                decision.get("child_node_ids"), f"NRIR-12 {mode} children"
            )
        )
        if (
            decision.get("decision_index") != index
            or node_id not in by_id
            or node_id in decision_nodes
            or decision.get("kind") != "expand"
            or len(child_ids) != 2
        ):
            raise ValueError(f"NRIR-12 {mode} decision identity differs")
        decision_nodes.add(node_id)
        for child_id in child_ids:
            child = by_id.get(child_id)
            if (
                child is None
                or _mapping(child["node"], "child").get("parent_node_id") != node_id
            ):
                raise ValueError(f"NRIR-12 {mode} decision child link differs")
            children.add(child_id)
    frontier_ids = {str(value) for value in frontier}
    root_id = str(_mapping(evaluations[0], "root")["node"]["node_id"])
    if (
        frontier_ids & decision_nodes
        or frontier_ids | decision_nodes != set(by_id)
        or children != set(by_id) - {root_id}
    ):
        raise ValueError(f"NRIR-12 {mode} frontier accounting differs")


def validate_native_optimized_bab_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("property_status") != "not_claimed"
        or evidence.get("compiler_version")
        != NATIVE_OPTIMIZED_RELU_SPLIT_BAB_COMPILER_VERSION
    ):
        raise ValueError("NRIR-12 evidence header differs")
    claim = str(evidence.get("claim_boundary", ""))
    for phrase in (
        "every node batch",
        "monotonic parent warm initialization",
        "not dynamic early stop",
        "complete property verdict",
        "speedup",
    ):
        if phrase not in claim:
            raise ValueError("NRIR-12 claim boundary omits a hard limitation")
    source = _mapping(evidence.get("source"), "NRIR-12 source")
    if (
        source.get("model_sha256") != MODEL_SHA256
        or source.get("vnnlib_sha256") != VNNLIB_SHA256
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("abcrown_commit") != ABCROWN_COMMIT
        or source.get("objective_index") != 0
    ):
        raise ValueError("NRIR-12 source identity differs")
    if _mapping(evidence.get("optimizer_policy"), "NRIR-12 policy") != POLICY.to_dict():
        raise ValueError("NRIR-12 optimizer policy differs")
    for mode, stack_count, batch_size in (
        ("packed", 3, PACKED_EVAL_BATCH_SIZE),
        ("serial", 7, 1),
    ):
        section = _mapping(evidence.get(mode), f"NRIR-12 {mode}")
        trace = _mapping(section.get("trace"), f"NRIR-12 {mode} trace")
        if section.get("trace_hash") != hashlib_sha256(trace):
            raise ValueError(f"NRIR-12 {mode} trace hash differs")
        _validate_trace(
            trace,
            mode=mode,
            expected_stack_count=stack_count,
            expected_batch_size=batch_size,
        )
    comparison = _mapping(evidence.get("comparison"), "NRIR-12 comparison")
    state = _mapping(comparison.get("state"), "NRIR-12 state comparison")
    if (
        comparison.get("node_ids_same") is not True
        or comparison.get("logical_queue_signature_same") is not True
        or comparison.get("split_state_hashes_same") is not True
        or comparison.get("lower_allclose") is not True
        or comparison.get("upper_allclose") is not True
        or float(comparison.get("lower_max_abs_diff", math.inf)) > BOUNDS_ATOL
        or float(comparison.get("upper_max_abs_diff", math.inf)) > BOUNDS_ATOL
        or state.get("split_tensors_exact") is not True
        or state.get("stable_scope_fields_equal") is not True
        or float(state.get("alpha_max_abs_diff", math.inf)) > STATE_ATOL
        or float(state.get("beta_max_abs_diff", math.inf)) > STATE_ATOL
    ):
        raise ValueError("NRIR-12 packed/serial numeric comparison differs")
    gates = _mapping(evidence.get("gates"), "NRIR-12 gates")
    expected_gates = {
        "fixed_resnet_and_property_source_are_digest_bound",
        "every_node_executes_optimizer_and_selected_native_ir_stacks",
        "every_child_uses_monotonic_parent_initialization_only",
        "active_split_batches_execute_nonzero_beta_gradients",
        "selected_states_reexecute_through_native_compiler",
        "best_first_queue_forms_three_generations_and_four_frontier_nodes",
        "packed_three_stacks_replace_seven_serial_stacks",
        "packed_serial_queue_bounds_and_state_tensors_match",
        "bounded_run_remains_unknown_and_correctness_only",
    }
    if set(gates) != expected_gates or any(
        value is not True for value in gates.values()
    ):
        raise ValueError("NRIR-12 gates differ")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 5:
        raise ValueError("NRIR-12 limitations differ")


def _generate(args: argparse.Namespace) -> None:
    evidence = build_native_optimized_bab_evidence(
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
        raise ValueError("NRIR-12 artifact manifest/header differs")
    stored = _mapping(artifact.get("evidence"), "NRIR-12 stored evidence")
    validate_native_optimized_bab_evidence(stored)
    if manifest.get("evidence_hash") != hashlib_sha256(stored):
        raise ValueError("NRIR-12 stored evidence hash differs")
    actual = build_native_optimized_bab_evidence(
        model=args.model, source_artifact_dir=args.source_artifact_dir
    )
    if stored != actual:
        raise ValueError("NRIR-12 replay differs from frozen evidence")
    print(_canonical_json({"status": "ok", "evidence_hash": hashlib_sha256(actual)}))


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
