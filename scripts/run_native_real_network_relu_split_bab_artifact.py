#!/usr/bin/env python3
"""Generate or replay NRIR-9 fixed-ResNet native ReLU-split BaB evidence."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,duplicate-code
# pylint: disable=missing-function-docstring,line-too-long

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
from boundflow.runtime.native_relu_split_bab_runtime import (
    NATIVE_RELU_SPLIT_BAB_COMPILER_VERSION,
    NATIVE_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION,
    NativeReluSplitBabConfig,
    NativeReluSplitBabTrace,
    run_native_relu_split_bab,
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

RELU_SPLIT_BAB_ARTIFACT_SCHEMA_VERSION = (
    "boundflow.native-real-network-relu-split-bab-artifact/v1"
)
RELU_SPLIT_BAB_EVIDENCE_SCHEMA_VERSION = (
    "boundflow.native-real-network-relu-split-bab-evidence/v1"
)
RUN_ID = "vnncomp21-resnet2b-prop0-native-ir9-relu-split-bab"
AVAILABLE_MEMORY_BYTES = 1 << 30
MAX_NODES = 7
MAX_DEPTH = 8
EXPANSION_BATCH_SIZE = 2
PACKED_EVAL_BATCH_SIZE = 4
ATOL = 2e-4
RTOL = 2e-4
ARTIFACT_FILES = ("relu_split_bab.json",)
MANIFEST_FILE = "manifest.json"


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


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _trace_comparison(
    packed: NativeReluSplitBabTrace,
    serial: NativeReluSplitBabTrace,
) -> dict[str, object]:
    packed_by_id = {item.node.node_id: item for item in packed.evaluations}
    serial_by_id = {item.node.node_id: item for item in serial.evaluations}
    if set(packed_by_id) != set(serial_by_id):
        raise ValueError("NRIR-9 packed/serial node identities differ")
    lower_difference = max(
        abs(packed_by_id[node_id].lower - serial_by_id[node_id].lower)
        for node_id in packed_by_id
    )
    upper_difference = max(
        abs(packed_by_id[node_id].upper - serial_by_id[node_id].upper)
        for node_id in packed_by_id
    )
    lower_scale = max(abs(item.lower) for item in serial.evaluations)
    upper_scale = max(abs(item.upper) for item in serial.evaluations)
    lower_allclose = lower_difference <= ATOL + RTOL * lower_scale
    upper_allclose = upper_difference <= ATOL + RTOL * upper_scale
    logical_same = packed.logical_queue_signature() == serial.logical_queue_signature()
    exact_states_same = all(
        packed_by_id[node_id].exact_state_hash == serial_by_id[node_id].exact_state_hash
        for node_id in packed_by_id
    )
    split_states_same = all(
        packed_by_id[node_id].node.split_state_hash
        == serial_by_id[node_id].node.split_state_hash
        for node_id in packed_by_id
    )
    return {
        "node_ids_same": True,
        "logical_queue_signature_same": logical_same,
        "exact_node_state_hashes_same": exact_states_same,
        "split_state_hashes_same": split_states_same,
        "lower_allclose": lower_allclose,
        "upper_allclose": upper_allclose,
        "lower_max_abs_diff": lower_difference,
        "upper_max_abs_diff": upper_difference,
        "atol": ATOL,
        "rtol": RTOL,
    }


def build_relu_split_bab_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    """Run packed-4 and same-policy serial-1 bounded queues on fixed ResNet."""

    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-9 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    linear_spec = tensors["linear_spec_c"]
    if tuple(linear_spec.shape) != (1, 9, 10):
        raise ValueError("NRIR-9 frozen objective layout differs")
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    legacy_module = plan_interval_ibp_v0(program)
    if tuple(op.op_type for op in legacy_module.get_entry_task().ops) != (
        EXPECTED_PRIMAL_OPS
    ):
        raise ValueError("NRIR-9 primal topology differs")
    root_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    objective = linear_spec[:, 0:1].contiguous()
    packed = run_native_relu_split_bab(
        legacy_module,
        root_spec,
        linear_spec_C=objective,
        run_id=RUN_ID,
        config=NativeReluSplitBabConfig(
            max_nodes=MAX_NODES,
            max_depth=MAX_DEPTH,
            expansion_batch_size=EXPANSION_BATCH_SIZE,
            max_eval_batch_size=PACKED_EVAL_BATCH_SIZE,
            threshold=0.0,
            available_memory_bytes=AVAILABLE_MEMORY_BYTES,
            memory_budget_bytes=AVAILABLE_MEMORY_BYTES,
        ),
    )
    serial = run_native_relu_split_bab(
        legacy_module,
        root_spec,
        linear_spec_C=objective,
        run_id=RUN_ID,
        config=NativeReluSplitBabConfig(
            max_nodes=MAX_NODES,
            max_depth=MAX_DEPTH,
            expansion_batch_size=EXPANSION_BATCH_SIZE,
            max_eval_batch_size=1,
            threshold=0.0,
            available_memory_bytes=AVAILABLE_MEMORY_BYTES,
            memory_budget_bytes=AVAILABLE_MEMORY_BYTES,
        ),
    )
    comparison = _trace_comparison(packed, serial)
    packed_stack_contract = all(
        stack.bound_split_input_count == 6
        and stack.bound_split_relu_op_count == 6
        and stack.bound_local_forward_relu_op_count == 6
        and stack.source_plan_split_state_present
        and stack.execution_plan_split_state_present
        and stack.split_capability_count >= 2
        and stack.local_forward_provenance_count >= 2
        and stack.schedule_launch_count == stack.task_count
        for stack in packed.native_stacks
    )
    serial_stack_contract = all(
        stack.bound_split_input_count == 6
        and stack.bound_split_relu_op_count == 6
        and stack.bound_local_forward_relu_op_count == 6
        for stack in serial.native_stacks
    )
    gates = {
        "fixed_resnet_and_property_source_are_digest_bound": True,
        "all_six_relu_splits_are_first_class_bound_inputs": bool(
            packed_stack_contract and serial_stack_contract
        ),
        "split_state_reaches_plan_task_and_schedule_ir": bool(
            packed_stack_contract
            and all(stack.task_count == 21 for stack in packed.native_stacks)
        ),
        "every_child_recomputes_exact_state_without_parent_reuse": bool(
            all(
                item.parent_state_consumed_as_exact is False
                and (
                    item.parent_exact_state_hash is None
                    or item.parent_exact_state_hash != item.exact_state_hash
                )
                for item in packed.evaluations
            )
        ),
        "best_first_queue_forms_three_generations_and_four_frontier_nodes": bool(
            len(packed.evaluations) == 7
            and len(packed.decisions) == 3
            and len(packed.final_frontier_node_ids) == 4
            and max(item.node.depth for item in packed.evaluations) == 2
        ),
        "packed_three_stacks_replace_seven_serial_stacks": bool(
            packed.native_stack_count == 3 and serial.native_stack_count == 7
        ),
        "packed_and_serial_bounds_and_queue_semantics_match": bool(
            all(
                comparison[name] is True
                for name in (
                    "node_ids_same",
                    "logical_queue_signature_same",
                    "split_state_hashes_same",
                    "lower_allclose",
                    "upper_allclose",
                )
            )
        ),
        "bounded_run_reports_unknown_budget_exhausted_without_performance_claim": bool(
            packed.status == serial.status == "budget_exhausted"
            and packed.property_status == serial.property_status == "not_claimed"
            and packed.performance_claimed is False
            and serial.performance_claimed is False
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-9 gates failed: {gates}")
    evidence: dict[str, object] = {
        "schema_version": RELU_SPLIT_BAB_EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "property_status": "not_claimed",
        "compiler_version": NATIVE_RELU_SPLIT_BAB_COMPILER_VERSION,
        "claim_boundary": (
            "fixed-ResNet first-class ReLU split inputs, local split-constrained IBP, "
            "plain-CROWN native Bound/Plan/Task/Schedule execution, deterministic "
            "best-first bounded queue, pruning/expansion control flow, and packed/serial "
            "correctness only; not alpha/beta optimization, complete property verdict, "
            "latency, memory, CUDA, or speedup evidence"
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
        "packed": {
            "trace": packed.to_dict(),
            "trace_hash": packed.stable_hash(),
        },
        "serial": {
            "trace": serial.to_dict(),
            "trace_hash": serial.stable_hash(),
        },
        "comparison": comparison,
        "gates": gates,
        "limitations": [
            "plain CROWN with exact split-constrained local IBP; no alpha/beta optimization",
            "seven-node bounded run ends budget_exhausted with property_status=not_claimed",
            "parent exact bounds are never child exact state; only discrete splits inherit",
            "three versus seven native stacks is a mechanism count, not a timing claim",
            "CPU correctness/control-flow evidence only; no latency, memory, CUDA, or speedup claim",
        ],
    }
    validate_relu_split_bab_evidence(evidence)
    return evidence


def _validate_frozen_queue_trace(
    trace: Mapping[str, Any],
    *,
    mode: str,
    expected_stack_count: int,
    expected_batch_size: int,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, tuple[object, ...]],
]:
    """Independently close stored queue lineage, stacks, and exact-state links."""

    evaluations = _list(trace.get("evaluations"), f"NRIR-9 {mode} evaluations")
    decisions = _list(trace.get("decisions"), f"NRIR-9 {mode} decisions")
    frontier = _list(trace.get("final_frontier_node_ids"), f"NRIR-9 {mode} frontier")
    stacks = _list(trace.get("native_stacks"), f"NRIR-9 {mode} stacks")
    config = _mapping(trace.get("config"), f"NRIR-9 {mode} config")
    if (
        trace.get("schema_version") != NATIVE_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION
        or trace.get("compiler_version") != NATIVE_RELU_SPLIT_BAB_COMPILER_VERSION
        or trace.get("run_id") != RUN_ID
        or trace.get("status") != "budget_exhausted"
        or trace.get("termination_reason") != "node_budget_exhausted"
        or trace.get("performance_claimed") is not False
        or trace.get("property_status") != "not_claimed"
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
        raise ValueError(f"NRIR-9 {mode} bounded queue shape differs")
    by_id: dict[str, Mapping[str, Any]] = {}
    positions: dict[str, int] = {}
    batches: dict[str, list[tuple[int, str]]] = {}
    root_id = f"{RUN_ID}:n000000"
    for position, evaluation_value in enumerate(evaluations):
        evaluation = _mapping(evaluation_value, f"NRIR-9 {mode} evaluation")
        node = _mapping(evaluation.get("node"), f"NRIR-9 {mode} node")
        node_id = str(node.get("node_id", ""))
        parent_id = node.get("parent_node_id")
        depth = node.get("depth")
        lower = evaluation.get("lower")
        upper = evaluation.get("upper")
        priority = evaluation.get("priority")
        numeric_values = (lower, upper, priority)
        numeric_values_valid = all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in numeric_values
        )
        exact_hash = str(evaluation.get("exact_state_hash", ""))
        split_hash = str(node.get("split_state_hash", ""))
        if (
            not node_id
            or node_id in by_id
            or not isinstance(depth, int)
            or depth < 0
            or not numeric_values_valid
            or len(exact_hash) != 64
            or len(split_hash) != 64
            or evaluation.get("parent_state_consumed_as_exact") is not False
            or evaluation.get("parent_state_validity")
            != "discrete_split_inheritance_only"
        ):
            raise ValueError(f"NRIR-9 {mode} node/evaluation identity differs")
        assert isinstance(lower, (int, float))
        assert isinstance(upper, (int, float))
        if float(lower) > float(upper):
            raise ValueError(f"NRIR-9 {mode} lower/upper order differs")
        if position == 0:
            if (
                node_id != root_id
                or depth != 0
                or parent_id is not None
                or node.get("branch_value") != 0
                or evaluation.get("parent_exact_state_hash") is not None
            ):
                raise ValueError(f"NRIR-9 {mode} root identity differs")
        else:
            parent = by_id.get(str(parent_id))
            if (
                parent is None
                or depth != int(parent["node"]["depth"]) + 1
                or node.get("branch_value") not in (-1, 1)
                or not node.get("branch_relu_input")
                or not isinstance(node.get("branch_neuron_index"), int)
                or evaluation.get("parent_exact_state_hash")
                != parent.get("exact_state_hash")
                or evaluation.get("parent_exact_state_hash") == exact_hash
            ):
                raise ValueError(f"NRIR-9 {mode} parent/exact-state link differs")
        batch_id = str(evaluation.get("eval_batch_id", ""))
        batch_position = evaluation.get("eval_batch_position")
        if not batch_id or not isinstance(batch_position, int) or batch_position < 0:
            raise ValueError(f"NRIR-9 {mode} evaluation batch identity differs")
        by_id[node_id] = evaluation
        positions[node_id] = position
        batches.setdefault(batch_id, []).append((batch_position, node_id))
    if any(
        [item[0] for item in items] != list(range(len(items)))
        for items in batches.values()
    ):
        raise ValueError(f"NRIR-9 {mode} evaluation batch positions differ")
    decision_nodes: set[str] = set()
    expanded_children: set[str] = set()
    decision_signature: dict[str, tuple[object, ...]] = {}
    for index, decision_value in enumerate(decisions):
        decision = _mapping(decision_value, f"NRIR-9 {mode} decision")
        node_id = str(decision.get("node_id", ""))
        kind = decision.get("kind")
        children = _list(
            decision.get("child_node_ids"), f"NRIR-9 {mode} decision children"
        )
        if (
            decision.get("decision_index") != index
            or node_id not in by_id
            or node_id in decision_nodes
            or kind not in {"prune", "expand", "terminal"}
            or not decision.get("reason")
        ):
            raise ValueError(f"NRIR-9 {mode} decision identity differs")
        decision_nodes.add(node_id)
        if kind == "expand":
            branch = _mapping(decision.get("branch_candidate"), f"NRIR-9 {mode} branch")
            if len(children) != 2:
                raise ValueError(f"NRIR-9 {mode} expansion arity differs")
            for child_index, child_id_value in enumerate(children):
                child_id = str(child_id_value)
                child = by_id.get(child_id)
                if child is None:
                    raise ValueError(f"NRIR-9 {mode} expansion child is absent")
                child_node = _mapping(child.get("node"), f"NRIR-9 {mode} child")
                if (
                    child_node.get("parent_node_id") != node_id
                    or child_node.get("branch_relu_input") != branch.get("relu_input")
                    or child_node.get("branch_neuron_index")
                    != branch.get("neuron_index")
                    or child_node.get("branch_value") != (-1 if child_index == 0 else 1)
                    or positions[child_id] <= positions[node_id]
                ):
                    raise ValueError(f"NRIR-9 {mode} expansion branch differs")
                expanded_children.add(child_id)
            decision_signature[node_id] = (
                kind,
                decision.get("reason"),
                branch.get("relu_input"),
                branch.get("neuron_index"),
                tuple(children),
            )
        else:
            if children or decision.get("branch_candidate") is not None:
                raise ValueError(f"NRIR-9 {mode} terminal/prune payload differs")
            decision_signature[node_id] = (kind, decision.get("reason"))
    frontier_ids = tuple(str(value) for value in frontier)
    if (
        len(frontier_ids) != len(set(frontier_ids))
        or not set(frontier_ids) <= set(by_id)
        or set(frontier_ids) & decision_nodes
        or set(frontier_ids) | decision_nodes != set(by_id)
        or expanded_children != set(by_id) - {root_id}
    ):
        raise ValueError(f"NRIR-9 {mode} frontier/accounting differs")
    stack_ids: set[str] = set()
    for stack_value in stacks:
        stack = _mapping(stack_value, f"NRIR-9 {mode} stack")
        stack_id = str(stack.get("stack_id", ""))
        node_ids = tuple(
            str(value)
            for value in _list(stack.get("node_ids"), f"NRIR-9 {mode} stack nodes")
        )
        expected_nodes = tuple(
            node_id for _position, node_id in batches.get(stack_id, [])
        )
        ir_hashes = _mapping(
            stack.get("native_ir_hashes"), f"NRIR-9 {mode} stack hashes"
        )
        if (
            not stack_id
            or stack_id in stack_ids
            or node_ids != expected_nodes
            or stack.get("domain_batch_size") != len(node_ids)
            or stack.get("bound_split_input_count") != 6
            or stack.get("bound_split_relu_op_count") != 6
            or stack.get("bound_local_forward_relu_op_count") != 6
            or stack.get("source_plan_split_state_present") is not True
            or stack.get("execution_plan_split_state_present") is not True
            or int(stack.get("split_capability_count", 0)) < 2
            or int(stack.get("local_forward_provenance_count", 0)) < 2
            or stack.get("schedule_launch_count") != stack.get("task_count")
            or set(ir_hashes)
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
            or any(len(str(value)) != 64 for value in ir_hashes.values())
        ):
            raise ValueError(f"NRIR-9 {mode} native IR stack contract differs")
        stack_ids.add(stack_id)
        for node_id in node_ids:
            evaluation_hashes = _mapping(
                by_id[node_id].get("native_ir_hashes"),
                f"NRIR-9 {mode} evaluation hashes",
            )
            if evaluation_hashes != ir_hashes:
                raise ValueError(f"NRIR-9 {mode} stack/node IR hashes differ")
    if stack_ids != set(batches):
        raise ValueError(f"NRIR-9 {mode} stack coverage differs")
    logical_signatures: dict[str, tuple[object, ...]] = {}
    for node_id, evaluation in by_id.items():
        node = _mapping(evaluation.get("node"), f"NRIR-9 {mode} node")
        logical_signatures[node_id] = (
            node.get("parent_node_id"),
            node.get("depth"),
            node.get("branch_relu_input"),
            node.get("branch_neuron_index"),
            node.get("branch_value"),
            decision_signature.get(node_id, ("frontier",)),
        )
    return by_id, logical_signatures


def validate_relu_split_bab_evidence(evidence: Mapping[str, Any]) -> None:
    """Reject incomplete, relinked, or claim-inflated NRIR-9 evidence."""

    if (
        evidence.get("schema_version") != RELU_SPLIT_BAB_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("property_status") != "not_claimed"
        or evidence.get("compiler_version") != NATIVE_RELU_SPLIT_BAB_COMPILER_VERSION
    ):
        raise ValueError("NRIR-9 evidence header differs")
    claim = str(evidence.get("claim_boundary", ""))
    for phrase in (
        "first-class ReLU split",
        "best-first bounded queue",
        "not alpha/beta optimization",
        "complete property verdict",
        "speedup",
    ):
        if phrase not in claim:
            raise ValueError("NRIR-9 claim boundary omits a hard limitation")
    source = _mapping(evidence.get("source"), "NRIR-9 source")
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
        raise ValueError("NRIR-9 source identity differs")
    gates = _mapping(evidence.get("gates"), "NRIR-9 gates")
    expected_gates = {
        "fixed_resnet_and_property_source_are_digest_bound",
        "all_six_relu_splits_are_first_class_bound_inputs",
        "split_state_reaches_plan_task_and_schedule_ir",
        "every_child_recomputes_exact_state_without_parent_reuse",
        "best_first_queue_forms_three_generations_and_four_frontier_nodes",
        "packed_three_stacks_replace_seven_serial_stacks",
        "packed_and_serial_bounds_and_queue_semantics_match",
        "bounded_run_reports_unknown_budget_exhausted_without_performance_claim",
    }
    if set(gates) != expected_gates or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("NRIR-9 gates are incomplete or failed")
    comparison = _mapping(evidence.get("comparison"), "NRIR-9 comparison")
    for name in (
        "node_ids_same",
        "logical_queue_signature_same",
        "split_state_hashes_same",
        "lower_allclose",
        "upper_allclose",
    ):
        if comparison.get(name) is not True:
            raise ValueError("NRIR-9 packed/serial comparison failed")
    validated_nodes: dict[str, dict[str, Mapping[str, Any]]] = {}
    validated_signatures: dict[str, dict[str, tuple[object, ...]]] = {}
    for mode, expected_stack_count, expected_batch_size in (
        ("packed", 3, PACKED_EVAL_BATCH_SIZE),
        ("serial", 7, 1),
    ):
        section = _mapping(evidence.get(mode), f"NRIR-9 {mode}")
        trace = _mapping(section.get("trace"), f"NRIR-9 {mode} trace")
        if section.get("trace_hash") != hashlib_sha256(trace):
            raise ValueError(f"NRIR-9 {mode} trace hash differs")
        nodes, signatures = _validate_frozen_queue_trace(
            trace,
            mode=mode,
            expected_stack_count=expected_stack_count,
            expected_batch_size=expected_batch_size,
        )
        if (
            trace.get("root_input_lower_hash") != source.get("root_input_lower_hash")
            or trace.get("root_input_upper_hash") != source.get("root_input_upper_hash")
            or trace.get("objective_hash") != source.get("objective_hash")
        ):
            raise ValueError(f"NRIR-9 {mode} trace/source identity differs")
        validated_nodes[mode] = nodes
        validated_signatures[mode] = signatures
    packed_nodes = validated_nodes["packed"]
    serial_nodes = validated_nodes["serial"]
    if set(packed_nodes) != set(serial_nodes):
        raise ValueError("NRIR-9 packed/serial node identities differ")
    split_same = all(
        _mapping(packed_nodes[node_id].get("node"), "NRIR-9 packed node").get(
            "split_state_hash"
        )
        == _mapping(serial_nodes[node_id].get("node"), "NRIR-9 serial node").get(
            "split_state_hash"
        )
        for node_id in packed_nodes
    )
    logical_same = validated_signatures["packed"] == validated_signatures["serial"]
    exact_same = all(
        packed_nodes[node_id].get("exact_state_hash")
        == serial_nodes[node_id].get("exact_state_hash")
        for node_id in packed_nodes
    )
    lower_difference = max(
        abs(
            float(packed_nodes[node_id]["lower"])
            - float(serial_nodes[node_id]["lower"])
        )
        for node_id in packed_nodes
    )
    upper_difference = max(
        abs(
            float(packed_nodes[node_id]["upper"])
            - float(serial_nodes[node_id]["upper"])
        )
        for node_id in packed_nodes
    )
    lower_scale = max(abs(float(item["lower"])) for item in serial_nodes.values())
    upper_scale = max(abs(float(item["upper"])) for item in serial_nodes.values())
    recomputed = {
        "node_ids_same": True,
        "logical_queue_signature_same": logical_same,
        "exact_node_state_hashes_same": exact_same,
        "split_state_hashes_same": split_same,
        "lower_allclose": lower_difference <= ATOL + RTOL * lower_scale,
        "upper_allclose": upper_difference <= ATOL + RTOL * upper_scale,
        "lower_max_abs_diff": lower_difference,
        "upper_max_abs_diff": upper_difference,
        "atol": ATOL,
        "rtol": RTOL,
    }
    if dict(comparison) != recomputed:
        raise ValueError(
            "NRIR-9 packed/serial comparison was not independently derived"
        )
    limitations = _list(evidence.get("limitations"), "NRIR-9 limitations")
    joined = " ".join(str(item) for item in limitations)
    for phrase in (
        "no alpha/beta optimization",
        "property_status=not_claimed",
        "never child exact state",
        "not a timing claim",
        "no latency, memory, CUDA, or speedup claim",
    ):
        if phrase not in joined:
            raise ValueError("NRIR-9 limitations are incomplete")


def _artifact_payload(evidence: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema_version": RELU_SPLIT_BAB_ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "evidence": dict(evidence),
    }


def _generate(args: argparse.Namespace) -> None:
    evidence = build_relu_split_bab_evidence(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = _artifact_payload(evidence)
    (args.artifact_dir / ARTIFACT_FILES[0]).write_text(
        _canonical_json(payload, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": RELU_SPLIT_BAB_ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "evidence_hash": hashlib_sha256(evidence),
        "files": {
            name: file_sha256(args.artifact_dir / name) for name in ARTIFACT_FILES
        },
    }
    (args.artifact_dir / MANIFEST_FILE).write_text(
        _canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        _canonical_json(
            {
                "status": "ok",
                "mode": "generate",
                "artifact_schema": RELU_SPLIT_BAB_ARTIFACT_SCHEMA_VERSION,
                "evidence_hash": hashlib_sha256(evidence),
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    manifest = json.loads(
        (args.artifact_dir / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    if not isinstance(manifest, dict) or (
        manifest.get("schema_version") != RELU_SPLIT_BAB_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or manifest.get("files")
        != {name: file_sha256(args.artifact_dir / name) for name in ARTIFACT_FILES}
    ):
        raise ValueError("NRIR-9 artifact manifest differs")
    path = args.artifact_dir / ARTIFACT_FILES[0]
    stored = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict):
        raise TypeError("NRIR-9 artifact root must be a mapping")
    if (
        stored.get("schema_version") != RELU_SPLIT_BAB_ARTIFACT_SCHEMA_VERSION
        or stored.get("status") != "ok"
        or stored.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-9 artifact header differs")
    stored_evidence = _mapping(stored.get("evidence"), "NRIR-9 stored evidence")
    validate_relu_split_bab_evidence(stored_evidence)
    if manifest.get("evidence_hash") != hashlib_sha256(stored_evidence):
        raise ValueError("NRIR-9 manifest/evidence hash differs")
    rebuilt = build_relu_split_bab_evidence(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
    )
    if _canonical_json(stored_evidence) != _canonical_json(rebuilt):
        raise ValueError("NRIR-9 replay differs from frozen evidence")
    print(
        _canonical_json(
            {
                "status": "ok",
                "mode": "replay",
                "artifact_schema": RELU_SPLIT_BAB_ARTIFACT_SCHEMA_VERSION,
                "evidence_hash": hashlib_sha256(rebuilt),
            }
        )
    )


def hashlib_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
