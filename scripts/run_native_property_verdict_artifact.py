#!/usr/bin/env python3
"""Generate or replay NRIR-13 three-state property verdict evidence."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=line-too-long,duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    execute_native_optimized_relu_split_bab,
)
from boundflow.runtime.native_property_verdict import (
    NATIVE_PROPERTY_VERDICT_COMPILER_VERSION,
    NATIVE_PROPERTY_VERDICT_SCHEMA_VERSION,
    derive_native_property_verdict,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import (
    InputSpec,
    execute_task_module_concrete,
)
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

ARTIFACT_SCHEMA_VERSION = "boundflow.native-property-verdict-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.native-property-verdict-evidence/v1"
ARTIFACT_FILE = "property_verdicts.json"
MANIFEST_FILE = "manifest.json"
FIXED_RUN_ID = "vnncomp21-resnet2b-prop0-native-ir13-property-verdict"
POLICY = NativeAlphaBetaOptimizerPolicy(
    steps=1,
    lr=0.05,
    alpha_init=0.5,
    beta_init=0.0,
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


def canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _toy_module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="native-property-artifact-toy",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="native-property-artifact-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0]]),
                "b1": torch.tensor([0.0]),
                "W2": torch.tensor([[1.0]]),
                "b2": torch.tensor([0.0]),
            }
        },
    )


def _toy_spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-1.0]]),
        upper=torch.tensor([[1.0]]),
    )


def _toy_queue(*, run_id: str, threshold: float):
    return execute_native_optimized_relu_split_bab(
        _toy_module(),
        _toy_spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        run_id=run_id,
        config=NativeReluSplitBabConfig(
            max_nodes=1,
            max_depth=0,
            expansion_batch_size=1,
            max_eval_batch_size=1,
            threshold=threshold,
        ),
        optimizer_policy=NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1),
    )


def _case(queue, verdict) -> dict[str, object]:
    queue_trace = queue.trace.to_dict()
    verdict_trace = verdict.trace.to_dict()
    return {
        "queue_trace": queue_trace,
        "queue_trace_hash": queue.trace.stable_hash(),
        "verdict_trace": verdict_trace,
        "verdict_trace_hash": verdict.trace.stable_hash(queue.trace),
        "counterexample_input_hash": (
            None
            if verdict.counterexample_input is None
            else tensor_content_hash(verdict.counterexample_input)
        ),
    }


def build_native_property_verdict_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-13 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    if tuple(op.op_type for op in module.get_entry_task().ops) != EXPECTED_PRIMAL_OPS:
        raise ValueError("NRIR-13 primal topology differs")
    fixed_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    fixed_objective = tensors["linear_spec_c"][:, 0:1].contiguous()
    fixed_queue = execute_native_optimized_relu_split_bab(
        module,
        fixed_spec,
        linear_spec_C=fixed_objective,
        run_id=FIXED_RUN_ID,
        config=NativeReluSplitBabConfig(
            max_nodes=7,
            max_depth=8,
            expansion_batch_size=2,
            max_eval_batch_size=4,
            threshold=0.0,
        ),
        optimizer_policy=POLICY,
    )
    fixed_root_id = fixed_queue.trace.evaluations[0].node.node_id
    fixed_verdict = derive_native_property_verdict(
        module,
        fixed_spec,
        linear_spec_C=fixed_objective,
        queue_execution=fixed_queue,
        candidate_counterexamples=((fixed_root_id, fixed_spec.center),),
    )
    center_execution = execute_task_module_concrete(
        module,
        fixed_spec.center,
        input_value_name=fixed_spec.value_name,
    )
    center_objective_value = float(
        (center_execution.output * fixed_objective[0]).sum().item()
    )

    toy_objective = torch.tensor([[1.0]])
    verified_queue = _toy_queue(run_id="native-property-verified-toy", threshold=-2.0)
    verified = derive_native_property_verdict(
        _toy_module(),
        _toy_spec(),
        linear_spec_C=toy_objective,
        queue_execution=verified_queue,
    )
    unresolved_queue = _toy_queue(
        run_id="native-property-unresolved-toy", threshold=0.5
    )
    unknown = derive_native_property_verdict(
        _toy_module(),
        _toy_spec(),
        linear_spec_C=toy_objective,
        queue_execution=unresolved_queue,
    )
    unsafe = derive_native_property_verdict(
        _toy_module(),
        _toy_spec(),
        linear_spec_C=toy_objective,
        queue_execution=unresolved_queue,
        candidate_counterexamples=(
            (unresolved_queue.trace.evaluations[0].node.node_id, torch.tensor([[0.0]])),
        ),
    )

    fixed_case = _case(fixed_queue, fixed_verdict)
    verified_case = _case(verified_queue, verified)
    unsafe_case = _case(unresolved_queue, unsafe)
    unknown_case = _case(unresolved_queue, unknown)
    gates = {
        "verified_requires_closed_sound_prune_leaves": bool(
            verified.trace.status == "verified"
            and verified_queue.trace.status == "complete"
            and not verified.trace.unresolved_leaf_node_ids
            and bool(verified.trace.sound_pruned_leaf_node_ids)
        ),
        "unsafe_requires_reexecuted_concrete_witness": bool(
            unsafe.trace.status == "unsafe"
            and unsafe.trace.counterexample is not None
            and unsafe.trace.counterexample.objective_value == 0.0
            and unsafe.trace.counterexample.objective_margin == -0.5
        ),
        "open_budget_and_depth_never_inflate_to_verified": bool(
            unknown.trace.status == "unknown"
            and unknown.trace.reason == "configured_depth_terminal_open"
        ),
        "fixed_resnet_center_reexecutes_primal_task_ir_without_counterexample": bool(
            center_objective_value > 0.0
            and fixed_verdict.trace.status == "unknown"
            and fixed_verdict.counterexample_input is None
        ),
        "fixed_resnet_frontier_remains_explicit_unknown": bool(
            fixed_queue.trace.status == "budget_exhausted"
            and fixed_verdict.trace.reason == "node_budget_frontier_open"
            and set(fixed_verdict.trace.unresolved_leaf_node_ids)
            == set(fixed_queue.trace.final_frontier_node_ids)
        ),
        "fixed_source_and_property_are_digest_bound": bool(
            source_manifest.get("schema_version")
            == "boundflow.native-real-network-ir-artifact/v1"
            and file_sha256(model) == MODEL_SHA256
        ),
        "all_verdicts_are_correctness_only": bool(
            all(
                _mapping(case["verdict_trace"], "NRIR-13 verdict").get(
                    "performance_claimed"
                )
                is False
                for case in (fixed_case, verified_case, unsafe_case, unknown_case)
            )
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-13 gates failed: {gates}")

    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "compiler_version": NATIVE_PROPERTY_VERDICT_COMPILER_VERSION,
        "performance_claimed": False,
        "property_status": "verified_unsafe_unknown_contract_matrix",
        "claim_boundary": (
            "sound three-state verdict derivation over immutable optimized queue traces; "
            "verified requires closed lower-bound pruning, unsafe requires primal Task IR "
            "counterexample replay, and open budget/depth remains unknown; correctness only"
        ),
        "source": {
            "native_ir_manifest_sha256": file_sha256(
                source_artifact_dir / "manifest.json"
            ),
            "native_ir_payload_sha256": file_sha256(source_artifact_dir / "payload.pt"),
            "model_sha256": MODEL_SHA256,
            "vnnlib_sha256": VNNLIB_SHA256,
            "vnncomp_commit": VNNCOMP_COMMIT,
            "abcrown_commit": ABCROWN_COMMIT,
        },
        "fixed_resnet": {
            **fixed_case,
            "center_input_hash": tensor_content_hash(fixed_spec.center),
            "center_output_hash": tensor_content_hash(center_execution.output),
            "center_objective_value": center_objective_value,
        },
        "toy_verified": verified_case,
        "toy_unsafe": unsafe_case,
        "toy_unknown": unknown_case,
        "gates": gates,
        "limitations": [
            "candidate discovery is supplied by the caller; adversarial attack integration is pending",
            "fixed ResNet remains a seven-node budget-bounded unknown, not a complete verdict",
            "single scalar lower-bound property only",
            "CPU correctness evidence only; no latency, memory, CUDA, or speedup claim",
            "timeout and dynamic optimizer early-stop policies are not implemented",
        ],
    }
    validate_native_property_verdict_evidence(evidence)
    return evidence


def _validate_case(
    value: object,
    *,
    expected_status: str,
    expected_reason: str,
) -> Mapping[str, Any]:
    case = _mapping(value, "NRIR-13 case")
    queue = _mapping(case.get("queue_trace"), "NRIR-13 queue trace")
    verdict = _mapping(case.get("verdict_trace"), "NRIR-13 verdict trace")
    if (
        case.get("queue_trace_hash") != canonical_hash(queue)
        or case.get("verdict_trace_hash") != canonical_hash(verdict)
        or verdict.get("schema_version") != NATIVE_PROPERTY_VERDICT_SCHEMA_VERSION
        or verdict.get("compiler_version") != NATIVE_PROPERTY_VERDICT_COMPILER_VERSION
        or verdict.get("status") != expected_status
        or verdict.get("reason") != expected_reason
        or verdict.get("performance_claimed") is not False
        or verdict.get("queue_trace_hash") != case.get("queue_trace_hash")
        or verdict.get("objective_hash") != queue.get("objective_hash")
        or verdict.get("threshold")
        != _mapping(queue.get("config"), "NRIR-13 queue config").get("threshold")
    ):
        raise ValueError("NRIR-13 case trace/hash/verdict differs")
    return case


def validate_native_property_verdict_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("compiler_version") != NATIVE_PROPERTY_VERDICT_COMPILER_VERSION
        or evidence.get("performance_claimed") is not False
        or evidence.get("property_status") != "verified_unsafe_unknown_contract_matrix"
        or not isinstance(evidence.get("claim_boundary"), str)
    ):
        raise ValueError("NRIR-13 evidence header differs")
    source = _mapping(evidence.get("source"), "NRIR-13 source")
    if (
        source.get("model_sha256") != MODEL_SHA256
        or source.get("vnnlib_sha256") != VNNLIB_SHA256
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("abcrown_commit") != ABCROWN_COMMIT
    ):
        raise ValueError("NRIR-13 source identity differs")
    fixed = _validate_case(
        evidence.get("fixed_resnet"),
        expected_status="unknown",
        expected_reason="node_budget_frontier_open",
    )
    verified = _validate_case(
        evidence.get("toy_verified"),
        expected_status="verified",
        expected_reason="all_leaves_soundly_pruned",
    )
    unsafe = _validate_case(
        evidence.get("toy_unsafe"),
        expected_status="unsafe",
        expected_reason="concrete_counterexample_reexecuted",
    )
    unknown = _validate_case(
        evidence.get("toy_unknown"),
        expected_status="unknown",
        expected_reason="configured_depth_terminal_open",
    )
    fixed_verdict = _mapping(fixed["verdict_trace"], "NRIR-13 fixed verdict")
    fixed_queue = _mapping(fixed["queue_trace"], "NRIR-13 fixed queue")
    if (
        fixed_queue.get("status") != "budget_exhausted"
        or fixed_queue.get("property_status") != "not_claimed"
        or set(fixed_verdict.get("unresolved_leaf_node_ids", []))
        != set(fixed_queue.get("final_frontier_node_ids", []))
        or fixed_verdict.get("counterexample") is not None
        or fixed.get("counterexample_input_hash") is not None
        or float(fixed.get("center_objective_value", -1.0)) <= 0.0
    ):
        raise ValueError("NRIR-13 fixed ResNet unknown boundary differs")
    verified_verdict = _mapping(verified["verdict_trace"], "verified verdict")
    if (
        verified_verdict.get("unresolved_leaf_node_ids") != []
        or not verified_verdict.get("sound_pruned_leaf_node_ids")
        or verified_verdict.get("counterexample") is not None
    ):
        raise ValueError("NRIR-13 verified proof closure differs")
    unsafe_verdict = _mapping(unsafe["verdict_trace"], "unsafe verdict")
    witness = _mapping(unsafe_verdict.get("counterexample"), "unsafe witness")
    if (
        unsafe.get("counterexample_input_hash") != witness.get("input_hash")
        or float(witness.get("objective_value", 1.0)) != 0.0
        or float(witness.get("objective_margin", 0.0)) != -0.5
        or float(witness.get("input_box_max_violation", 1.0)) != 0.0
    ):
        raise ValueError("NRIR-13 unsafe witness differs")
    unknown_verdict = _mapping(unknown["verdict_trace"], "unknown verdict")
    if (
        not unknown_verdict.get("unresolved_leaf_node_ids")
        or unknown_verdict.get("counterexample") is not None
    ):
        raise ValueError("NRIR-13 unknown leaf accounting differs")
    gates = _mapping(evidence.get("gates"), "NRIR-13 gates")
    expected_gates = {
        "verified_requires_closed_sound_prune_leaves",
        "unsafe_requires_reexecuted_concrete_witness",
        "open_budget_and_depth_never_inflate_to_verified",
        "fixed_resnet_center_reexecutes_primal_task_ir_without_counterexample",
        "fixed_resnet_frontier_remains_explicit_unknown",
        "fixed_source_and_property_are_digest_bound",
        "all_verdicts_are_correctness_only",
    }
    if set(gates) != expected_gates or any(
        value is not True for value in gates.values()
    ):
        raise ValueError("NRIR-13 gates differ")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 5:
        raise ValueError("NRIR-13 limitations differ")


def _generate(args: argparse.Namespace) -> None:
    evidence = build_native_property_verdict_evidence(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "evidence": evidence,
    }
    artifact_path = args.artifact_dir / ARTIFACT_FILE
    artifact_path.write_text(
        _canonical_json(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "files": {ARTIFACT_FILE: file_sha256(artifact_path)},
        "evidence_hash": canonical_hash(evidence),
    }
    (args.artifact_dir / MANIFEST_FILE).write_text(
        _canonical_json(manifest, indent=2) + "\n",
        encoding="utf-8",
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
        raise ValueError("NRIR-13 artifact manifest/header differs")
    stored = _mapping(artifact.get("evidence"), "NRIR-13 stored evidence")
    validate_native_property_verdict_evidence(stored)
    if manifest.get("evidence_hash") != canonical_hash(stored):
        raise ValueError("NRIR-13 stored evidence hash differs")
    actual = build_native_property_verdict_evidence(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
    )
    if stored != actual:
        raise ValueError("NRIR-13 replay differs from frozen evidence")
    print(_canonical_json({"status": "ok", "evidence_hash": canonical_hash(actual)}))


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
