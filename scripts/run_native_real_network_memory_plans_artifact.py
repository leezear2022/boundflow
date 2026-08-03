#!/usr/bin/env python3
"""Generate or replay NRIR-2 storage-plan evidence on the frozen real ResNet."""

# pylint: disable=duplicate-code,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.ir.schedule import AllocateAction
from boundflow.planner import plan_interval_ibp_v0
from boundflow.planner.plan_ir_selector import NoFeasiblePlanError
from boundflow.runtime.abcrown_adapter import (
    bind_intermediate_bounds,
    deserialize_intermediate_bounds,
    file_sha256,
    intermediate_bounds_sha256,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_verifier_ir_integration import (
    NATIVE_PLAIN_CROWN_MEMORY_COMPILER_VERSION,
    compile_native_plain_crown_memory_query,
    execute_native_plain_crown_memory_query,
)
from boundflow.runtime.schedule_ir_executor import execute_schedule_reference
from boundflow.runtime.task_executor import InputSpec
from scripts.run_native_real_network_ir_artifact import (
    ABCROWN_COMMIT,
    ARTIFACT_PAYLOAD_SCHEMA_VERSION,
    ARTIFACT_SCHEMA_VERSION,
    EXPECTED_PRIMAL_OPS,
    INTERMEDIATE_BOUNDS_SHA256,
    MODEL_SHA256,
    VNNCOMP_COMMIT,
    VNNLIB_SHA256,
)

MEMORY_ARTIFACT_SCHEMA_VERSION = "boundflow.native-real-network-memory-artifact/v1"
MEMORY_EVIDENCE_SCHEMA_VERSION = "boundflow.native-real-network-memory-evidence/v1"
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir2-memory"
AVAILABLE_MEMORY_BYTES = 1 << 30
ATOL = 2e-4
RTOL = 2e-4
ARTIFACT_FILES = ("plans.json",)


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


def _payload_tensors(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name in ("input_lower", "input_upper", "linear_spec_c", "external_lower"):
        value = payload.get(name)
        if not torch.is_tensor(value):
            raise TypeError(f"source artifact payload {name} must be a tensor")
        tensors[name] = value.detach().cpu().contiguous()
    if payload.get("schema_version") != ARTIFACT_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("source artifact payload schema differs")
    if not bool((tensors["input_lower"] <= tensors["input_upper"]).all()):
        raise ValueError("source artifact input interval is malformed")
    return tensors


def _load_source_artifact(source_dir: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    manifest_path = source_dir / "manifest.json"
    payload_path = source_dir / "payload.pt"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("source native IR artifact contract differs")
    inputs = _mapping(manifest.get("inputs"), "source inputs")
    expected_inputs = {
        "model_sha256": MODEL_SHA256,
        "vnnlib_sha256": VNNLIB_SHA256,
        "vnncomp_commit": VNNCOMP_COMMIT,
        "abcrown_commit": ABCROWN_COMMIT,
        "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
    }
    for name, expected in expected_inputs.items():
        if inputs.get(name) != expected:
            raise ValueError(f"source artifact identity differs: {name}")
    payload_meta = _mapping(manifest.get("payload"), "source payload metadata")
    if payload_meta.get("sha256") != file_sha256(payload_path):
        raise ValueError("source artifact payload digest differs")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("source artifact payload root must be a mapping")
    tensors = _payload_tensors(payload)
    tensor_hashes = {
        name: tensor_content_hash(tensor) for name, tensor in sorted(tensors.items())
    }
    if payload_meta.get("tensor_sha256") != tensor_hashes:
        raise ValueError("source artifact tensor digests differ")
    return manifest, payload


def build_memory_plan_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    """Recompile and execute both storage plans from frozen semantic inputs."""

    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-2 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    external_bounds = deserialize_intermediate_bounds(
        _mapping(payload.get("external_intermediate_bounds"), "intermediate bounds")
    )
    if intermediate_bounds_sha256(external_bounds) != INTERMEDIATE_BOUNDS_SHA256:
        raise ValueError("NRIR-2 intermediate-bound digest differs")

    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    legacy_module = plan_interval_ibp_v0(program)
    primal_ops = tuple(op.op_type for op in legacy_module.get_entry_task().ops)
    if primal_ops != EXPECTED_PRIMAL_OPS:
        raise ValueError("NRIR-2 primal topology differs")
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    interval_env, local_relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    relu_pre = bind_intermediate_bounds(external_bounds, local_relu_pre)

    high = compile_native_plain_crown_memory_query(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=tensors["linear_spec_c"],
        intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
        query_id=QUERY_ID,
        available_memory_bytes=AVAILABLE_MEMORY_BYTES,
        memory_budget_bytes=AVAILABLE_MEMORY_BYTES,
    )
    candidates = {
        candidate.candidate_id: candidate
        for candidate in high.template.storage_candidates
    }
    retain = candidates["storage:native-retain-all-v1"]
    reuse = candidates["storage:native-lifetime-reuse-v1"]
    low_budget = reuse.cost.predicted_peak_bytes
    low = compile_native_plain_crown_memory_query(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=tensors["linear_spec_c"],
        intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
        query_id=QUERY_ID,
        available_memory_bytes=AVAILABLE_MEMORY_BYTES,
        memory_budget_bytes=low_budget,
    )
    below_min_failures: list[dict[str, object]] = []
    try:
        compile_native_plain_crown_memory_query(
            legacy_module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=tensors["linear_spec_c"],
            intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
            query_id=QUERY_ID,
            available_memory_bytes=AVAILABLE_MEMORY_BYTES,
            memory_budget_bytes=low_budget - 1,
        )
    except NoFeasiblePlanError as error:
        below_min_failures = [
            {"reason": failure.reason, "count": failure.count}
            for failure in error.failures
        ]
    if not below_min_failures:
        raise ValueError("NRIR-2 accepted a budget below the smallest storage plan")

    high_result, high_task_trace, high_storage_trace = (
        execute_native_plain_crown_memory_query(
            high,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=tensors["linear_spec_c"],
        )
    )
    low_result, low_task_trace, low_storage_trace = (
        execute_native_plain_crown_memory_query(
            low,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=tensors["linear_spec_c"],
        )
    )
    high_schedule_trace = execute_schedule_reference(
        high.schedule,
        bound_module=high.bound_module,
        template=high.template,
        instance=high.instance,
    )
    low_schedule_trace = execute_schedule_reference(
        low.schedule,
        bound_module=low.bound_module,
        template=low.template,
        instance=low.instance,
    )

    expected = tensors["external_lower"].to(high_result.lower)
    high_diff = (high_result.lower - expected).abs()
    low_diff = (low_result.lower - expected).abs()
    plans_equal = bool(
        torch.allclose(high_result.lower, low_result.lower, atol=0.0, rtol=0.0)
        and torch.allclose(high_result.upper, low_result.upper, atol=0.0, rtol=0.0)
    )
    semantics = {
        "plans_bitwise_equal": plans_equal,
        "retain_vs_external": {
            "allclose": bool(
                torch.allclose(high_result.lower, expected, atol=ATOL, rtol=RTOL)
            ),
            "max_abs_diff": float(high_diff.max().item()),
            "sign_agreement": int(
                ((high_result.lower >= 0) == (expected >= 0)).sum().item()
            ),
            "sign_total": int(expected.numel()),
        },
        "reuse_vs_external": {
            "allclose": bool(
                torch.allclose(low_result.lower, expected, atol=ATOL, rtol=RTOL)
            ),
            "max_abs_diff": float(low_diff.max().item()),
            "sign_agreement": int(
                ((low_result.lower >= 0) == (expected >= 0)).sum().item()
            ),
            "sign_total": int(expected.numel()),
        },
        "lower_sha256": tensor_content_hash(high_result.lower),
        "upper_sha256": tensor_content_hash(high_result.upper),
    }
    if (
        not plans_equal
        or semantics["retain_vs_external"]["allclose"] is not True  # type: ignore[index]
        or semantics["reuse_vs_external"]["allclose"] is not True  # type: ignore[index]
        or semantics["retain_vs_external"]["sign_agreement"] != 9  # type: ignore[index]
        or semantics["reuse_vs_external"]["sign_agreement"] != 9  # type: ignore[index]
    ):
        raise ValueError("NRIR-2 storage plans differ semantically")

    high_hashes = high.hashes()
    low_hashes = low.hashes()
    template_same = (
        high_hashes["plan_template_hash"] == low_hashes["plan_template_hash"]
    )
    decision_switched = (
        high.instance.storage_decision.candidate_id == retain.candidate_id
        and low.instance.storage_decision.candidate_id == reuse.candidate_id
    )
    alias_pairs = _physical_alias_pairs(low, reuse.bindings)
    alias_pairs_sha256 = hashlib.sha256(
        _canonical_json(alias_pairs).encode("utf-8")
    ).hexdigest()
    early_releases = sum(
        len(event.evicted_value_ids) for event in low_storage_trace.events[:-1]
    )
    gates = {
        "same_bound_ir_and_plan_template": bool(
            high_hashes["bound_module_hash"] == low_hashes["bound_module_hash"]
            and template_same
        ),
        "budget_switches_storage_decision": decision_switched,
        "schedule_arena_changes": (
            high_schedule_trace.peak_memory_bytes > low_schedule_trace.peak_memory_bytes
        ),
        "lifetime_reuse_has_physical_aliases": bool(alias_pairs),
        "runtime_releases_before_final_task": early_releases > 0,
        "runtime_peak_residency_reduced": (
            high_storage_trace.observed_peak_live_bytes
            > low_storage_trace.observed_peak_live_bytes
        ),
        "both_plans_match_external_semantics": bool(
            semantics["retain_vs_external"]["allclose"]  # type: ignore[index]
            and semantics["reuse_vs_external"]["allclose"]  # type: ignore[index]
        ),
        "below_minimum_budget_fails_closed": any(
            item["reason"] == "memory_budget_exceeded" for item in below_min_failures
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-2 gates failed: {gates}")

    evidence: dict[str, object] = {
        "schema_version": MEMORY_EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "compiler_version": NATIVE_PLAIN_CROWN_MEMORY_COMPILER_VERSION,
        "claim_boundary": (
            "real-network budget selection, Schedule arena, and runtime last-use "
            "release correctness only; no latency, GPU, allocator, or OOM claim"
        ),
        "source": {
            "native_ir_artifact_schema": source_manifest["schema_version"],
            "native_ir_manifest_sha256": file_sha256(
                source_artifact_dir / "manifest.json"
            ),
            "native_ir_payload_sha256": file_sha256(source_artifact_dir / "payload.pt"),
            "model_sha256": MODEL_SHA256,
            "vnnlib_sha256": VNNLIB_SHA256,
            "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
        },
        "graph": {
            "primal_op_count": len(primal_ops),
            "bound_op_count": len(high.bound_module.graph.ops),
            "task_count": len(high.task_module.tasks),
        },
        "template": {
            "storage_candidate_count": len(high.template.storage_candidates),
            "storage_candidate_ids": sorted(candidates),
            "plan_template_hash": high_hashes["plan_template_hash"],
        },
        "retain_all": _plan_summary(
            high,
            high_task_trace.stable_hash(),
            high_storage_trace.to_dict(),
            high_schedule_trace.peak_memory_bytes,
        ),
        "lifetime_reuse": {
            **_plan_summary(
                low,
                low_task_trace.stable_hash(),
                low_storage_trace.to_dict(),
                low_schedule_trace.peak_memory_bytes,
            ),
            "physical_alias_pair_count": len(alias_pairs),
            "physical_alias_pairs_sha256": alias_pairs_sha256,
            "physical_alias_pairs_sample": alias_pairs[:16],
            "early_release_count": early_releases,
        },
        "below_minimum_budget": {
            "budget_bytes": low_budget - 1,
            "failures": below_min_failures,
        },
        "semantics": semantics,
        "gates": gates,
        "limitations": [
            "CPU correctness and compiler/runtime ownership evidence only",
            "storage arena offsets are enforced as a logical non-aliasing contract",
            (
                "last-use release deletes runtime tensor/operator references but "
                "does not expose a custom torch allocator"
            ),
            (
                "the 0.001 ms reuse policy cost is deterministic ordering metadata, "
                "not benchmark evidence"
            ),
            "no latency, speedup, CUDA peak-memory, backend OOM rescue, or performance claim",
            "representation and batch alternatives remain outside NRIR-2 v1",
        ],
    }
    validate_memory_plan_evidence(evidence)
    return evidence


def validate_memory_plan_evidence(evidence: Mapping[str, Any]) -> None:
    """Fail closed on internally inconsistent or claim-inflated NRIR-2 evidence."""

    if (
        evidence.get("schema_version") != MEMORY_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-2 evidence claim contract differs")
    gates = _mapping(evidence.get("gates"), "NRIR-2 gates")
    expected_gate_ids = {
        "same_bound_ir_and_plan_template",
        "budget_switches_storage_decision",
        "schedule_arena_changes",
        "lifetime_reuse_has_physical_aliases",
        "runtime_releases_before_final_task",
        "runtime_peak_residency_reduced",
        "both_plans_match_external_semantics",
        "below_minimum_budget_fails_closed",
    }
    if set(gates) != expected_gate_ids or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("NRIR-2 evidence gates are incomplete or failed")
    template = _mapping(evidence.get("template"), "NRIR-2 template")
    if (
        template.get("storage_candidate_count") != 2
        or set(template.get("storage_candidate_ids", ()))
        != {
            "storage:native-retain-all-v1",
            "storage:native-lifetime-reuse-v1",
        }
        or len(str(template.get("plan_template_hash", ""))) != 64
    ):
        raise ValueError("NRIR-2 template storage contract differs")
    retain = _mapping(evidence.get("retain_all"), "NRIR-2 retain plan")
    reuse = _mapping(evidence.get("lifetime_reuse"), "NRIR-2 reuse plan")
    if retain.get("storage_candidate_id") != "storage:native-retain-all-v1":
        raise ValueError("NRIR-2 retain plan identity differs")
    if reuse.get("storage_candidate_id") != "storage:native-lifetime-reuse-v1":
        raise ValueError("NRIR-2 reuse plan identity differs")
    retain_peak = int(retain.get("schedule_peak_bytes", 0))
    reuse_peak = int(reuse.get("schedule_peak_bytes", 0))
    if retain_peak <= reuse_peak or reuse_peak <= 0:
        raise ValueError("NRIR-2 storage peaks do not establish a budget switch")
    retain_hashes = _mapping(retain.get("ir_hashes"), "retain IR hashes")
    reuse_hashes = _mapping(reuse.get("ir_hashes"), "reuse IR hashes")
    expected_hash_keys = {
        "bound_module_hash",
        "plan_template_hash",
        "plan_instance_hash",
        "task_module_hash",
        "schedule_hash",
    }
    if (
        set(retain_hashes) != expected_hash_keys
        or set(reuse_hashes) != expected_hash_keys
    ):
        raise ValueError("NRIR-2 IR hash set differs")
    if any(
        len(str(value)) != 64
        for value in (*retain_hashes.values(), *reuse_hashes.values())
    ):
        raise ValueError("NRIR-2 IR identity is not SHA-256")
    if (
        retain_hashes["bound_module_hash"] != reuse_hashes["bound_module_hash"]
        or retain_hashes["plan_template_hash"] != reuse_hashes["plan_template_hash"]
        or retain_hashes["plan_template_hash"] != template["plan_template_hash"]
        or retain_hashes["plan_instance_hash"] == reuse_hashes["plan_instance_hash"]
        or retain_hashes["schedule_hash"] == reuse_hashes["schedule_hash"]
    ):
        raise ValueError("NRIR-2 cross-plan IR identity contract differs")
    _validate_storage_trace_contract(
        _mapping(retain.get("storage_trace"), "retain storage trace"),
        plan=retain,
        ir_hashes=retain_hashes,
    )
    _validate_storage_trace_contract(
        _mapping(reuse.get("storage_trace"), "reuse storage trace"),
        plan=reuse,
        ir_hashes=reuse_hashes,
    )
    retain_trace = _mapping(retain.get("storage_trace"), "retain storage trace")
    reuse_trace = _mapping(reuse.get("storage_trace"), "reuse storage trace")
    if int(retain_trace["observed_peak_live_bytes"]) <= int(
        reuse_trace["observed_peak_live_bytes"]
    ):
        raise ValueError("NRIR-2 runtime residency was not reduced")
    if (
        int(reuse.get("physical_alias_pair_count", 0)) <= 0
        or len(str(reuse.get("physical_alias_pairs_sha256", ""))) != 64
        or int(reuse.get("early_release_count", 0)) <= 0
    ):
        raise ValueError("NRIR-2 reuse evidence lacks aliases or early release")
    below = _mapping(evidence.get("below_minimum_budget"), "below-minimum budget")
    failures = below.get("failures")
    if (
        int(below.get("budget_bytes", 0)) != reuse_peak - 1
        or not isinstance(failures, list)
        or not any(
            isinstance(item, Mapping) and item.get("reason") == "memory_budget_exceeded"
            for item in failures
        )
    ):
        raise ValueError("NRIR-2 below-minimum budget did not fail closed")
    semantics = _mapping(evidence.get("semantics"), "NRIR-2 semantics")
    retain_semantics = _mapping(semantics.get("retain_vs_external"), "retain semantics")
    reuse_semantics = _mapping(semantics.get("reuse_vs_external"), "reuse semantics")
    if (
        semantics.get("plans_bitwise_equal") is not True
        or retain_semantics.get("allclose") is not True
        or reuse_semantics.get("allclose") is not True
        or retain_semantics.get("sign_agreement") != 9
        or reuse_semantics.get("sign_agreement") != 9
    ):
        raise ValueError("NRIR-2 semantic evidence differs")


def _validate_storage_trace_contract(
    trace: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    ir_hashes: Mapping[str, Any],
) -> None:
    if (
        trace.get("storage_candidate_id") != plan.get("storage_candidate_id")
        or trace.get("plan_template_hash") != ir_hashes.get("plan_template_hash")
        or trace.get("plan_instance_hash") != ir_hashes.get("plan_instance_hash")
        or int(trace.get("planned_peak_bytes", 0))
        != int(plan.get("schedule_peak_bytes", 0))
        or int(trace.get("observed_peak_live_bytes", 0))
        > int(trace.get("planned_peak_bytes", 0))
        or int(trace.get("released_value_count", -1)) < 0
        or not isinstance(trace.get("events"), list)
    ):
        raise ValueError("NRIR-2 storage execution trace linkage differs")


def _plan_summary(
    compilation,
    task_trace_hash: str,
    storage_trace: Mapping[str, Any],
    schedule_peak_bytes: int,
) -> dict[str, object]:
    allocations = [
        action
        for action in compilation.schedule.actions
        if isinstance(action, AllocateAction)
    ]
    return {
        "memory_budget_bytes": compilation.instance.memory_budget_bytes,
        "storage_candidate_id": compilation.instance.storage_decision.candidate_id,
        "ir_hashes": compilation.hashes(),
        "schedule_allocation_count": len(allocations),
        "schedule_peak_bytes": schedule_peak_bytes,
        "task_trace_hash": task_trace_hash,
        "storage_trace": dict(storage_trace),
    }


def _physical_alias_pairs(compilation, bindings) -> list[list[str]]:
    op_index = {
        op.op_id: index for index, op in enumerate(compilation.bound_module.graph.ops)
    }
    result: list[list[str]] = []
    for index, left in enumerate(bindings):
        for right in bindings[index + 1 :]:
            byte_overlap = (
                left.arena_id == right.arena_id
                and left.offset_bytes < right.end_bytes
                and right.offset_bytes < left.end_bytes
            )
            lifetime_overlap = (
                op_index[left.live_from_op_id] <= op_index[right.live_to_op_id]
                and op_index[right.live_from_op_id] <= op_index[left.live_to_op_id]
            )
            if byte_overlap and not lifetime_overlap:
                result.append([left.value_id, right.value_id])
    return result


def generate_artifact(
    artifact_dir: Path, *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    """Write immutable NRIR-2 evidence without copying the parent payload."""

    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_memory_plan_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    (artifact_dir / "plans.json").write_text(
        _canonical_json(evidence, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": MEMORY_ARTIFACT_SCHEMA_VERSION,
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
    """Reject digest and semantic drift by recomputing both real-network plans."""

    manifest = _load_json(artifact_dir / "manifest.json")
    if (
        manifest.get("schema_version") != MEMORY_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-2 manifest contract differs")
    files = _mapping(manifest.get("files"), "NRIR-2 manifest files")
    if set(files) != set(ARTIFACT_FILES):
        raise ValueError("NRIR-2 manifest file set differs")
    for name in ARTIFACT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError(f"NRIR-2 artifact digest differs: {name}")
    stored = _load_json(artifact_dir / "plans.json")
    validate_memory_plan_evidence(stored)
    actual = build_memory_plan_evidence(
        model=model, source_artifact_dir=source_artifact_dir
    )
    if _canonical_json(stored) != _canonical_json(actual):
        raise ValueError("NRIR-2 semantic replay differs")
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
