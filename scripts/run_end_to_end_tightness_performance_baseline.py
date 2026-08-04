#!/usr/bin/env python3
"""Generate or replay the NRIR-15 tightness/performance diagnostic baseline."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-boolean-expressions,duplicate-code
# pylint: disable=missing-function-docstring,line-too-long

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Mapping, cast

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.ir.bound import IntermediateBoundSource
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.abcrown_adapter import (
    bind_intermediate_bounds,
    deserialize_intermediate_bounds,
    file_sha256,
    intermediate_bounds_sha256,
)
from boundflow.runtime.complete_verifier_query import (
    CompleteVerifierQueryPolicy,
    execute_complete_verifier_query,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
    search_native_box_counterexample,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    execute_native_optimized_relu_split_bab,
)
from boundflow.runtime.native_property_verdict import derive_native_property_verdict
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec
from scripts.run_complete_verifier_query_artifact import (
    ARTIFACT_SCHEMA_VERSION as LOCAL_ARTIFACT_SCHEMA_VERSION,
)
from scripts.run_complete_verifier_query_artifact import (
    canonical_hash as complete_query_hash,
)
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

ARTIFACT_SCHEMA_VERSION = "boundflow.end-to-end-tightness-performance-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.end-to-end-tightness-performance-evidence/v1"
ARTIFACT_FILE = "baseline.json"
MANIFEST_FILE = "manifest.json"
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir15-external-complete-query"
TIMING_CLAUSE_INDEX = 0
TIMING_GROUPS = 3
TIMING_WARMUPS = 1
EXTERNAL_ORACLE_ATOL = 2e-4
EXTERNAL_ORACLE_RTOL = 2e-4
SEARCH_POLICY = NativeProjectedGradientSearchPolicy(steps=4, step_size=0.002)
QUEUE_CONFIG = NativeReluSplitBabConfig(
    max_nodes=1,
    max_depth=0,
    expansion_batch_size=1,
    max_eval_batch_size=1,
)
LOCAL_POLICY = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.05)
EXTERNAL_CONSTANT_POLICY = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.05)
EXTERNAL_ADAPTIVE_POLICY = NativeAlphaBetaOptimizerPolicy(
    steps=1,
    lr=0.05,
    alpha_initialization_mode="adaptive",
)
VARIANT_ORDERS = (
    ("local_constant", "external_constant", "external_adaptive"),
    ("external_adaptive", "local_constant", "external_constant"),
    ("external_constant", "external_adaptive", "local_constant"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model", type=Path, required=True)
        subparser.add_argument("--source-artifact-dir", type=Path, required=True)
        subparser.add_argument("--local-artifact-dir", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
        subparser.add_argument("--torch-threads", type=int, default=8)
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


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_local_reference(local_dir: Path) -> dict[str, object]:
    manifest_path = local_dir / "manifest.json"
    artifact_path = local_dir / "complete_query.json"
    manifest = _load_json(manifest_path)
    artifact = _load_json(artifact_path)
    if (
        manifest.get("schema_version") != LOCAL_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != {"complete_query.json": file_sha256(artifact_path)}
        or artifact.get("schema_version") != LOCAL_ARTIFACT_SCHEMA_VERSION
        or artifact.get("status") != "ok"
    ):
        raise ValueError("NRIR-15 local reference artifact differs")
    evidence = _mapping(artifact.get("evidence"), "local reference evidence")
    if manifest.get("evidence_hash") != complete_query_hash(evidence):
        raise ValueError("NRIR-15 local reference evidence hash differs")
    fixed = _mapping(evidence.get("fixed_resnet"), "local fixed ResNet")
    query = _mapping(fixed.get("query_trace"), "local query trace")
    clauses = _list(fixed.get("clauses"), "local clauses")
    rows: list[dict[str, object]] = []
    for index, raw_clause in enumerate(clauses):
        clause = _mapping(raw_clause, "local clause")
        queue = _mapping(clause.get("queue_trace"), "local queue")
        evaluations = _list(queue.get("evaluations"), "local evaluations")
        root = _mapping(evaluations[0], "local root evaluation")
        search = _mapping(clause.get("search_trace"), "local search")
        verdict = _mapping(clause.get("verdict_trace"), "local verdict")
        rows.append(
            {
                "clause_index": index,
                "lower": float(root["lower"]),
                "candidate_best": float(search["best_objective_value"]),
                "threshold": 0.0,
                "status": verdict["status"],
                "proof_deficit": max(0.0, -float(root["lower"])),
            }
        )
    if (
        len(rows) != 9
        or query.get("status") != "unknown"
        or query.get("unresolved_clause_indices") != list(range(9))
        or any(row["status"] != "unknown" for row in rows)
    ):
        raise ValueError("NRIR-15 local reference boundary differs")
    return {
        "artifact_schema_version": LOCAL_ARTIFACT_SCHEMA_VERSION,
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "artifact_sha256": file_sha256(artifact_path),
        "evidence_hash": manifest["evidence_hash"],
        "status": query["status"],
        "unresolved_clause_indices": query["unresolved_clause_indices"],
        "rows": rows,
    }


def _build_context(
    *, model: Path, source_artifact_dir: Path, local_artifact_dir: Path
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-15 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    external_items = deserialize_intermediate_bounds(
        _mapping(payload.get("external_intermediate_bounds"), "intermediate bounds")
    )
    if intermediate_bounds_sha256(external_items) != INTERMEDIATE_BOUNDS_SHA256:
        raise ValueError("NRIR-15 external intermediate digest differs")
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    if tuple(op.op_type for op in module.get_entry_task().ops) != EXPECTED_PRIMAL_OPS:
        raise ValueError("NRIR-15 primal topology differs")
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    _interval_env, local_relu_pre = _forward_ibp_trace_mlp(module, input_spec)
    external_relu_pre = bind_intermediate_bounds(external_items, local_relu_pre)
    local_reference = _load_local_reference(local_artifact_dir)
    return {
        "module": module,
        "input_spec": input_spec,
        "tensors": tensors,
        "external_relu_pre": external_relu_pre,
        "local_reference": local_reference,
        "setup_ns": time.perf_counter_ns() - started_ns,
        "source": {
            "native_ir_artifact_schema": source_manifest["schema_version"],
            "native_ir_manifest_sha256": file_sha256(
                source_artifact_dir / "manifest.json"
            ),
            "native_ir_payload_sha256": file_sha256(source_artifact_dir / "payload.pt"),
            "local_complete_query_manifest_sha256": local_reference[
                "artifact_manifest_sha256"
            ],
            "local_complete_query_artifact_sha256": local_reference["artifact_sha256"],
            "model_sha256": MODEL_SHA256,
            "vnnlib_sha256": VNNLIB_SHA256,
            "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
            "vnncomp_commit": VNNCOMP_COMMIT,
            "abcrown_commit": ABCROWN_COMMIT,
            "input_lower_hash": tensor_content_hash(tensors["input_lower"]),
            "input_upper_hash": tensor_content_hash(tensors["input_upper"]),
            "objective_matrix_hash": tensor_content_hash(tensors["linear_spec_c"]),
            "external_oracle_lower_hash": tensor_content_hash(
                tensors["external_lower"]
            ),
        },
    }


def _external_semantic_summary(context: Mapping[str, Any]) -> dict[str, object]:
    tensors = _mapping(context["tensors"], "context tensors")
    execution = execute_complete_verifier_query(
        context["module"],
        context["input_spec"],
        linear_spec_C=tensors["linear_spec_c"],
        thresholds=torch.zeros(9, dtype=tensors["linear_spec_c"].dtype),
        query_id=QUERY_ID,
        query_policy=CompleteVerifierQueryPolicy(),
        search_policy=SEARCH_POLICY,
        queue_config=QUEUE_CONFIG,
        optimizer_policy=EXTERNAL_ADAPTIVE_POLICY,
        relu_pre_override=context["external_relu_pre"],
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
    )
    expected = tensors["external_lower"].reshape(-1)
    rows: list[dict[str, object]] = []
    native: list[Any] = []
    for clause in execution.clauses:
        root = clause.queue.trace.evaluations[0]
        lower = float(root.lower)
        oracle = float(expected[clause.trace.clause_index].item())
        rows.append(
            {
                "clause_index": clause.trace.clause_index,
                "lower": lower,
                "external_initial_lower": oracle,
                "improvement_over_external_initial": lower - oracle,
                "candidate_best": clause.search.trace.best_objective_value,
                "threshold": clause.trace.threshold,
                "status": clause.trace.status,
                "proof_deficit": max(0.0, clause.trace.threshold - lower),
                "queue_trace_hash": clause.queue.trace.stable_hash(),
                "selected_state_hash": root.selected_state_hash,
            }
        )
        native.extend(clause.queue.trace.native_stacks)
    lower_tensor = torch.tensor([row["lower"] for row in rows], dtype=expected.dtype)
    comparison = {
        "native_never_weaker_than_external_initial_with_tolerance": bool(
            torch.all(lower_tensor >= expected - EXTERNAL_ORACLE_ATOL)
        ),
        "sign_agreement": int(((lower_tensor >= 0) == (expected >= 0)).sum()),
        "sign_total": int(expected.numel()),
        "max_regression": float(torch.clamp(expected - lower_tensor, min=0).max()),
        "max_improvement": float((lower_tensor - expected).max()),
        "atol": EXTERNAL_ORACLE_ATOL,
        "rtol": EXTERNAL_ORACLE_RTOL,
    }
    return {
        "query_trace": execution.trace.to_dict(),
        "query_trace_hash": execution.trace.stable_hash(),
        "rows": rows,
        "comparison": comparison,
        "verified_clause_indices": [
            row["clause_index"] for row in rows if row["status"] == "verified"
        ],
        "unresolved_clause_indices": [
            row["clause_index"] for row in rows if row["status"] == "unknown"
        ],
        "max_selected_native_lower_abs_diff": max(
            item.selected_native_lower_max_abs_diff for item in native
        ),
        "max_selected_native_upper_abs_diff": max(
            item.selected_native_upper_max_abs_diff for item in native
        ),
    }


def _variant_arguments(
    context: Mapping[str, Any], variant: str
) -> tuple[
    NativeAlphaBetaOptimizerPolicy, Mapping[str, Any] | None, IntermediateBoundSource
]:
    if variant == "local_constant":
        return LOCAL_POLICY, None, IntermediateBoundSource.LOCAL_FORWARD
    if variant == "external_constant":
        return (
            EXTERNAL_CONSTANT_POLICY,
            context["external_relu_pre"],
            IntermediateBoundSource.EXTERNAL_VERIFIER,
        )
    if variant == "external_adaptive":
        return (
            EXTERNAL_ADAPTIVE_POLICY,
            context["external_relu_pre"],
            IntermediateBoundSource.EXTERNAL_VERIFIER,
        )
    raise ValueError(f"unknown NRIR-15 variant: {variant}")


def _run_timed_queue(
    context: Mapping[str, Any], *, variant: str, suffix: str
) -> tuple[Any, int]:
    tensors = _mapping(context["tensors"], "context tensors")
    policy, relu_pre, source = _variant_arguments(context, variant)
    objective = tensors["linear_spec_c"][
        :, TIMING_CLAUSE_INDEX : TIMING_CLAUSE_INDEX + 1
    ]
    started_ns = time.perf_counter_ns()
    execution = execute_native_optimized_relu_split_bab(
        context["module"],
        context["input_spec"],
        linear_spec_C=objective,
        run_id=f"nrir15-timing:{suffix}:{variant}",
        config=QUEUE_CONFIG,
        optimizer_policy=policy,
        relu_pre_override=relu_pre,
        intermediate_bound_source=source,
    )
    return execution, time.perf_counter_ns() - started_ns


def _percentile_90(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def _timing_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    phases: dict[str, list[int]] = {}
    for sample in samples:
        key = f"{sample['phase']}:{sample.get('variant', 'shared')}"
        phases.setdefault(key, []).append(cast(int, sample["elapsed_ns"]))
    return {
        key: {
            "runs": len(values),
            "raw_ns": values,
            "median_ns": int(statistics.median(values)),
            "p90_ns": _percentile_90(values),
        }
        for key, values in sorted(phases.items())
    }


def _measure_timing(context: Mapping[str, Any]) -> dict[str, object]:
    tensors = _mapping(context["tensors"], "context tensors")
    objective = tensors["linear_spec_c"][:, TIMING_CLAUSE_INDEX, :].contiguous()
    threshold = 0.0

    warm_search = search_native_box_counterexample(
        context["module"],
        context["input_spec"],
        linear_spec_C=objective,
        threshold=threshold,
        policy=SEARCH_POLICY,
    )
    for variant in VARIANT_ORDERS[0]:
        queue, _elapsed_ns = _run_timed_queue(context, variant=variant, suffix="warmup")
        derive_native_property_verdict(
            context["module"],
            context["input_spec"],
            linear_spec_C=objective,
            queue_execution=queue,
            candidate_counterexamples=(
                (queue.trace.evaluations[0].node.node_id, warm_search.best_input),
            ),
        )

    samples: list[dict[str, object]] = []
    for group, order in enumerate(VARIANT_ORDERS):
        started_ns = time.perf_counter_ns()
        search = search_native_box_counterexample(
            context["module"],
            context["input_spec"],
            linear_spec_C=objective,
            threshold=threshold,
            policy=SEARCH_POLICY,
        )
        samples.append(
            {
                "group": group,
                "phase": "candidate_search",
                "variant": "shared",
                "order": list(order),
                "elapsed_ns": time.perf_counter_ns() - started_ns,
                "best_objective_value": search.trace.best_objective_value,
            }
        )
        for position, variant in enumerate(order):
            queue, elapsed_ns = _run_timed_queue(
                context, variant=variant, suffix=f"group:{group}:position:{position}"
            )
            root = queue.trace.evaluations[0]
            samples.append(
                {
                    "group": group,
                    "phase": "audit_queue",
                    "variant": variant,
                    "position": position,
                    "elapsed_ns": elapsed_ns,
                    "root_lower": root.lower,
                    "queue_trace_hash": queue.trace.stable_hash(),
                }
            )
            started_ns = time.perf_counter_ns()
            verdict = derive_native_property_verdict(
                context["module"],
                context["input_spec"],
                linear_spec_C=objective,
                queue_execution=queue,
                candidate_counterexamples=((root.node.node_id, search.best_input),),
            )
            samples.append(
                {
                    "group": group,
                    "phase": "verdict",
                    "variant": variant,
                    "position": position,
                    "elapsed_ns": time.perf_counter_ns() - started_ns,
                    "status": verdict.trace.status,
                    "verdict_trace_hash": verdict.trace.stable_hash(queue.trace),
                }
            )
    return {
        "performance_claimed": False,
        "measurement_kind": "CPU audit-path diagnostic; not production latency",
        "torch_threads": torch.get_num_threads(),
        "warmups": TIMING_WARMUPS,
        "groups": TIMING_GROUPS,
        "timing_clause_index": TIMING_CLAUSE_INDEX,
        "variant_orders": [list(order) for order in VARIANT_ORDERS],
        "setup_ns": int(context["setup_ns"]),
        "samples": samples,
        "summary": _timing_summary(samples),
    }


def build_evidence(
    *, model: Path, source_artifact_dir: Path, local_artifact_dir: Path
) -> dict[str, object]:
    context = _build_context(
        model=model,
        source_artifact_dir=source_artifact_dir,
        local_artifact_dir=local_artifact_dir,
    )
    external = _external_semantic_summary(context)
    comparison = _mapping(external["comparison"], "external comparison")
    gates = {
        "source_model_property_and_intermediates_are_digest_bound": True,
        "local_reference_remains_nine_of_nine_unknown": bool(
            context["local_reference"]["status"] == "unknown"
            and context["local_reference"]["unresolved_clause_indices"]
            == list(range(9))
        ),
        "external_semantics_restore_six_verified_three_unknown": bool(
            external["verified_clause_indices"] == [1, 3, 5, 6, 7, 8]
            and external["unresolved_clause_indices"] == [0, 2, 4]
        ),
        "adaptive_optimizer_never_regresses_frozen_external_initial": bool(
            comparison["native_never_weaker_than_external_initial_with_tolerance"]
            is True
            and comparison["sign_agreement"] == comparison["sign_total"] == 9
        ),
        "external_optimizer_and_selected_native_stacks_agree": bool(
            cast(float, external["max_selected_native_lower_abs_diff"]) <= 2e-3
            and cast(float, external["max_selected_native_upper_abs_diff"]) <= 2e-3
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-15 semantic gates failed: {gates}")
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "property_status": "validated_reduced",
        "claim_boundary": (
            "fixed ResNet CPU diagnosis showing external-intermediate propagation "
            "restores six of nine root proofs and exposing three-group audit-path phase "
            "samples; not production latency, CUDA, multi-workload, competitor speedup, "
            "or complete verification evidence"
        ),
        "source": context["source"],
        "protocol": {
            "query_id": QUERY_ID,
            "search_policy": SEARCH_POLICY.to_dict(),
            "queue_config": QUEUE_CONFIG.to_dict(),
            "local_policy": LOCAL_POLICY.to_dict(),
            "external_constant_policy": EXTERNAL_CONSTANT_POLICY.to_dict(),
            "external_adaptive_policy": EXTERNAL_ADAPTIVE_POLICY.to_dict(),
            "intermediate_bound_source": "external_verifier",
            "groups": TIMING_GROUPS,
            "warmups": TIMING_WARMUPS,
            "variant_orders": [list(order) for order in VARIANT_ORDERS],
        },
        "semantic": {
            "local_constant_frozen_reference": context["local_reference"],
            "external_adaptive": external,
        },
        "timing": _measure_timing(context),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
        },
        "gates": gates,
        "limitations": [
            "only one fixed ResNet/VNNLIB pair is measured",
            "CPU audit path recompiles and re-executes selected native state for validation",
            "timing is diagnostic and is not a production-path latency or speedup claim",
            "six root proofs still leave clauses 0, 2, and 4 unresolved",
            "CUDA, multi-workload, and fastest-competitor end-to-end comparisons remain pending",
        ],
    }
    validate_evidence(evidence)
    return evidence


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("property_status") != "validated_reduced"
    ):
        raise ValueError("NRIR-15 evidence header differs")
    claim = str(evidence.get("claim_boundary", ""))
    for phrase in (
        "fixed ResNet CPU diagnosis",
        "not production latency",
        "competitor speedup",
        "complete verification evidence",
    ):
        if phrase not in claim:
            raise ValueError("NRIR-15 claim boundary differs")
    source = _mapping(evidence.get("source"), "NRIR-15 source")
    expected_source = {
        "model_sha256": MODEL_SHA256,
        "vnnlib_sha256": VNNLIB_SHA256,
        "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
        "vnncomp_commit": VNNCOMP_COMMIT,
        "abcrown_commit": ABCROWN_COMMIT,
    }
    if any(source.get(name) != value for name, value in expected_source.items()) or any(
        not _sha256(source.get(name))
        for name in (
            "native_ir_manifest_sha256",
            "native_ir_payload_sha256",
            "local_complete_query_manifest_sha256",
            "local_complete_query_artifact_sha256",
            "input_lower_hash",
            "input_upper_hash",
            "objective_matrix_hash",
            "external_oracle_lower_hash",
        )
    ):
        raise ValueError("NRIR-15 source identity differs")
    protocol = _mapping(evidence.get("protocol"), "NRIR-15 protocol")
    if (
        protocol.get("query_id") != QUERY_ID
        or protocol.get("groups") != TIMING_GROUPS
        or protocol.get("warmups") != TIMING_WARMUPS
        or protocol.get("variant_orders") != [list(order) for order in VARIANT_ORDERS]
        or protocol.get("queue_config") != QUEUE_CONFIG.to_dict()
        or protocol.get("local_policy") != LOCAL_POLICY.to_dict()
        or protocol.get("external_adaptive_policy")
        != EXTERNAL_ADAPTIVE_POLICY.to_dict()
    ):
        raise ValueError("NRIR-15 protocol differs")
    semantic = _mapping(evidence.get("semantic"), "NRIR-15 semantic")
    local = _mapping(semantic.get("local_constant_frozen_reference"), "NRIR-15 local")
    external = _mapping(semantic.get("external_adaptive"), "NRIR-15 external")
    query = _mapping(external.get("query_trace"), "NRIR-15 query")
    rows = _list(external.get("rows"), "NRIR-15 rows")
    comparison = _mapping(external.get("comparison"), "NRIR-15 comparison")
    if (
        local.get("status") != "unknown"
        or local.get("unresolved_clause_indices") != list(range(9))
        or len(_list(local.get("rows"), "NRIR-15 local rows")) != 9
        or query.get("status") != "unknown"
        or query.get("unresolved_clause_indices") != [0, 2, 4]
        or external.get("verified_clause_indices") != [1, 3, 5, 6, 7, 8]
        or external.get("unresolved_clause_indices") != [0, 2, 4]
        or len(rows) != 9
        or external.get("query_trace_hash") != canonical_hash(query)
        or comparison.get("native_never_weaker_than_external_initial_with_tolerance")
        is not True
        or comparison.get("sign_agreement") != 9
        or comparison.get("sign_total") != 9
    ):
        raise ValueError("NRIR-15 semantic boundary differs")
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, "NRIR-15 row")
        if (
            row.get("clause_index") != index
            or not all(
                isinstance(row.get(name), (int, float))
                and math.isfinite(float(row[name]))
                for name in (
                    "lower",
                    "external_initial_lower",
                    "improvement_over_external_initial",
                    "candidate_best",
                    "threshold",
                    "proof_deficit",
                )
            )
            or not _sha256(row.get("queue_trace_hash"))
            or not _sha256(row.get("selected_state_hash"))
            or row.get("status")
            != (
                "verified"
                if float(row["lower"]) >= float(row["threshold"])
                else "unknown"
            )
        ):
            raise ValueError("NRIR-15 clause row differs")
    timing = _mapping(evidence.get("timing"), "NRIR-15 timing")
    samples = _list(timing.get("samples"), "NRIR-15 timing samples")
    if (
        timing.get("performance_claimed") is not False
        or timing.get("groups") != TIMING_GROUPS
        or timing.get("warmups") != TIMING_WARMUPS
        or timing.get("variant_orders") != [list(order) for order in VARIANT_ORDERS]
        or len(samples) != TIMING_GROUPS * 7
    ):
        raise ValueError("NRIR-15 timing protocol differs")
    for sample in samples:
        if (
            sample.get("group") not in range(TIMING_GROUPS)
            or sample.get("phase") not in {"candidate_search", "audit_queue", "verdict"}
            or not isinstance(sample.get("elapsed_ns"), int)
            or int(sample["elapsed_ns"]) <= 0
        ):
            raise ValueError("NRIR-15 timing sample differs")
    if timing.get("summary") != _timing_summary(samples):
        raise ValueError("NRIR-15 timing summary differs")
    gates = _mapping(evidence.get("gates"), "NRIR-15 gates")
    if len(gates) != 5 or any(value is not True for value in gates.values()):
        raise ValueError("NRIR-15 gates differ")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 5:
        raise ValueError("NRIR-15 limitations differ")


def _generate(args: argparse.Namespace) -> None:
    evidence = build_evidence(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
        local_artifact_dir=args.local_artifact_dir,
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
        "evidence_hash": canonical_hash(evidence),
    }
    (args.artifact_dir / MANIFEST_FILE).write_text(
        _canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def _replay(args: argparse.Namespace) -> None:
    manifest_path = args.artifact_dir / MANIFEST_FILE
    artifact_path = args.artifact_dir / ARTIFACT_FILE
    manifest = _load_json(manifest_path)
    artifact = _load_json(artifact_path)
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != {ARTIFACT_FILE: file_sha256(artifact_path)}
        or artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or artifact.get("status") != "ok"
    ):
        raise ValueError("NRIR-15 artifact manifest/header differs")
    stored = _mapping(artifact.get("evidence"), "NRIR-15 stored evidence")
    validate_evidence(stored)
    if manifest.get("evidence_hash") != canonical_hash(stored):
        raise ValueError("NRIR-15 stored evidence hash differs")
    context = _build_context(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
        local_artifact_dir=args.local_artifact_dir,
    )
    actual_external = _external_semantic_summary(context)
    semantic = _mapping(stored.get("semantic"), "NRIR-15 replay semantic")
    if semantic.get("external_adaptive") != actual_external:
        raise ValueError("NRIR-15 semantic replay differs from frozen evidence")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-15 torch thread count must be positive")
    torch.set_num_threads(args.torch_threads)
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
