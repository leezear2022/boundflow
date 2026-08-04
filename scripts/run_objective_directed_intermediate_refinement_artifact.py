#!/usr/bin/env python3
"""Generate or replay the NRIR-20 objective-directed refinement artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions
# pylint: disable=duplicate-code,import-outside-toplevel,import-error
# pylint: disable=missing-function-docstring,line-too-long,protected-access
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_multiworkload_competitor_e2e_artifact import (
    VNNCOMP_COMMIT,
    WORKLOAD_ROWS,
    _csv_selection,
    _onnx_inventory,
    canonical_hash,
    file_sha256,
)
from scripts.run_native_intermediate_refinement_artifact import (
    _canonical_json,
    _git_revision,
    _load_json,
    _mapping,
    _sha256,
    _write_json,
)

ARTIFACT_SCHEMA_VERSION = (
    "boundflow.objective-directed-intermediate-refinement-artifact/v1"
)
EVIDENCE_SCHEMA_VERSION = (
    "boundflow.objective-directed-intermediate-refinement-evidence/v1"
)
WORKER_RESULT_SCHEMA_VERSION = (
    "boundflow.objective-directed-intermediate-refinement-worker/v1"
)
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
WORKLOAD_ID = "cifar10_resnet:000"
CLAUSE_INDICES = (0, 1)
MODES = ("width", "objective")
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 60
ALPHA_STEPS = 5
MAX_NODES = 1
MAX_DEPTH = 1
TARGETS_PER_RELU = 16
BACKWARD_CHUNK_SIZE = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--mode", choices=MODES, required=True)
    worker.add_argument("--clause-index", type=int, required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--input-shape", type=int, nargs="+", required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def _native_code_revision() -> str:
    paths = (
        "boundflow/ir/refinement.py",
        "boundflow/runtime/crown_ibp.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/runtime/native_optimized_relu_split_bab_runtime.py",
        "scripts/run_native_intermediate_refinement_artifact.py",
        "scripts/run_objective_directed_intermediate_refinement_artifact.py",
    )
    return canonical_hash({path: file_sha256(REPO_ROOT / path) for path in paths})


def _resolved_source(benchmark_root: Path) -> dict[str, object]:
    if _git_revision(benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("NRIR-20 VNN-COMP source revision differs")
    from boundflow.frontends.vnnlib import import_vnnlib_box_query

    definition = next(
        item for item in WORKLOAD_ROWS if item["workload_id"] == WORKLOAD_ID
    )
    csv_path, model_path, property_path = _csv_selection(benchmark_root, definition)
    input_shape, output_dim, onnx_ops = _onnx_inventory(model_path)
    query = import_vnnlib_box_query(property_path, query_id=WORKLOAD_ID)
    return {
        "vnncomp_commit": VNNCOMP_COMMIT,
        "native_code_revision": _native_code_revision(),
        "workload_id": WORKLOAD_ID,
        "csv_path": csv_path,
        "model_path": model_path,
        "property_path": property_path,
        "csv_sha256": file_sha256(csv_path),
        "model_sha256": file_sha256(model_path),
        "property_sha256": file_sha256(property_path),
        "query_hash": query.stable_hash(),
        "input_shape": input_shape,
        "output_dim": output_dim,
        "onnx_ops": onnx_ops,
    }


def _source_payload(source: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in source.items()
        if key not in {"csv_path", "model_path", "property_path"}
    }


def _policy_payload() -> dict[str, object]:
    return {
        "device": "cpu",
        "torch_threads": TORCH_THREADS,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "alpha_steps": ALPHA_STEPS,
        "max_nodes": MAX_NODES,
        "max_depth": MAX_DEPTH,
        "targets_per_relu": TARGETS_PER_RELU,
        "backward_chunk_size": BACKWARD_CHUNK_SIZE,
        "width_policy_id": "top_ambiguous_width_per_relu_v1",
        "objective_policy_id": "objective_influence_width_per_relu_v1",
        "selection_score": "ambiguous_width*max(abs(A_u),abs(A_l))",
        "performance_claimed": False,
    }


def _worker_command(
    *, mode: str, clause_index: int, source: Mapping[str, object], result: Path
) -> list[str]:
    input_shape = cast(tuple[int, ...], source["input_shape"])
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--mode",
        mode,
        "--clause-index",
        str(clause_index),
        "--model",
        str(source["model_path"]),
        "--property",
        str(source["property_path"]),
        "--input-shape",
        *(str(value) for value in input_shape[1:]),
        "--result-json",
        str(result),
    ]


def _run_worker(
    *, mode: str, clause_index: int, source: Mapping[str, object], result: Path
) -> tuple[dict[str, Any], str, int, int]:
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(
        _worker_command(
            mode=mode, clause_index=clause_index, source=source, result=result
        ),
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    elapsed_ns = time.perf_counter_ns() - started_ns
    if completed.returncode != 0 or not result.is_file():
        raise RuntimeError(
            f"NRIR-20 clause {clause_index} {mode} worker failed with "
            f"{completed.returncode}: {completed.stdout[-8000:]}"
        )
    return _load_json(result), completed.stdout, completed.returncode, elapsed_ns


def _refinement_policy(mode: str) -> Any:
    from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR

    return NativeIntermediateRefinementPolicyIR(
        passes=1,
        max_neurons_per_relu=TARGETS_PER_RELU,
        backward_chunk_size=BACKWARD_CHUNK_SIZE,
        candidate_policy_id=(
            "objective_influence_width_per_relu_v1"
            if mode == "objective"
            else "top_ambiguous_width_per_relu_v1"
        ),
    )


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.frontends.vnnlib import (
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
    from boundflow.ir.bound import IntermediateBoundSource
    from boundflow.planner import plan_interval_ibp_v0
    from boundflow.runtime.native_alpha_beta_optimization_state import (
        NativeAlphaBetaOptimizerPolicy,
    )
    from boundflow.runtime.native_intermediate_refinement import (
        compile_native_intermediate_refinement_program,
        execute_native_intermediate_refinement_program,
    )
    from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
        execute_native_optimized_relu_split_bab,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import (
        NativeReluSplitBabConfig,
    )
    from boundflow.runtime.task_executor import InputSpec

    if args.clause_index not in CLAUSE_INDICES:
        raise ValueError("NRIR-20 clause index is outside the frozen set")
    torch.set_num_threads(TORCH_THREADS)
    started_ns = time.perf_counter_ns()
    query = import_vnnlib_box_query(args.property, query_id=WORKLOAD_ID)
    tensors = materialize_vnnlib_box_query(query, input_shape=tuple(args.input_shape))
    primal = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(primal)
    input_spec = InputSpec.box(
        value_name=primal.graph.inputs[0],
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    objective = tensors.linear_spec_c[:, args.clause_index, :].contiguous()
    refinement_started_ns = time.perf_counter_ns()
    program = compile_native_intermediate_refinement_program(
        module,
        input_spec,
        policy=_refinement_policy(args.mode),
        plan_id=f"nrir20:{WORKLOAD_ID}:clause:{args.clause_index:04d}:{args.mode}",
        linear_spec_C=objective if args.mode == "objective" else None,
    )
    execution = execute_native_intermediate_refinement_program(
        program, module, input_spec
    )
    refinement_ns = time.perf_counter_ns() - refinement_started_ns
    queue_started_ns = time.perf_counter_ns()
    queue = execute_native_optimized_relu_split_bab(
        module,
        input_spec,
        linear_spec_C=objective,
        run_id=f"nrir20:{args.clause_index:04d}:{args.mode}",
        config=NativeReluSplitBabConfig(
            max_nodes=MAX_NODES,
            max_depth=MAX_DEPTH,
            expansion_batch_size=1,
            max_eval_batch_size=1,
            threshold=float(tensors.thresholds[args.clause_index]),
        ),
        optimizer_policy=NativeAlphaBetaOptimizerPolicy(
            steps=ALPHA_STEPS,
            lr=0.1,
            alpha_initialization_mode="adaptive",
        ),
        relu_pre_override=execution.relu_pre,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
    )
    queue_ns = time.perf_counter_ns() - queue_started_ns
    root = queue.trace.evaluations[0]
    trace = execution.trace.to_dict()
    result = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "workload_id": WORKLOAD_ID,
        "clause_index": args.clause_index,
        "mode": args.mode,
        "execution_state": "completed",
        "threshold": float(tensors.thresholds[args.clause_index]),
        "refinement": {
            "plan": program.plan.to_dict(),
            "task": program.task_module.to_dict(plan=program.plan),
            "schedule": program.schedule.to_dict(
                plan=program.plan, task_module=program.task_module
            ),
            "hashes": program.hashes(),
            "execution_trace": trace,
            "execution_trace_hash": canonical_hash(trace),
        },
        "target_identities": [
            [target.relu_input, target.neuron_index] for target in program.plan.targets
        ],
        "target_count": len(program.plan.targets),
        "root_lower": root.lower,
        "root_upper": root.upper,
        "queue_trace": queue.trace.to_dict(),
        "queue_trace_hash": queue.trace.stable_hash(),
        "refinement_ns": refinement_ns,
        "queue_ns": queue_ns,
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "performance_claimed": False,
    }
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok",
                "clause_index": args.clause_index,
                "mode": args.mode,
                "root_lower": root.lower,
            }
        )
    )


def _comparison(
    clause_index: int,
    width: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> dict[str, object]:
    width_targets = {tuple(item) for item in width["target_identities"]}
    objective_targets = {tuple(item) for item in objective["target_identities"]}
    return {
        "clause_index": clause_index,
        "width_target_count": width["target_count"],
        "objective_target_count": objective["target_count"],
        "target_overlap_count": len(width_targets & objective_targets),
        "target_union_count": len(width_targets | objective_targets),
        "width_root_lower": width["root_lower"],
        "objective_root_lower": objective["root_lower"],
        "root_lower_delta": float(objective["root_lower"]) - float(width["root_lower"]),
        "width_root_upper": width["root_upper"],
        "objective_root_upper": objective["root_upper"],
        "performance_claimed": False,
    }


def _build_evidence(
    benchmark_root: Path, artifact_dir: Path
) -> tuple[dict[str, object], dict[str, str]]:
    source = _resolved_source(benchmark_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir20-") as temporary:
        temp_root = Path(temporary)
        for clause_index in CLAUSE_INDICES:
            for mode in MODES:
                result_path = temp_root / f"clause-{clause_index}-{mode}.json"
                result, log, returncode, elapsed_ns = _run_worker(
                    mode=mode,
                    clause_index=clause_index,
                    source=source,
                    result=result_path,
                )
                log_path = artifact_dir / "logs" / f"clause-{clause_index}-{mode}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(log, encoding="utf-8")
                relative_log = str(log_path.relative_to(artifact_dir))
                files[relative_log] = file_sha256(log_path)
                records.append(
                    {
                        "clause_index": clause_index,
                        "mode": mode,
                        "process_returncode": returncode,
                        "e2e_elapsed_ns": elapsed_ns,
                        "log_path": relative_log,
                        "log_sha256": files[relative_log],
                        "result": result,
                    }
                )
    by_identity = {
        (cast(int, record["clause_index"]), str(record["mode"])): cast(
            Mapping[str, Any], record["result"]
        )
        for record in records
    }
    comparisons = [
        _comparison(
            clause_index,
            by_identity[(clause_index, "width")],
            by_identity[(clause_index, "objective")],
        )
        for clause_index in CLAUSE_INDICES
    ]
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "claim_boundary": (
            "same-budget clause-sensitive objective-directed versus width-only "
            "native intermediate refinement on two fixed VNN-COMP ResNet clauses; "
            "root-bound tightness only; diagnostic timings; no speedup or closure claim"
        ),
        "source": _source_payload(source),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": TORCH_THREADS,
        },
        "policy": _policy_payload(),
        "records": records,
        "comparisons": comparisons,
        "limitations": [
            "The fixed experiment evaluates two clauses and one root node per mode; it does not claim complete property closure.",
            "Objective-directed selection is root-global; per-child influence and intermediate-bound recomputation are not implemented.",
            "The ranking heuristic need not improve every objective; soundness comes from selected CROWN and interval intersection.",
            "CPU timing is one fresh-process diagnostic observation per clause/mode and is not a performance claim.",
            "CUDA, competitor parity, repeated timing, and ASPLOS-ready claims remain pending.",
        ],
        "performance_claimed": False,
    }
    return evidence, files


def _validate_worker_result(result: Mapping[str, Any]) -> None:
    mode = result.get("mode")
    refinement = _mapping(result.get("refinement"), "NRIR-20 refinement")
    plan = _mapping(refinement.get("plan"), "NRIR-20 Plan")
    task = _mapping(refinement.get("task"), "NRIR-20 Task")
    schedule = _mapping(refinement.get("schedule"), "NRIR-20 Schedule")
    hashes = _mapping(refinement.get("hashes"), "NRIR-20 hashes")
    trace = _mapping(refinement.get("execution_trace"), "NRIR-20 trace")
    policy = _mapping(plan.get("policy"), "NRIR-20 policy")
    targets = plan.get("targets")
    if (
        result.get("schema_version") != WORKER_RESULT_SCHEMA_VERSION
        or result.get("workload_id") != WORKLOAD_ID
        or result.get("clause_index") not in CLAUSE_INDICES
        or mode not in MODES
        or result.get("execution_state") != "completed"
        or result.get("performance_claimed") is not False
        or result.get("target_count") != 96
        or not isinstance(targets, list)
        or len(targets) != result.get("target_count")
        or not isinstance(result.get("root_lower"), (float, int))
        or not isinstance(result.get("root_upper"), (float, int))
        or float(result["root_lower"]) > float(result["root_upper"])
        or hashes.get("refinement_plan_hash") != canonical_hash(plan)
        or hashes.get("refinement_task_module_hash") != canonical_hash(task)
        or hashes.get("refinement_schedule_hash") != canonical_hash(schedule)
        or task.get("refinement_plan_hash") != hashes.get("refinement_plan_hash")
        or schedule.get("refinement_plan_hash") != hashes.get("refinement_plan_hash")
        or schedule.get("refinement_task_module_hash")
        != hashes.get("refinement_task_module_hash")
        or trace.get("plan_hash") != hashes.get("refinement_plan_hash")
        or trace.get("task_module_hash") != hashes.get("refinement_task_module_hash")
        or trace.get("schedule_hash") != hashes.get("refinement_schedule_hash")
        or refinement.get("execution_trace_hash") != canonical_hash(trace)
        or result.get("queue_trace_hash") != canonical_hash(result.get("queue_trace"))
    ):
        raise ValueError("NRIR-20 worker IR/trace linkage differs")
    objective_mode = mode == "objective"
    if objective_mode != (plan.get("objective_hash") is not None) or (
        objective_mode
        != (
            policy.get("candidate_policy_id") == "objective_influence_width_per_relu_v1"
        )
    ):
        raise ValueError("NRIR-20 objective policy binding differs")
    for target in targets:
        item = _mapping(target, "NRIR-20 target")
        has_score = "objective_influence" in item and "selection_score" in item
        if objective_mode != has_score:
            raise ValueError("NRIR-20 target scoring semantics differ")
        if has_score and abs(
            float(item["selection_score"])
            - float(item["objective_influence"]) * float(item["initial_width"])
        ) > max(1e-7, 1e-6 * float(item["selection_score"])):
            raise ValueError("NRIR-20 target score differs")
    select_task = cast(list[Mapping[str, Any]], task["tasks"])[2]
    expected_inputs = (
        [
            "refine.bounds.p0",
            "refine.candidates",
            "refine.policy",
            "refine.objective_influence",
        ]
        if objective_mode
        else ["refine.candidates", "refine.policy"]
    )
    if select_task.get("input_value_ids") != expected_inputs:
        raise ValueError("NRIR-20 target-selection dependency differs")


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or "no speedup" not in str(evidence.get("claim_boundary"))
    ):
        raise ValueError("NRIR-20 evidence header differs")
    source = _mapping(evidence.get("source"), "NRIR-20 source")
    if (
        source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("workload_id") != WORKLOAD_ID
        or not _sha256(source.get("native_code_revision"))
        or not _sha256(source.get("model_sha256"))
        or not _sha256(source.get("property_sha256"))
    ):
        raise ValueError("NRIR-20 source identity differs")
    records = evidence.get("records")
    if not isinstance(records, list):
        raise TypeError("NRIR-20 records must be a list")
    identities: set[tuple[int, str]] = set()
    for record in records:
        item = _mapping(record, "NRIR-20 record")
        identity = (int(item["clause_index"]), str(item["mode"]))
        result = _mapping(item.get("result"), "NRIR-20 result")
        _validate_worker_result(result)
        if (
            identity in identities
            or identity != (int(result["clause_index"]), str(result["mode"]))
            or item.get("process_returncode") != 0
            or not isinstance(item.get("e2e_elapsed_ns"), int)
            or int(item["e2e_elapsed_ns"]) <= 0
            or not str(item.get("log_path", "")).startswith("logs/")
            or not _sha256(item.get("log_sha256"))
        ):
            raise ValueError("NRIR-20 execution record differs")
        identities.add(identity)
    if identities != {
        (clause_index, mode) for clause_index in CLAUSE_INDICES for mode in MODES
    }:
        raise ValueError("NRIR-20 record coverage differs")
    comparisons = evidence.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != len(CLAUSE_INDICES):
        raise ValueError("NRIR-20 comparison coverage differs")
    for comparison in comparisons:
        item = _mapping(comparison, "NRIR-20 comparison")
        if (
            item.get("width_target_count") != item.get("objective_target_count")
            or item.get("width_target_count") != 96
            or float(item.get("root_lower_delta", 0.0)) <= 0.0
            or item.get("performance_claimed") is not False
        ):
            raise ValueError("NRIR-20 same-budget tightness comparison differs")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 5:
        raise ValueError("NRIR-20 limitation ledger differs")


def _generate(args: argparse.Namespace) -> None:
    evidence, files = _build_evidence(args.benchmark_root, args.artifact_dir)
    validate_evidence_structure(evidence)
    evidence_path = args.artifact_dir / EVIDENCE_FILE
    _write_json(evidence_path, evidence)
    files[EVIDENCE_FILE] = file_sha256(evidence_path)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "files": dict(sorted(files.items())),
        "evidence_hash": canonical_hash(evidence),
    }
    _write_json(args.artifact_dir / MANIFEST_FILE, manifest)
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def _deterministic_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(trace)
    result.pop("elapsed_ns", None)
    return result


def _recompile_refinement_ir(
    benchmark_root: Path, evidence: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    import torch

    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.frontends.vnnlib import (
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
    from boundflow.planner import plan_interval_ibp_v0
    from boundflow.runtime.native_intermediate_refinement import (
        compile_native_intermediate_refinement_program,
        execute_native_intermediate_refinement_program,
    )
    from boundflow.runtime.task_executor import InputSpec

    torch.set_num_threads(TORCH_THREADS)
    source = _resolved_source(benchmark_root)
    query = import_vnnlib_box_query(
        cast(Path, source["property_path"]), query_id=WORKLOAD_ID
    )
    input_shape = cast(tuple[int, ...], source["input_shape"])
    tensors = materialize_vnnlib_box_query(query, input_shape=input_shape[1:])
    primal = import_onnx(str(source["model_path"]), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(primal)
    input_spec = InputSpec.box(
        value_name=primal.graph.inputs[0],
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    hashes: dict[str, dict[str, str]] = {}
    for record in cast(list[Mapping[str, Any]], evidence["records"]):
        clause_index = int(record["clause_index"])
        mode = str(record["mode"])
        objective = tensors.linear_spec_c[:, clause_index, :].contiguous()
        program = compile_native_intermediate_refinement_program(
            module,
            input_spec,
            policy=_refinement_policy(mode),
            plan_id=f"nrir20:{WORKLOAD_ID}:clause:{clause_index:04d}:{mode}",
            linear_spec_C=objective if mode == "objective" else None,
        )
        execution = execute_native_intermediate_refinement_program(
            program, module, input_spec
        )
        expected = _mapping(
            _mapping(record["result"], "result")["refinement"], "refinement"
        )
        if (
            program.plan.to_dict() != expected["plan"]
            or program.task_module.to_dict(plan=program.plan) != expected["task"]
            or program.schedule.to_dict(
                plan=program.plan, task_module=program.task_module
            )
            != expected["schedule"]
            or _deterministic_trace(execution.trace.to_dict())
            != _deterministic_trace(
                _mapping(expected["execution_trace"], "execution trace")
            )
        ):
            raise ValueError("NRIR-20 source-to-refinement semantic replay differs")
        hashes[f"clause-{clause_index}-{mode}"] = program.hashes()
    return hashes


def _replay(args: argparse.Namespace) -> None:
    manifest = _load_json(args.artifact_dir / MANIFEST_FILE)
    evidence = _load_json(args.artifact_dir / EVIDENCE_FILE)
    actual_files = {
        str(path.relative_to(args.artifact_dir)): file_sha256(path)
        for path in sorted(args.artifact_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != actual_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-20 artifact manifest differs")
    validate_evidence_structure(evidence)
    expected_source = _source_payload(_resolved_source(args.benchmark_root))
    if evidence.get("source") != expected_source:
        raise ValueError("NRIR-20 source replay differs")
    hashes = _recompile_refinement_ir(args.benchmark_root, evidence)
    print(
        _canonical_json(
            {
                "status": "replayed",
                "evidence_hash": manifest["evidence_hash"],
                "recompiled_refinement_ir_hashes": hashes,
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    elif args.command == "replay":
        _replay(args)
    else:
        _worker(args)


if __name__ == "__main__":
    main()
