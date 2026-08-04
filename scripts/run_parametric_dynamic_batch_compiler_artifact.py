#!/usr/bin/env python3
"""Generate or replay the NRIR-28 parametric compiler full-query artifact."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-arguments,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,duplicate-code
# pylint: disable=protected-access,import-error,line-too-long

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

from scripts.run_multiworkload_competitor_e2e_artifact import (
    VNNCOMP_COMMIT,
    WORKLOAD_ROWS,
    _csv_selection,
    _git_revision,
    _onnx_inventory,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.parametric-compiler-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.parametric-compiler-evidence/v1"
WORKER_SCHEMA_VERSION = "boundflow.parametric-compiler-worker/v1"
ARTIFACT_DIR = Path(
    "artifacts/parametric-dynamic-batch-compiler/vnncomp21-three-topology-cpu-v1"
)
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
REPEATS = 3
TORCH_THREADS = 8
TIMEOUT_SECONDS = 60
ALPHA_STEPS = 5
SEARCH_STEPS = 4
MAX_NODES = 7
MODES = ("production_v1", "parametric_v2")
MODE_ORDERS = (
    ("production_v1", "parametric_v2"),
    ("parametric_v2", "production_v1"),
    ("production_v1", "parametric_v2"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
        subparser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--mode", choices=MODES, required=True)
    worker.add_argument("--workload-id", required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--torch-threads", type=int, required=True)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/parametric_optimizer.py",
        "boundflow/runtime/native_parametric_optimizer.py",
        "boundflow/runtime/native_parametric_production_verifier.py",
        "boundflow/runtime/native_parametric_production_complete_query.py",
        "boundflow/runtime/native_production_verifier.py",
        "boundflow/runtime/native_production_complete_query.py",
        "scripts/run_parametric_dynamic_batch_compiler_artifact.py",
    )
    return canonical_hash({path: file_sha256(root / path) for path in paths})


def _resolve_workloads(benchmark_root: Path) -> list[dict[str, object]]:
    if _git_revision(benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("NRIR-28 VNN-COMP commit differs")
    resolved: list[dict[str, object]] = []
    for definition in WORKLOAD_ROWS:
        workload_id = str(definition["workload_id"])
        csv_path, model_path, property_path = _csv_selection(benchmark_root, definition)
        ordinal = definition["csv_ordinal"]
        if not isinstance(ordinal, int):
            raise TypeError("NRIR-28 CSV ordinal must be an integer")
        input_shape, output_dim, ops = _onnx_inventory(model_path)
        resolved.append(
            {
                "workload_id": workload_id,
                "category": str(definition["category"]),
                "csv_ordinal": ordinal,
                "csv_relative_path": str(definition["csv"]),
                "model_relative_path": str(definition["model"]),
                "property_relative_path": str(definition["property"]),
                "csv_sha256": file_sha256(csv_path),
                "model_sha256": file_sha256(model_path),
                "property_sha256": file_sha256(property_path),
                "model_input_shape": list(input_shape),
                "model_output_dim": output_dim,
                "onnx_ops": list(ops),
                "model": model_path,
                "property": property_path,
            }
        )
    return resolved


def _semantic_query(execution: Any) -> dict[str, object]:
    clauses = []
    semantic_clauses = []
    for clause in execution.clauses:
        queue_execution = clause.queue
        queue = queue_execution.trace
        selected_state_hashes = [item.selected_state_hash for item in queue.evaluations]
        signature_hash = canonical_hash(queue.logical_queue_signature())
        item = {
            "clause_index": clause.trace.clause_index,
            "status": clause.trace.status,
            "root_lower": queue.evaluations[0].lower,
            "root_upper": queue.evaluations[0].upper,
            "queue_status": queue.status,
            "evaluated_nodes": len(queue.evaluations),
            "logical_queue_signature_hash": signature_hash,
            "selected_state_hashes": selected_state_hashes,
            "queue_trace_hash": queue.stable_hash(),
        }
        clauses.append(item)
        semantic_clauses.append(
            {
                key: item[key]
                for key in (
                    "clause_index",
                    "status",
                    "root_lower",
                    "root_upper",
                    "queue_status",
                    "evaluated_nodes",
                    "logical_queue_signature_hash",
                    "selected_state_hashes",
                )
            }
        )
    semantic = {
        "solver_status": execution.trace.status,
        "completed_clause_count": len(execution.clauses),
        "unresolved_clause_indices": list(execution.trace.unresolved_clause_indices),
        "pending_clause_indices": list(execution.trace.pending_clause_indices),
        "clauses": semantic_clauses,
    }
    return {
        **semantic,
        "clauses": clauses,
        "query_trace_hash": execution.trace.stable_hash(),
        "semantic_signature_hash": canonical_hash(semantic),
    }


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.frontends.vnnlib import (
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
    from boundflow.planner import plan_interval_ibp_v0
    from boundflow.runtime.complete_verifier_query import CompleteVerifierQueryPolicy
    from boundflow.runtime.native_alpha_beta_optimization_state import (
        NativeAlphaBetaOptimizerPolicy,
    )
    from boundflow.runtime.native_candidate_search import (
        NativeProjectedGradientSearchPolicy,
    )
    from boundflow.runtime.native_parametric_production_complete_query import (
        execute_native_parametric_production_complete_verifier_query,
    )
    from boundflow.runtime.native_production_complete_query import (
        execute_native_production_complete_verifier_query,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import (
        NativeReluSplitBabConfig,
    )
    from boundflow.runtime.task_executor import InputSpec

    torch.set_num_threads(args.torch_threads)
    started_ns = time.perf_counter_ns()
    query = import_vnnlib_box_query(args.property, query_id=args.workload_id)
    input_shape, output_dim, _ops = _onnx_inventory(args.model)
    if len(query.output_names) != output_dim:
        raise ValueError("NRIR-28 worker output dimension differs")
    tensors = materialize_vnnlib_box_query(query, input_shape=input_shape[1:])
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    setup_ns = time.perf_counter_ns() - started_ns
    query_policy = CompleteVerifierQueryPolicy(timeout_ns=TIMEOUT_SECONDS * 10**9)
    search_policy = NativeProjectedGradientSearchPolicy(
        steps=SEARCH_STEPS, step_size=0.002
    )
    queue_config = NativeReluSplitBabConfig(
        max_nodes=MAX_NODES,
        max_depth=2,
        expansion_batch_size=2,
        max_eval_batch_size=4,
    )
    optimizer_policy = NativeAlphaBetaOptimizerPolicy(
        steps=ALPHA_STEPS,
        lr=0.1,
        alpha_initialization_mode="adaptive",
    )
    execute_started_ns = time.perf_counter_ns()
    execution: Any
    if args.mode == "production_v1":
        execution = execute_native_production_complete_verifier_query(
            module,
            input_spec,
            linear_spec_C=tensors.linear_spec_c,
            thresholds=tensors.thresholds,
            query_id=f"{query.query_id}:full",
            query_policy=query_policy,
            search_policy=search_policy,
            queue_config=queue_config,
            optimizer_policy=optimizer_policy,
        )
        compiler = None
    else:
        execution = execute_native_parametric_production_complete_verifier_query(
            module,
            input_spec,
            linear_spec_C=tensors.linear_spec_c,
            thresholds=tensors.thresholds,
            query_id=f"{query.query_id}:full",
            query_policy=query_policy,
            search_policy=search_policy,
            queue_config=queue_config,
            optimizer_policy=optimizer_policy,
        )
        compiler = {
            "cache": execution.compiler_cache_trace.to_dict(),
            "batches": [
                batch.to_dict()
                for clause in execution.clauses
                for batch in clause.queue.compiler_batches
            ],
        }
    execution_ns = time.perf_counter_ns() - execute_started_ns
    result = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "workload_id": args.workload_id,
        "mode": args.mode,
        "setup_ns": setup_ns,
        "execution_ns": execution_ns,
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "execution_mode": (
            "production_prepared_v1"
            if args.mode == "production_v1"
            else "production_parametric_template_instance_v2"
        ),
        "performance_claimed": False,
        "compiler": compiler,
        **_semantic_query(execution),
    }
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok",
                "mode": args.mode,
                "solver_status": execution.trace.status,
            }
        )
    )


def _worker_command(
    *, mode: str, workload: Mapping[str, object], result_path: Path, torch_threads: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--mode",
        mode,
        "--workload-id",
        str(workload["workload_id"]),
        "--model",
        str(workload["model"]),
        "--property",
        str(workload["property"]),
        "--result-json",
        str(result_path),
        "--torch-threads",
        str(torch_threads),
    ]


def _run_worker(
    *, mode: str, workload: Mapping[str, object], result_path: Path, torch_threads: int
) -> tuple[dict[str, Any], str, int]:
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(
        _worker_command(
            mode=mode,
            workload=workload,
            result_path=result_path,
            torch_threads=torch_threads,
        ),
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        timeout=TIMEOUT_SECONDS + 120,
        check=False,
    )
    elapsed_ns = time.perf_counter_ns() - started_ns
    log = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"NRIR-28 {mode} worker failed with {completed.returncode}: {log}"
        )
    return _load_json(result_path), log, elapsed_ns


def _p90(values: list[int]) -> int:
    return sorted(values)[-1]


def _summaries(records: list[dict[str, Any]]) -> dict[str, object]:
    summaries: dict[str, object] = {}
    workload_ids = sorted({str(record["workload_id"]) for record in records})
    for workload_id in workload_ids:
        modes: dict[str, dict[str, object]] = {}
        for mode in MODES:
            raw = [
                int(record["e2e_elapsed_ns"])
                for record in records
                if record["workload_id"] == workload_id and record["mode"] == mode
            ]
            modes[mode] = {
                "raw_e2e_ns": raw,
                "median_e2e_ns": int(statistics.median(raw)),
                "p90_e2e_ns": _p90(raw),
                "solver_statuses": [
                    str(record["result"]["solver_status"])
                    for record in records
                    if record["workload_id"] == workload_id and record["mode"] == mode
                ],
            }
        baseline_value = modes["production_v1"]["median_e2e_ns"]
        parametric_value = modes["parametric_v2"]["median_e2e_ns"]
        if not isinstance(baseline_value, int) or not isinstance(parametric_value, int):
            raise TypeError("NRIR-28 summary median must be an integer")
        baseline_median = baseline_value
        parametric_median = parametric_value
        summaries[workload_id] = {
            **modes,
            "internal_speedup": baseline_median / parametric_median,
            "semantic_parity": True,
            "performance_claimed": True,
        }
    return summaries


def _validate_compiler_payload(compiler: Mapping[str, Any]) -> None:
    cache = _mapping(compiler.get("cache"), "NRIR-28 compiler cache")
    templates = _list(cache.get("templates"), "NRIR-28 templates")
    events = _list(cache.get("events"), "NRIR-28 events")
    batches = _list(compiler.get("batches"), "NRIR-28 compiler batches")
    if (
        len(templates) != 1
        or cache.get("template_count") != 1
        or cache.get("instance_count") != len(events)
        or cache.get("miss_count") != 1
        or cache.get("hit_count") != len(events) - 1
        or len(events) != len(batches)
    ):
        raise ValueError("NRIR-28 cache accounting differs")
    template = _mapping(templates[0], "NRIR-28 template trace")
    template_ir = _mapping(template.get("template"), "NRIR-28 template IR")
    task_ir = _mapping(template.get("task_ir"), "NRIR-28 task IR")
    schedule = _mapping(template.get("schedule"), "NRIR-28 schedule IR")
    if (
        canonical_hash(template_ir) != template.get("template_hash")
        or canonical_hash(task_ir) != template.get("task_hash")
        or canonical_hash(schedule) != template.get("schedule_hash")
        or template_ir.get("cache_key")
        != canonical_hash(
            {
                key: value
                for key, value in template_ir.items()
                if key not in {"template_id", "cache_key", "performance_claimed"}
            }
        )
    ):
        raise ValueError("NRIR-28 template source-to-IR digest differs")
    template_hash = template["template_hash"]
    cache_key = template_ir["cache_key"]
    for index, (event_value, batch_value) in enumerate(zip(events, batches)):
        event = _mapping(event_value, "NRIR-28 cache event")
        batch = _mapping(batch_value, "NRIR-28 compiler batch")
        instance = _mapping(batch.get("instance"), "NRIR-28 instance")
        if (
            event.get("event_index") != index
            or event.get("cache_key") != cache_key
            or event.get("template_hash") != template_hash
            or batch.get("cache_event") != event
            or canonical_hash(event) != batch.get("cache_event_hash")
            or canonical_hash(instance) != batch.get("instance_hash")
            or instance.get("template_hash") != template_hash
            or instance.get("cache_key") != cache_key
            or batch.get("template_hash") != template_hash
            or batch.get("task_hash") != template.get("task_hash")
            or batch.get("schedule_hash") != template.get("schedule_hash")
            or event.get("outcome")
            != ("miss_compiled" if index == 0 else "hit_exact_contract")
        ):
            raise ValueError("NRIR-28 cache event/instance linkage differs")


def _validate_worker_result(result: Mapping[str, Any]) -> None:
    clauses = _list(result.get("clauses"), "NRIR-28 clauses")
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("mode") not in MODES
        or result.get("solver_status") not in {"verified", "unsafe", "unknown"}
        or result.get("performance_claimed") is not False
        or not clauses
        or any(
            not isinstance(result.get(name), int) or int(result[name]) < 0
            for name in ("setup_ns", "execution_ns", "worker_elapsed_ns")
        )
    ):
        raise ValueError("NRIR-28 worker result differs")
    if result["mode"] == "production_v1":
        if result.get("compiler") is not None:
            raise ValueError("NRIR-28 baseline unexpectedly has compiler evidence")
    else:
        _validate_compiler_payload(_mapping(result.get("compiler"), "NRIR-28 compiler"))


def _semantic_projection(result: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: result[key]
        for key in (
            "solver_status",
            "completed_clause_count",
            "unresolved_clause_indices",
            "pending_clause_indices",
            "semantic_signature_hash",
        )
    }


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("property_status") != "validated_reduced"
        or evidence.get("performance_claimed") is not True
    ):
        raise ValueError("NRIR-28 evidence header differs")
    records = _list(evidence.get("records"), "NRIR-28 records")
    if len(records) != len(WORKLOAD_ROWS) * REPEATS * len(MODES):
        raise ValueError("NRIR-28 record count differs")
    normalized: list[dict[str, Any]] = []
    for value in records:
        record = dict(_mapping(value, "NRIR-28 record"))
        _validate_worker_result(_mapping(record.get("result"), "NRIR-28 result"))
        if (
            record.get("mode") not in MODES
            or record.get("group_index") not in range(REPEATS)
            or record.get("order_index") not in range(len(MODES))
            or not isinstance(record.get("e2e_elapsed_ns"), int)
            or int(record["e2e_elapsed_ns"]) <= 0
            or not _sha256(record.get("log_sha256"))
        ):
            raise ValueError("NRIR-28 record identity differs")
        normalized.append(record)
    for workload in sorted({str(record["workload_id"]) for record in normalized}):
        for group_index in range(REPEATS):
            pair = {
                str(record["mode"]): _mapping(record["result"], "NRIR-28 pair")
                for record in normalized
                if record["workload_id"] == workload
                and record["group_index"] == group_index
            }
            if set(pair) != set(MODES):
                raise ValueError("NRIR-28 group mode coverage differs")
            if _semantic_projection(pair["production_v1"]) != _semantic_projection(
                pair["parametric_v2"]
            ):
                raise ValueError("NRIR-28 query semantic parity differs")
            baseline_clauses = _list(
                pair["production_v1"]["clauses"], "baseline clauses"
            )
            parametric_clauses = _list(
                pair["parametric_v2"]["clauses"], "parametric clauses"
            )
            if len(baseline_clauses) != len(parametric_clauses):
                raise ValueError("NRIR-28 clause coverage differs")
            for baseline, parametric in zip(baseline_clauses, parametric_clauses):
                left = _mapping(baseline, "baseline clause")
                right = _mapping(parametric, "parametric clause")
                for key in (
                    "clause_index",
                    "status",
                    "queue_status",
                    "evaluated_nodes",
                    "logical_queue_signature_hash",
                    "selected_state_hashes",
                ):
                    if left[key] != right[key]:
                        raise ValueError("NRIR-28 clause semantic parity differs")
                if any(
                    abs(float(left[key]) - float(right[key])) > 2e-3
                    for key in ("root_lower", "root_upper")
                ):
                    raise ValueError("NRIR-28 clause numeric parity differs")
    summaries = _mapping(evidence.get("summaries"), "NRIR-28 summaries")
    if dict(summaries) != _summaries(normalized):
        raise ValueError("NRIR-28 summary replay differs")
    if any(
        float(_mapping(value, "NRIR-28 summary")["internal_speedup"]) <= 1.0
        for value in summaries.values()
    ):
        raise ValueError("NRIR-28 full-query performance gate failed")


def _public_workload(workload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in workload.items()
        if key not in {"model", "property"}
    }


def _build_evidence(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, str]]:
    workloads = _resolve_workloads(args.benchmark_root)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.artifact_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir28-") as temporary:
        temporary_root = Path(temporary)
        for workload in workloads:
            workload_id = str(workload["workload_id"])
            for group_index, order in enumerate(MODE_ORDERS):
                for order_index, mode in enumerate(order):
                    result_path = temporary_root / (
                        f"{workload_id.replace(':', '-')}-{group_index}-{mode}.json"
                    )
                    result, log, elapsed_ns = _run_worker(
                        mode=mode,
                        workload=workload,
                        result_path=result_path,
                        torch_threads=args.torch_threads,
                    )
                    _validate_worker_result(result)
                    log_name = (
                        f"logs/{workload_id.replace(':', '-')}-"
                        f"g{group_index}-{mode}.log"
                    )
                    log_path = args.artifact_dir / log_name
                    log_path.write_text(log, encoding="utf-8")
                    files[log_name] = file_sha256(log_path)
                    records.append(
                        {
                            "workload_id": workload_id,
                            "group_index": group_index,
                            "order_index": order_index,
                            "mode": mode,
                            "e2e_elapsed_ns": elapsed_ns,
                            "log_path": log_name,
                            "log_sha256": files[log_name],
                            "result": result,
                        }
                    )
                    print(
                        _canonical_json(
                            {
                                "workload_id": workload_id,
                                "group_index": group_index,
                                "mode": mode,
                                "e2e_elapsed_ns": elapsed_ns,
                                "solver_status": result["solver_status"],
                            }
                        ),
                        flush=True,
                    )
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "property_status": "validated_reduced",
        "performance_claimed": True,
        "claim_boundary": "three real VNN-COMP CPU workloads; repeated same-algorithm production-v1 to parametric-v2 full-query internal speedup only; no competitor speedup, no GPU, no ASPLOS-ready claim",
        "source": {
            "vnncomp_commit": VNNCOMP_COMMIT,
            "native_code_revision": _code_revision(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": args.torch_threads,
            "cuda_executed": False,
        },
        "protocol": {
            "repeats": REPEATS,
            "mode_orders": [list(order) for order in MODE_ORDERS],
            "timing_boundary": "fresh_process_start_to_structured_result",
            "timeout_seconds": TIMEOUT_SECONDS,
            "search_steps": SEARCH_STEPS,
            "alpha_steps": ALPHA_STEPS,
            "max_nodes": MAX_NODES,
            "comparison": "identical_full_query_production_v1_vs_parametric_v2",
        },
        "workloads": [_public_workload(workload) for workload in workloads],
        "records": records,
        "summaries": _summaries(records),
        "limitations": [
            "The performance claim is an internal production-v1 versus parametric-v2 full-query CPU comparison with identical search and optimizer budgets.",
            "The generating host has no visible CUDA driver; no GPU, allocator, OOM, or Pareto claim is made.",
            "BoundFlow remains bounded to 7 nodes/depth 2 and may return unknown.",
            "No alpha-beta-CROWN competitor speedup or complete-property closure is claimed.",
            "The cache is query-local, single-threaded, and in-memory; cross-process persistence is not claimed.",
        ],
    }
    validate_evidence_structure(evidence)
    return evidence, files


def _generate(args: argparse.Namespace) -> None:
    evidence, files = _build_evidence(args)
    evidence_path = args.artifact_dir / EVIDENCE_FILE
    _write_json(evidence_path, evidence)
    files[EVIDENCE_FILE] = file_sha256(evidence_path)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": True,
        "files": dict(sorted(files.items())),
        "evidence_hash": canonical_hash(evidence),
    }
    _write_json(args.artifact_dir / MANIFEST_FILE, manifest)
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def _replay(args: argparse.Namespace) -> None:
    manifest = _load_json(args.artifact_dir / MANIFEST_FILE)
    evidence = _load_json(args.artifact_dir / EVIDENCE_FILE)
    files = _mapping(manifest.get("files"), "NRIR-28 manifest files")
    actual_files = {
        str(path.relative_to(args.artifact_dir)): file_sha256(path)
        for path in sorted(args.artifact_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not True
        or dict(files) != actual_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-28 artifact manifest differs")
    source = _mapping(evidence.get("source"), "NRIR-28 source")
    if (
        source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("native_code_revision") != _code_revision()
    ):
        raise ValueError("NRIR-28 source revision differs")
    expected_workloads = [
        _public_workload(workload)
        for workload in _resolve_workloads(args.benchmark_root)
    ]
    if evidence.get("workloads") != expected_workloads:
        raise ValueError("NRIR-28 source workload replay differs")
    validate_evidence_structure(evidence)
    for record in _list(evidence["records"], "NRIR-28 records"):
        item = _mapping(record, "NRIR-28 record")
        if file_sha256(args.artifact_dir / str(item["log_path"])) != item["log_sha256"]:
            raise ValueError("NRIR-28 log digest differs")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-28 torch thread count must be positive")
    if args.command == "worker":
        _worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
