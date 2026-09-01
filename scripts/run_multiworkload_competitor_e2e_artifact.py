#!/usr/bin/env python3
"""Generate or replay the NRIR-18 multi-workload competitor E2E artifact."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-arguments,duplicate-code
# pylint: disable=too-many-boolean-expressions,import-outside-toplevel,import-error
# pylint: disable=missing-function-docstring,line-too-long

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

ARTIFACT_SCHEMA_VERSION = "boundflow.multiworkload-competitor-e2e-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.multiworkload-competitor-e2e-evidence/v1"
WORKER_RESULT_SCHEMA_VERSION = "boundflow.multiworkload-worker-result/v1"
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
PLAN_ID = "vnncomp21-three-topology-cpu-v1"
NATIVE_TIMEOUT_SECONDS = 60
ABCROWN_TIMEOUT_SECONDS = 60
NATIVE_ALPHA_STEPS = 5
ABCROWN_ALPHA_STEPS = 25
ABCROWN_BETA_STEPS = 10
SEARCH_STEPS = 4
MAX_NODES = 7
WORKLOAD_ROWS = (
    {
        "workload_id": "mnistfc:000",
        "category": "mnistfc",
        "csv_ordinal": 0,
        "csv": "benchmarks/mnistfc/mnistfc_instances.csv",
        "model": "benchmarks/mnistfc/mnist-net_256x2.onnx",
        "property": "benchmarks/mnistfc/prop_0_0.03.vnnlib",
    },
    {
        "workload_id": "cifar10_resnet:000",
        "category": "cifar10_resnet",
        "csv_ordinal": 0,
        "csv": "benchmarks/cifar10_resnet/cifar10_resnet_instances.csv",
        "model": "benchmarks/cifar10_resnet/onnx/resnet_2b.onnx",
        "property": "benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib",
    },
    {
        "workload_id": "oval21:000",
        "category": "oval21",
        "csv_ordinal": 0,
        "csv": "benchmarks/oval21/oval21_instances.csv",
        "model": "benchmarks/oval21/nets/cifar_base_kw.onnx",
        "property": "benchmarks/oval21/vnnlib/cifar_base_kw-img4549-eps0.00392156862745098.vnnlib",
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--abcrown-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
        subparser.add_argument("--torch-threads", type=int, default=8)
    for command in ("worker-native", "worker-abcrown"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--workload-id", required=True)
        subparser.add_argument("--model", type=Path, required=True)
        subparser.add_argument("--property", type=Path, required=True)
        subparser.add_argument("--result-json", type=Path, required=True)
        subparser.add_argument("--torch-threads", type=int, required=True)
        subparser.add_argument("--timeout-seconds", type=int, required=True)
        subparser.add_argument("--alpha-steps", type=int, required=True)
        subparser.add_argument("--beta-steps", type=int, required=True)
        subparser.add_argument("--search-steps", type=int, required=True)
        subparser.add_argument("--max-nodes", type=int, required=True)
        subparser.add_argument("--abcrown-root", type=Path)
        subparser.add_argument(
            "--input-shape",
            type=int,
            nargs="+",
            help="materialize a symbolic ONNX input shape for a frozen worker run",
        )
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _native_code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/query.py",
        "boundflow/ir/workload.py",
        "boundflow/frontends/vnnlib.py",
        "boundflow/runtime/crown_ibp.py",
        "boundflow/runtime/complete_verifier_query.py",
        "boundflow/runtime/native_optimized_relu_split_bab_runtime.py",
        "scripts/run_multiworkload_competitor_e2e_artifact.py",
    )
    return canonical_hash({path: file_sha256(root / path) for path in paths})


def _onnx_inventory(
    model_path: Path, input_shape_override: tuple[int, ...] | None = None
) -> tuple[tuple[int, ...], int, tuple[str, ...]]:
    import onnx  # pylint: disable=import-outside-toplevel

    model = onnx.load(str(model_path), load_external_data=False)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("NRIR-18 requires one ONNX input and one output")

    def raw_shape(value: Any) -> tuple[int, ...]:
        return tuple(
            int(dimension.dim_value) for dimension in value.type.tensor_type.shape.dim
        )

    input_dimensions = raw_shape(model.graph.input[0])
    if input_shape_override is None:
        input_shape = input_dimensions
        if not input_shape or any(dimension < 1 for dimension in input_shape):
            raise ValueError("NRIR-18 requires static positive ONNX shapes")
    else:
        input_shape = tuple(input_shape_override)
        if (
            len(input_shape) != len(input_dimensions)
            or any(dimension < 1 for dimension in input_shape)
            or any(
                observed > 0 and observed != override
                for observed, override in zip(input_dimensions, input_shape)
            )
        ):
            raise ValueError("NRIR-18 input-shape override contradicts ONNX input")
    output_shape = raw_shape(model.graph.output[0])
    if not output_shape or any(dimension < 1 for dimension in output_shape[1:]):
        raise ValueError("NRIR-18 requires a static positive ONNX output dimension")
    if output_shape[0] < 1 and input_shape_override is not None:
        output_shape = (input_shape[0], *output_shape[1:])
    if input_shape[0] != 1 or len(output_shape) != 2 or output_shape[0] != 1:
        raise ValueError("NRIR-18 requires batch-one model IO")
    return (
        input_shape,
        output_shape[1],
        tuple(sorted({node.op_type for node in model.graph.node})),
    )


def _csv_selection(
    benchmark_root: Path, definition: Mapping[str, object]
) -> tuple[Path, Path, Path]:
    csv_path = benchmark_root / str(definition["csv"])
    model_path = benchmark_root / str(definition["model"])
    property_path = benchmark_root / str(definition["property"])
    ordinal_value = definition["csv_ordinal"]
    if not isinstance(ordinal_value, int):
        raise TypeError("NRIR-18 CSV ordinal must be an integer")
    ordinal = ordinal_value
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if ordinal >= len(rows) or len(rows[ordinal]) < 2:
        raise ValueError("NRIR-18 CSV selection is missing")
    category_root = csv_path.parent
    selected_model = (category_root / rows[ordinal][0]).resolve()
    selected_property = (category_root / rows[ordinal][1]).resolve()
    if (
        selected_model != model_path.resolve()
        or selected_property != property_path.resolve()
    ):
        raise ValueError("NRIR-18 selected CSV row differs")
    if not model_path.is_file() or not property_path.is_file():
        raise FileNotFoundError("NRIR-18 selected source file is missing")
    return csv_path, model_path, property_path


def _build_ir(
    benchmark_root: Path, abcrown_root: Path, torch_threads: int
) -> tuple[Any, Any, Any, list[dict[str, object]]]:
    from boundflow.frontends.vnnlib import (  # pylint: disable=import-outside-toplevel
        import_vnnlib_box_query,
    )
    from boundflow.ir.workload import (  # pylint: disable=import-outside-toplevel
        MultiWorkloadPlanIR,
        VerificationWorkloadSourceIR,
        VerifierBackendKind,
        VerifierExecutionPolicyIR,
        compile_multiworkload_schedule_ir,
        compile_multiworkload_task_ir,
    )

    if _git_revision(benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("NRIR-18 VNN-COMP commit differs")
    if _git_revision(abcrown_root) != ABCROWN_COMMIT:
        raise ValueError("NRIR-18 alpha-beta-CROWN commit differs")
    sources = []
    resolved: list[dict[str, object]] = []
    for definition in WORKLOAD_ROWS:
        csv_path, model_path, property_path = _csv_selection(benchmark_root, definition)
        workload_id = str(definition["workload_id"])
        query = import_vnnlib_box_query(property_path, query_id=workload_id)
        input_shape, output_dim, ops = _onnx_inventory(model_path)
        if (
            len(query.input_names) != math.prod(input_shape[1:])
            or len(query.output_names) != output_dim
        ):
            raise ValueError("NRIR-18 ONNX/VNNLIB dimensions differ")
        ordinal_value = definition["csv_ordinal"]
        if not isinstance(ordinal_value, int):
            raise TypeError("NRIR-18 workload ordinal must be an integer")
        source = VerificationWorkloadSourceIR(
            workload_id=workload_id,
            category=str(definition["category"]),
            csv_ordinal=ordinal_value,
            csv_relative_path=str(definition["csv"]),
            model_relative_path=str(definition["model"]),
            property_relative_path=str(definition["property"]),
            csv_sha256=file_sha256(csv_path),
            model_sha256=file_sha256(model_path),
            property_sha256=file_sha256(property_path),
            query_ir_hash=query.stable_hash(),
            model_input_shape=input_shape,
            model_output_dim=output_dim,
            onnx_ops=ops,
        )
        source.validate()
        sources.append(source)
        resolved.append(
            {
                "workload_id": workload_id,
                "model": model_path,
                "property": property_path,
                "query": query,
            }
        )
    native = VerifierExecutionPolicyIR(
        backend=VerifierBackendKind.BOUNDFLOW_NATIVE,
        implementation_id="boundflow-native-complete-query-v1",
        implementation_revision=_native_code_revision(),
        device="cpu",
        torch_threads=torch_threads,
        timeout_seconds=NATIVE_TIMEOUT_SECONDS,
        alpha_steps=NATIVE_ALPHA_STEPS,
        beta_steps=NATIVE_ALPHA_STEPS,
        search_steps=SEARCH_STEPS,
        max_nodes=MAX_NODES,
        attack_policy="native_projected_gradient",
        complete_verifier="bounded_relu_bab",
    )
    competitor = VerifierExecutionPolicyIR(
        backend=VerifierBackendKind.EXTERNAL_ABCROWN,
        implementation_id="alpha-beta-CROWN",
        implementation_revision=ABCROWN_COMMIT,
        device="cpu",
        torch_threads=torch_threads,
        timeout_seconds=ABCROWN_TIMEOUT_SECONDS,
        alpha_steps=ABCROWN_ALPHA_STEPS,
        beta_steps=ABCROWN_BETA_STEPS,
        search_steps=0,
        max_nodes=0,
        attack_policy="skip",
        complete_verifier="bab",
    )
    plan = MultiWorkloadPlanIR(
        plan_id=PLAN_ID,
        benchmark_commit=VNNCOMP_COMMIT,
        workloads=tuple(sources),
        policies=(native, competitor),
    )
    task_ir = compile_multiworkload_task_ir(plan)
    schedule = compile_multiworkload_schedule_ir(plan, task_ir)
    return plan, task_ir, schedule, resolved


def _worker_command(
    *,
    backend: str,
    workload: Mapping[str, object],
    result_path: Path,
    abcrown_root: Path,
    torch_threads: int,
) -> list[str]:
    if backend == "boundflow_native":
        command = "worker-native"
        timeout_seconds = NATIVE_TIMEOUT_SECONDS
        alpha_steps = NATIVE_ALPHA_STEPS
        beta_steps = NATIVE_ALPHA_STEPS
        search_steps = SEARCH_STEPS
        max_nodes = MAX_NODES
    else:
        command = "worker-abcrown"
        timeout_seconds = ABCROWN_TIMEOUT_SECONDS
        alpha_steps = ABCROWN_ALPHA_STEPS
        beta_steps = ABCROWN_BETA_STEPS
        search_steps = 0
        max_nodes = 0
    result = [
        sys.executable,
        str(Path(__file__).resolve()),
        command,
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
        "--timeout-seconds",
        str(timeout_seconds),
        "--alpha-steps",
        str(alpha_steps),
        "--beta-steps",
        str(beta_steps),
        "--search-steps",
        str(search_steps),
        "--max-nodes",
        str(max_nodes),
    ]
    if backend == "external_abcrown":
        result.extend(("--abcrown-root", str(abcrown_root)))
    return result


def _timeout_result(workload_id: str, backend: str) -> dict[str, object]:
    return {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "workload_id": workload_id,
        "backend": backend,
        "execution_state": "timed_out",
        "solver_status": "timeout",
        "success": False,
        "performance_claimed": False,
    }


def _run_worker(
    *,
    backend: str,
    workload: Mapping[str, object],
    result_path: Path,
    abcrown_root: Path,
    torch_threads: int,
) -> tuple[dict[str, object], str, int, int]:
    command = _worker_command(
        backend=backend,
        workload=workload,
        result_path=result_path,
        abcrown_root=abcrown_root,
        torch_threads=torch_threads,
    )
    started_ns = time.perf_counter_ns()
    timeout_seconds = (
        NATIVE_TIMEOUT_SECONDS
        if backend == "boundflow_native"
        else ABCROWN_TIMEOUT_SECONDS
    )
    try:
        completed = subprocess.run(
            command,
            cwd=_repo_root(),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds + 30,
            check=False,
        )
        elapsed_ns = time.perf_counter_ns() - started_ns
        log = completed.stdout
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                f"NRIR-18 {backend} worker failed with {completed.returncode}: {log[-4000:]}"
            )
        result = _load_json(result_path)
        return result, log, completed.returncode, elapsed_ns
    except subprocess.TimeoutExpired as error:
        elapsed_ns = time.perf_counter_ns() - started_ns
        stdout = error.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        result = _timeout_result(str(workload["workload_id"]), backend)
        return result, stdout + "\nNRIR-18 parent timeout\n", 124, elapsed_ns


def _validate_worker_result(result: Mapping[str, object]) -> None:
    if (
        result.get("schema_version") != WORKER_RESULT_SCHEMA_VERSION
        or not result.get("workload_id")
        or result.get("backend") not in {"boundflow_native", "external_abcrown"}
        or result.get("execution_state") not in {"completed", "timed_out"}
        or not result.get("solver_status")
        or not isinstance(result.get("success"), bool)
        or result.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-18 worker result header differs")
    if result.get("execution_state") == "timed_out":
        if (
            result.get("solver_status") != "timeout"
            or result.get("success") is not False
        ):
            raise ValueError("NRIR-18 timeout result differs")
        return
    worker_elapsed_ns = result.get("worker_elapsed_ns")
    if not isinstance(worker_elapsed_ns, int) or worker_elapsed_ns <= 0:
        raise ValueError("NRIR-18 completed worker timing differs")
    if result.get("backend") == "boundflow_native":
        if (
            result.get("solver_status") not in {"verified", "unsafe", "unknown"}
            or not _sha256(result.get("query_ir_hash"))
            or not _sha256(result.get("query_trace_hash"))
            or not isinstance(result.get("clauses"), list)
        ):
            raise ValueError("NRIR-18 native result differs")
    elif result.get("solver_status") not in {
        "verified",
        "unsafe",
        "unknown",
        "timeout",
    }:
        raise ValueError("NRIR-18 alpha-beta-CROWN result differs")


def _native_worker(args: argparse.Namespace) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    from boundflow.frontends.onnx.frontend import (  # pylint: disable=import-outside-toplevel
        import_onnx,
    )
    from boundflow.frontends.vnnlib import (  # pylint: disable=import-outside-toplevel
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
    from boundflow.planner import (
        plan_interval_ibp_v0,
    )  # pylint: disable=import-outside-toplevel
    from boundflow.runtime.complete_verifier_query import (  # pylint: disable=import-outside-toplevel
        CompleteVerifierQueryPolicy,
        execute_complete_verifier_query,
    )
    from boundflow.runtime.native_alpha_beta_optimization_state import (  # pylint: disable=import-outside-toplevel
        NativeAlphaBetaOptimizerPolicy,
    )
    from boundflow.runtime.native_candidate_search import (  # pylint: disable=import-outside-toplevel
        NativeProjectedGradientSearchPolicy,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import (  # pylint: disable=import-outside-toplevel
        NativeReluSplitBabConfig,
    )
    from boundflow.runtime.task_executor import (
        InputSpec,
    )  # pylint: disable=import-outside-toplevel

    torch.set_num_threads(args.torch_threads)
    started_ns = time.perf_counter_ns()
    query = import_vnnlib_box_query(args.property, query_id=args.workload_id)
    input_shape_override = (
        tuple(args.input_shape) if args.input_shape is not None else None
    )
    input_shape, output_dim, _ops = _onnx_inventory(
        args.model, input_shape_override=input_shape_override
    )
    if len(query.output_names) != output_dim:
        raise ValueError("NRIR-18 native worker output dimension differs")
    tensors = materialize_vnnlib_box_query(query, input_shape=input_shape[1:])
    program = import_onnx(
        str(args.model),
        do_shape_infer=True,
        input_shapes=[list(input_shape)] if input_shape_override is not None else None,
        normalize=True,
    )
    module = plan_interval_ibp_v0(program)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    setup_ns = time.perf_counter_ns() - started_ns
    execute_started_ns = time.perf_counter_ns()
    execution = execute_complete_verifier_query(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id=query.query_id,
        query_policy=CompleteVerifierQueryPolicy(
            timeout_ns=args.timeout_seconds * 1_000_000_000
        ),
        search_policy=NativeProjectedGradientSearchPolicy(
            steps=args.search_steps, step_size=0.002
        ),
        queue_config=NativeReluSplitBabConfig(
            max_nodes=args.max_nodes,
            max_depth=2,
            expansion_batch_size=2,
            max_eval_batch_size=4,
        ),
        optimizer_policy=NativeAlphaBetaOptimizerPolicy(
            steps=args.alpha_steps,
            lr=0.1,
            alpha_initialization_mode="adaptive",
        ),
    )
    execution_ns = time.perf_counter_ns() - execute_started_ns
    clauses = []
    for clause in execution.clauses:
        queue = clause.queue.trace
        root = queue.evaluations[0]
        clauses.append(
            {
                "clause_index": clause.trace.clause_index,
                "status": clause.trace.status,
                "root_lower": root.lower,
                "root_upper": root.upper,
                "queue_status": queue.status,
                "evaluated_nodes": len(queue.evaluations),
                "final_frontier_nodes": len(queue.final_frontier_node_ids),
                "query_clause_trace_hash": canonical_hash(clause.trace.to_dict()),
            }
        )
    result = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "workload_id": args.workload_id,
        "backend": "boundflow_native",
        "execution_state": "completed",
        "solver_status": execution.trace.status,
        "success": execution.trace.status in {"verified", "unsafe"},
        "query_ir_hash": query.stable_hash(),
        "query_trace_hash": execution.trace.stable_hash(),
        "setup_ns": setup_ns,
        "execution_ns": execution_ns,
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "completed_clause_count": len(execution.clauses),
        "unresolved_clause_indices": list(execution.trace.unresolved_clause_indices),
        "pending_clause_indices": list(execution.trace.pending_clause_indices),
        "clauses": clauses,
        "primal_ops": [op.op_type for op in module.get_entry_task().ops],
        "performance_claimed": False,
    }
    _write_json(args.result_json, result)
    print(_canonical_json({"status": "ok", "solver_status": execution.trace.status}))


def _abcrown_worker(args: argparse.Namespace) -> None:
    if args.abcrown_root is None:
        raise ValueError("NRIR-18 alpha-beta-CROWN root is required")
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch  # pylint: disable=import-outside-toplevel

    from abcrown import (  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
        ABCrownSolver,
        ConfigBuilder,
        IOConstraints,
    )

    torch.set_num_threads(args.torch_threads)
    started_ns = time.perf_counter_ns()
    config = (
        ConfigBuilder.from_defaults()
        .set("general/device", "cpu")
        .set("general/complete_verifier", "bab")
        .set("attack/pgd_order", "skip")
        .set("bab/timeout", args.timeout_seconds)
        .set("solver/batch_size", 64)
        .set("solver/alpha-crown/iteration", args.alpha_steps)
        .set("solver/beta-crown/iteration", args.beta_steps)
    )
    solver = ABCrownSolver(str(args.model), config=config)
    result = solver.verify(constraints=IOConstraints(vnnlib_path=str(args.property)))
    solver_status = str(result.status)
    record = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "workload_id": args.workload_id,
        "backend": "external_abcrown",
        "execution_state": "completed",
        "solver_status": solver_status,
        "success": bool(result.success),
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "abcrown_commit": _git_revision(args.abcrown_root),
        "performance_claimed": False,
    }
    _write_json(args.result_json, record)
    print(_canonical_json({"status": "ok", "solver_status": solver_status}))


def _build_evidence(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, str]]:
    plan, task_ir, schedule, workloads = _build_ir(
        args.benchmark_root, args.abcrown_root, args.torch_threads
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.artifact_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir18-") as temporary:
        temporary_root = Path(temporary)
        for workload in workloads:
            for backend in ("boundflow_native", "external_abcrown"):
                result_path = (
                    temporary_root
                    / f"{str(workload['workload_id']).replace(':', '-')}-{backend}.json"
                )
                result, log, returncode, e2e_elapsed_ns = _run_worker(
                    backend=backend,
                    workload=workload,
                    result_path=result_path,
                    abcrown_root=args.abcrown_root,
                    torch_threads=args.torch_threads,
                )
                _validate_worker_result(result)
                log_name = f"logs/{str(workload['workload_id']).replace(':', '-')}-{backend}.log"
                log_path = args.artifact_dir / log_name
                log_path.write_text(log, encoding="utf-8")
                files[log_name] = file_sha256(log_path)
                records.append(
                    {
                        "workload_id": workload["workload_id"],
                        "backend": backend,
                        "process_returncode": returncode,
                        "e2e_elapsed_ns": e2e_elapsed_ns,
                        "log_path": log_name,
                        "log_sha256": files[log_name],
                        "result": result,
                    }
                )
                print(
                    _canonical_json(
                        {
                            "workload_id": workload["workload_id"],
                            "backend": backend,
                            "solver_status": result["solver_status"],
                            "e2e_elapsed_ns": e2e_elapsed_ns,
                        }
                    ),
                    flush=True,
                )
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "property_status": "validated_reduced",
        "performance_claimed": False,
        "claim_boundary": "three real VNN-COMP CPU workloads; executable correctness/coverage and diagnostic E2E only; no speedup, GPU, Pareto, or ASPLOS-ready claim",
        "source": {
            "vnncomp_commit": VNNCOMP_COMMIT,
            "abcrown_commit": ABCROWN_COMMIT,
            "native_code_revision": _native_code_revision(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": args.torch_threads,
            "cuda_executed": False,
        },
        "ir": {
            "plan": plan.to_dict(),
            "plan_hash": plan.stable_hash(),
            "task": task_ir.to_dict(),
            "task_hash": task_ir.stable_hash(),
            "schedule": schedule.to_dict(),
            "schedule_hash": schedule.stable_hash(task_ir),
        },
        "records": records,
        "limitations": [
            "BoundFlow native execution uses bounded 7-node/depth-2 search and may return unknown.",
            "alpha-beta-CROWN uses its own complete BaB policy; different algorithms are not a speed comparison.",
            "CPU diagnostic timings are single fresh-process observations, not performance claims.",
            "CUDA driver/device is unavailable on the generating host; the GPU matrix remains pending.",
            "The VNNLIB v1 frontend accepts box inputs and one linear inequality per unsafe DNF disjunct only.",
        ],
    }
    return evidence, files


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("property_status") != "validated_reduced"
        or evidence.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-18 evidence header differs")
    claim = str(evidence.get("claim_boundary", ""))
    for phrase in ("three real VNN-COMP CPU", "no speedup", "GPU", "ASPLOS-ready"):
        if phrase not in claim:
            raise ValueError("NRIR-18 claim boundary differs")
    source = _mapping(evidence.get("source"), "NRIR-18 source")
    if (
        source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("abcrown_commit") != ABCROWN_COMMIT
        or not _sha256(source.get("native_code_revision"))
    ):
        raise ValueError("NRIR-18 source identity differs")
    environment = _mapping(evidence.get("environment"), "NRIR-18 environment")
    if (
        environment.get("device") != "cpu"
        or environment.get("cuda_executed") is not False
        or not isinstance(environment.get("torch_threads"), int)
        or int(environment["torch_threads"]) < 1
    ):
        raise ValueError("NRIR-18 environment boundary differs")
    ir = _mapping(evidence.get("ir"), "NRIR-18 IR")
    plan = _mapping(ir.get("plan"), "NRIR-18 Plan IR")
    task = _mapping(ir.get("task"), "NRIR-18 Task IR")
    schedule = _mapping(ir.get("schedule"), "NRIR-18 Schedule IR")
    if (
        ir.get("plan_hash") != canonical_hash(plan)
        or ir.get("task_hash") != canonical_hash(task)
        or ir.get("schedule_hash") != canonical_hash(schedule)
        or task.get("plan_hash") != ir.get("plan_hash")
        or schedule.get("plan_hash") != ir.get("plan_hash")
        or schedule.get("task_ir_hash") != ir.get("task_hash")
        or plan.get("performance_claimed") is not False
        or plan.get("claim_boundary") != "cpu_diagnostic_no_speedup"
        or len(_list(plan.get("workloads"), "NRIR-18 workloads")) != 3
        or len(_list(task.get("tasks"), "NRIR-18 tasks")) != 21
        or len(_list(schedule.get("fresh_process_task_ids"), "NRIR-18 fresh workers"))
        != 6
    ):
        raise ValueError("NRIR-18 Plan/Task/Schedule linkage differs")
    records = _list(evidence.get("records"), "NRIR-18 records")
    identities: set[tuple[str, str]] = set()
    for record in records:
        item = _mapping(record, "NRIR-18 record")
        identity = (str(item.get("workload_id")), str(item.get("backend")))
        result = _mapping(item.get("result"), "NRIR-18 worker result")
        _validate_worker_result(result)
        if (
            identity in identities
            or result.get("workload_id") != identity[0]
            or result.get("backend") != identity[1]
            or not isinstance(item.get("process_returncode"), int)
            or not isinstance(item.get("e2e_elapsed_ns"), int)
            or int(item["e2e_elapsed_ns"]) <= 0
            or not str(item.get("log_path", "")).startswith("logs/")
            or not _sha256(item.get("log_sha256"))
        ):
            raise ValueError("NRIR-18 execution record differs")
        identities.add(identity)
    expected = {
        (str(workload["workload_id"]), backend)
        for workload in WORKLOAD_ROWS
        for backend in ("boundflow_native", "external_abcrown")
    }
    if identities != expected:
        raise ValueError("NRIR-18 workload/backend coverage differs")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 5:
        raise ValueError("NRIR-18 limitation ledger differs")


def _generate(args: argparse.Namespace) -> None:
    evidence, files = _build_evidence(args)
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


def _replay(args: argparse.Namespace) -> None:
    manifest = _load_json(args.artifact_dir / MANIFEST_FILE)
    evidence = _load_json(args.artifact_dir / EVIDENCE_FILE)
    files = _mapping(manifest.get("files"), "NRIR-18 manifest files")
    actual_files = {
        str(path.relative_to(args.artifact_dir)): file_sha256(path)
        for path in sorted(args.artifact_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or dict(files) != actual_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-18 artifact manifest differs")
    validate_evidence_structure(evidence)
    torch_threads = int(
        _mapping(evidence["environment"], "environment")["torch_threads"]
    )
    plan, task_ir, schedule, _workloads = _build_ir(
        args.benchmark_root, args.abcrown_root, torch_threads
    )
    stored_ir = _mapping(evidence["ir"], "NRIR-18 stored IR")
    expected_ir = {
        "plan": plan.to_dict(),
        "plan_hash": plan.stable_hash(),
        "task": task_ir.to_dict(),
        "task_hash": task_ir.stable_hash(),
        "schedule": schedule.to_dict(),
        "schedule_hash": schedule.stable_hash(task_ir),
    }
    if dict(stored_ir) != expected_ir:
        raise ValueError("NRIR-18 source-to-IR replay differs")
    for record in _list(evidence["records"], "NRIR-18 replay records"):
        item = _mapping(record, "NRIR-18 replay record")
        log_path = args.artifact_dir / str(item["log_path"])
        if file_sha256(log_path) != item["log_sha256"]:
            raise ValueError("NRIR-18 execution log digest differs")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-18 positive resource values are required")
    if hasattr(args, "timeout_seconds") and args.timeout_seconds < 1:
        raise ValueError("NRIR-18 positive resource values are required")
    if args.command == "worker-native":
        _native_worker(args)
    elif args.command == "worker-abcrown":
        _abcrown_worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
