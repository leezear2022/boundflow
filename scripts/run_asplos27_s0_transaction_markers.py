#!/usr/bin/env python3
"""Run or replay explicit αβ-CROWN solver transaction marker evidence."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-arguments,import-outside-toplevel
# pylint: disable=protected-access,missing-function-docstring,line-too-long
# pylint: disable=wrong-import-position,import-error,consider-using-with
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, cast, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.gpu_attribution import canonical_hash  # noqa: E402
from boundflow.runtime.solver_transaction_observer import (  # noqa: E402
    SolverTransactionObserver,
    TransactionCategory,
    TransactionResolution,
    TransactionTarget,
    host_transaction_span_from_dict,
    summarize_solver_transactions,
)
from scripts import run_fsg1_official_control_baseline as fsg1  # noqa: E402

WORKER_SCHEMA_VERSION = "boundflow.asplos27-s0-transaction-worker/v1"
ARTIFACT_SCHEMA_VERSION = "boundflow.asplos27-s0-transaction-artifact/v1"
DEFAULT_ARTIFACT = Path("artifacts/asplos27-s0-transactions/official-b0-five-pair-v1")
TARGET_BLOBS = {
    "complete_verifier/api.py": "8634ced960d73aa31593c2c3c4df19e9fcca2677",
    "complete_verifier/abcrown.py": "5bb2392f84eecc9f442035376d18117398358dc3",
    "complete_verifier/complete_verifier_func.py": (
        "18c02c98173bc72fe166f5e0723da47442e0952e"
    ),
    "complete_verifier/activation_split/bab_bootstrap.py": (
        "29f5153bf4b7adc1524ccb795255250b5029fd2b"
    ),
    "complete_verifier/activation_split/stage_preprocess.py": (
        "6ec5c1a794e57e3994a2d616444480737e97e9d1"
    ),
    "complete_verifier/activation_split/stage_solve.py": (
        "75f2d2b7847772d38797066740a57b69190d214f"
    ),
    "complete_verifier/activation_split/stage_postprocess.py": (
        "b5585bb060151c43e48999bdd30d785b299fae18"
    ),
    "complete_verifier/activation_split/update_bounds_phases.py": (
        "60fe57bd78a155cd5bfda68a0161cdfa37519b5a"
    ),
    "complete_verifier/branching_domains.py": (
        "7ef216f6f26a035316447a69da181de4625751a4"
    ),
    "complete_verifier/activation_split/decision_precompute.py": (
        "52bd81c76b14d36119650f37566eb4f1aeb80a8a"
    ),
    "complete_verifier/heuristics/__init__.py": (
        "7035fbc811131342233b0685d46edbf219caa720"
    ),
}
TARGET_SPECS = (
    ("api.ABCrownSolver.__init__", "frontend_setup", "exact_transaction"),
    ("api.ABCrownSolver.verify", "verify_scope", "coarse_scope"),
    (
        "abcrown.ABCROWN.incomplete_verifier",
        "incomplete_verification",
        "exact_transaction",
    ),
    (
        "abcrown.ABCROWN.complete_verifier",
        "complete_verification_scope",
        "coarse_scope",
    ),
    ("abcrown.ABCROWN.bab", "bab_scope", "coarse_scope"),
    (
        "complete_verifier_func.general_bab",
        "bab_bootstrap_scope",
        "coarse_scope",
    ),
    (
        "incomplete.SpecHandler.expand_intermediate",
        "spec_handoff",
        "exact_transaction",
    ),
    (
        "beta_solver.LiRPANet.build_with_refined_bounds",
        "bab_bootstrap",
        "exact_transaction",
    ),
    (
        "bab_bootstrap.branch_and_bound_preprocess",
        "domain_preprocess",
        "exact_transaction",
    ),
    (
        "bab_bootstrap.branch_and_bound_solve",
        "domain_solve",
        "exact_transaction",
    ),
    (
        "bab_bootstrap.branch_and_bound_postprocess",
        "domain_postprocess",
        "exact_transaction",
    ),
    (
        "stage_preprocess.update_bounds_pre",
        "bound_prepare",
        "exact_transaction",
    ),
    (
        "stage_solve.update_bounds_core",
        "bound_core",
        "exact_transaction",
    ),
    (
        "stage_postprocess.update_bounds_post",
        "bound_postprocess",
        "exact_transaction",
    ),
    (
        "complete_verifier_func.prepare_for_act_bab",
        "spec_handoff",
        "exact_transaction",
    ),
    (
        "complete_verifier_func._format_result_act_bab",
        "result_publish",
        "exact_transaction",
    ),
    ("api.IOConstraints.__init__", "constraint_import", "exact_transaction"),
    (
        "api.ABCrownSolver._normalize_constraint",
        "constraint_import",
        "exact_transaction",
    ),
    (
        "api.ABCrownSolver._prepare_environment",
        "environment_setup",
        "exact_transaction",
    ),
    (
        "api.ABCrownSolver._prepare_model",
        "model_prepare",
        "exact_transaction",
    ),
    (
        "api.ABCrownSolver._prepare_runtime_spec",
        "spec_prepare",
        "exact_transaction",
    ),
    (
        "api.ABCrownSolver._build_vnnlib_handler",
        "spec_prepare",
        "exact_transaction",
    ),
    (
        "api.incomplete_verifier_core",
        "incomplete_verification",
        "exact_transaction",
    ),
    ("api.complete_verifier_core", "complete_verification_scope", "coarse_scope"),
    (
        "api._ApiLogger.summarize_results",
        "result_publish",
        "exact_transaction",
    ),
    (
        "api._ApiLogger.finish",
        "solver_termination",
        "exact_transaction",
    ),
    (
        "branching_domains.BatchedDomainList.__init__",
        "bab_bootstrap",
        "exact_transaction",
    ),
    (
        "bab_bootstrap.get_unstable_neurons",
        "bab_bootstrap",
        "exact_transaction",
    ),
    (
        "bab_bootstrap.get_branching_heuristic",
        "bab_bootstrap",
        "exact_transaction",
    ),
    (
        "bab_bootstrap.compute_first_iteration_decision",
        "bab_bootstrap",
        "exact_transaction",
    ),
    ("api.ABCrownSolver.bab", "bab_scope", "exact_transaction"),
    ("runtime.gc.collect", "host_garbage_collection", "exact_transaction"),
    (
        "runtime.torch.cuda.empty_cache",
        "device_cache_release",
        "exact_transaction",
    ),
)
CODE_FILES = (
    Path("boundflow/runtime/solver_transaction_observer.py"),
    Path("scripts/run_asplos27_s0_transaction_markers.py"),
    Path("scripts/run_fsg1_official_control_baseline.py"),
)
OUTPUT_FILES = (
    "protocol.json",
    "worker_runs.jsonl",
    "pairs.jsonl",
    "summary.json",
    "failure_rows.jsonl",
    "replay_stdout.txt",
    "README.md",
)


def canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"S0 transaction JSON root differs: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"S0 transaction JSONL row differs: {path}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _number(value: Mapping[str, Any], key: str) -> float:
    observed = value.get(key)
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        raise TypeError(f"S0 transaction numeric field differs: {key}")
    return float(observed)


def _target_inventory() -> list[dict[str, str]]:
    return [
        {"target_id": target_id, "category": category, "resolution": resolution}
        for target_id, category, resolution in TARGET_SPECS
    ]


def _verify_external_source(abcrown_root: Path, benchmark_root: Path) -> None:
    if fsg1._git_revision(abcrown_root) != fsg1.ABCROWN_COMMIT:
        raise ValueError("S0 transaction alpha-beta-CROWN commit differs")
    if fsg1._git_revision(abcrown_root / "auto_LiRPA") != fsg1.AUTO_LIRPA_COMMIT:
        raise ValueError("S0 transaction auto_LiRPA commit differs")
    if fsg1._git_revision(benchmark_root) != fsg1.VNNCOMP_COMMIT:
        raise ValueError("S0 transaction VNN-COMP commit differs")
    observed = {
        path: fsg1._git_value(abcrown_root, "rev-parse", f"HEAD:{path}")
        for path in TARGET_BLOBS
    }
    if observed != TARGET_BLOBS:
        raise ValueError("S0 transaction target source blob differs")


def _transaction_targets(
    abcrown_solver: Any, torch_module: Any
) -> tuple[TransactionTarget, ...]:
    api_module = importlib.import_module("api")
    abcrown_module = importlib.import_module("abcrown")
    complete_module = importlib.import_module("complete_verifier_func")
    incomplete_module = importlib.import_module("incomplete_verifier_func")
    beta_solver = importlib.import_module("beta_CROWN_solver")
    bootstrap = importlib.import_module("activation_split.bab_bootstrap")
    stage_pre = importlib.import_module("activation_split.stage_preprocess")
    stage_solve = importlib.import_module("activation_split.stage_solve")
    stage_post = importlib.import_module("activation_split.stage_postprocess")
    branching_domains = importlib.import_module("branching_domains")
    targets = (
        TransactionTarget(
            abcrown_solver,
            "__init__",
            TARGET_SPECS[0][0],
            TransactionCategory.FRONTEND_SETUP,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            abcrown_solver,
            "verify",
            TARGET_SPECS[1][0],
            TransactionCategory.VERIFY_SCOPE,
            TransactionResolution.COARSE_SCOPE,
        ),
        TransactionTarget(
            abcrown_module.ABCROWN,
            "incomplete_verifier",
            TARGET_SPECS[2][0],
            TransactionCategory.INCOMPLETE_VERIFICATION,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            abcrown_module.ABCROWN,
            "complete_verifier",
            TARGET_SPECS[3][0],
            TransactionCategory.COMPLETE_VERIFICATION_SCOPE,
            TransactionResolution.COARSE_SCOPE,
        ),
        TransactionTarget(
            abcrown_module.ABCROWN,
            "bab",
            TARGET_SPECS[4][0],
            TransactionCategory.BAB_SCOPE,
            TransactionResolution.COARSE_SCOPE,
        ),
        TransactionTarget(
            complete_module,
            "general_bab",
            TARGET_SPECS[5][0],
            TransactionCategory.BAB_BOOTSTRAP_SCOPE,
            TransactionResolution.COARSE_SCOPE,
        ),
        TransactionTarget(
            incomplete_module.SpecHandler,
            "expand_intermediate",
            TARGET_SPECS[6][0],
            TransactionCategory.SPEC_HANDOFF,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            beta_solver.LiRPANet,
            "build_with_refined_bounds",
            TARGET_SPECS[7][0],
            TransactionCategory.BAB_BOOTSTRAP,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            bootstrap,
            "branch_and_bound_preprocess",
            TARGET_SPECS[8][0],
            TransactionCategory.DOMAIN_PREPROCESS,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            bootstrap,
            "branch_and_bound_solve",
            TARGET_SPECS[9][0],
            TransactionCategory.DOMAIN_SOLVE,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            bootstrap,
            "branch_and_bound_postprocess",
            TARGET_SPECS[10][0],
            TransactionCategory.DOMAIN_POSTPROCESS,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            stage_pre,
            "update_bounds_pre",
            TARGET_SPECS[11][0],
            TransactionCategory.BOUND_PREPARE,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            stage_solve,
            "update_bounds_core",
            TARGET_SPECS[12][0],
            TransactionCategory.BOUND_CORE,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            stage_post,
            "update_bounds_post",
            TARGET_SPECS[13][0],
            TransactionCategory.BOUND_POSTPROCESS,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            complete_module,
            "prepare_for_act_bab",
            TARGET_SPECS[14][0],
            TransactionCategory.SPEC_HANDOFF,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            complete_module,
            "_format_result_act_bab",
            TARGET_SPECS[15][0],
            TransactionCategory.RESULT_PUBLISH,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            api_module.IOConstraints,
            "__init__",
            TARGET_SPECS[16][0],
            TransactionCategory.CONSTRAINT_IMPORT,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            abcrown_solver,
            "_normalize_constraint",
            TARGET_SPECS[17][0],
            TransactionCategory.CONSTRAINT_IMPORT,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            abcrown_solver,
            "_prepare_environment",
            TARGET_SPECS[18][0],
            TransactionCategory.ENVIRONMENT_SETUP,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            abcrown_solver,
            "_prepare_model",
            TARGET_SPECS[19][0],
            TransactionCategory.MODEL_PREPARE,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            abcrown_solver,
            "_prepare_runtime_spec",
            TARGET_SPECS[20][0],
            TransactionCategory.SPEC_PREPARE,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            abcrown_solver,
            "_build_vnnlib_handler",
            TARGET_SPECS[21][0],
            TransactionCategory.SPEC_PREPARE,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            api_module,
            "incomplete_verifier_core",
            TARGET_SPECS[22][0],
            TransactionCategory.INCOMPLETE_VERIFICATION,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            api_module,
            "complete_verifier_core",
            TARGET_SPECS[23][0],
            TransactionCategory.COMPLETE_VERIFICATION_SCOPE,
            TransactionResolution.COARSE_SCOPE,
        ),
        TransactionTarget(
            api_module._ApiLogger,
            "summarize_results",
            TARGET_SPECS[24][0],
            TransactionCategory.RESULT_PUBLISH,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            api_module._ApiLogger,
            "finish",
            TARGET_SPECS[25][0],
            TransactionCategory.SOLVER_TERMINATION,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            branching_domains.BatchedDomainList,
            "__init__",
            TARGET_SPECS[26][0],
            TransactionCategory.BAB_BOOTSTRAP,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            bootstrap,
            "get_unstable_neurons",
            TARGET_SPECS[27][0],
            TransactionCategory.BAB_BOOTSTRAP,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            bootstrap,
            "get_branching_heuristic",
            TARGET_SPECS[28][0],
            TransactionCategory.BAB_BOOTSTRAP,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            bootstrap,
            "compute_first_iteration_decision",
            TARGET_SPECS[29][0],
            TransactionCategory.BAB_BOOTSTRAP,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            abcrown_solver,
            "bab",
            TARGET_SPECS[30][0],
            TransactionCategory.BAB_SCOPE,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            gc,
            "collect",
            TARGET_SPECS[31][0],
            TransactionCategory.HOST_GARBAGE_COLLECTION,
            TransactionResolution.EXACT_TRANSACTION,
        ),
        TransactionTarget(
            torch_module.cuda,
            "empty_cache",
            TARGET_SPECS[32][0],
            TransactionCategory.DEVICE_CACHE_RELEASE,
            TransactionResolution.EXACT_TRANSACTION,
        ),
    )
    for target in targets:
        target.validate()
    observed = [
        {
            "target_id": target.target_id,
            "category": target.category.value,
            "resolution": target.resolution.value,
        }
        for target in targets
    ]
    if observed != _target_inventory():
        raise ValueError("S0 transaction resolved target inventory differs")
    return targets


def _solver_protocol(args: argparse.Namespace) -> dict[str, object]:
    return {
        "device": "cuda",
        "seed": 100,
        "reset_seed_after_precompile": True,
        "timeout_seconds": args.timeout_seconds,
        "max_iterations": args.max_iterations,
        "alpha_steps": args.alpha_steps,
        "beta_steps": args.beta_steps,
        "batch_size": args.batch_size,
        "auto_enlarge_batch_size": False,
        "complete_verifier": "bab",
        "attack_policy": "skip",
        "synchronize_outer_scope": True,
        "property_cache": "cold_isolated_copy",
    }


def _worker(args: argparse.Namespace) -> None:  # pylint: disable=too-many-locals
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints  # type: ignore[import-not-found]
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped]

    if not torch.cuda.is_available():
        raise RuntimeError("S0 transaction worker requires CUDA")
    _verify_external_source(args.abcrown_root, args.benchmark_root)
    targets = _transaction_targets(ABCrownSolver, torch)
    property_workspace = tempfile.TemporaryDirectory(prefix="boundflow-s0-txn-")
    isolated_property = Path(property_workspace.name) / args.property.name
    shutil.copy2(args.property, isolated_property)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    anchor = torch.cuda.Event(enable_timing=True)
    finish_event = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()
    anchor.record(stream)
    scope_started_ns = time.perf_counter_ns()
    compute_observer = fsg1._OfficialCallObserver(torch, scope_started_ns, anchor)
    transaction_observer = SolverTransactionObserver(scope_started_ns=scope_started_ns)
    config = (
        ConfigBuilder.from_defaults()
        .set("general/device", "cuda")
        .set("general/seed", 100)
        .set("general/reset_seed_after_precompile", True)
        .set("general/complete_verifier", "bab")
        .set("attack/pgd_order", "skip")
        .set("bab/timeout", args.timeout_seconds)
        .set("bab/max_iterations", args.max_iterations)
        .set("solver/batch_size", args.batch_size)
        .set("solver/auto_enlarge_batch_size", False)
        .set("solver/alpha-crown/iteration", args.alpha_steps)
        .set("solver/beta-crown/iteration", args.beta_steps)
    )
    with ExitStack() as stack:
        if args.mode == "profile":
            stack.enter_context(compute_observer.instrument(BoundedModule))
            stack.enter_context(transaction_observer.instrument(targets))
        else:
            stack.enter_context(nullcontext())
        solver = ABCrownSolver(str(args.model), config=config)
        result = solver.verify(
            constraints=IOConstraints(vnnlib_path=str(isolated_property))
        )
    finish_event.record(stream)
    torch.cuda.synchronize()
    host_scope_ns = time.perf_counter_ns() - scope_started_ns
    cuda_scope_ns = int(round(anchor.elapsed_time(finish_event) * 1e6))
    scope_ns = max(host_scope_ns, cuda_scope_ns)
    if args.mode == "profile":
        calls = compute_observer.finish(scope_ns=scope_ns)
        spans = transaction_observer.finish(scope_ns=scope_ns)
        transaction_summary: Mapping[str, object] | None = (
            summarize_solver_transactions(spans, compute_calls=calls, scope_ns=scope_ns)
        )
    else:
        calls = []
        spans = ()
        transaction_summary = None
    properties = torch.cuda.get_device_properties(0)
    record = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "run_id": args.run_id,
        "configuration_id": "B0",
        "workload_id": args.workload_id,
        "mode": args.mode,
        "repeat_index": args.repeat_index,
        "pair_order": args.pair_order,
        "source": {
            "abcrown_commit": fsg1._git_revision(args.abcrown_root),
            "auto_lirpa_commit": fsg1._git_revision(args.abcrown_root / "auto_LiRPA"),
            "vnncomp_commit": fsg1._git_revision(args.benchmark_root),
            "target_blobs": TARGET_BLOBS,
            "model_relative_path": args.model_relative_path,
            "property_relative_path": args.property_relative_path,
            "model_sha256": file_sha256(args.model),
            "property_sha256": file_sha256(args.property),
        },
        "solver_protocol": _solver_protocol(args),
        "observer_protocol": {
            "clock": "time.perf_counter_ns",
            "cuda_synchronization_added_by_transaction_observer": False,
            "stack_inspection": False,
            "tensor_reads": False,
            "minimum_mechanism_coverage": 0.97,
            "maximum_median_profile_perturbation": 1.05,
            "target_inventory": _target_inventory(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "gpu_name": properties.name,
            "gpu_total_memory": int(properties.total_memory),
        },
        "result": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": fsg1._visited_domains(result),
        },
        "scope_ns": scope_ns,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "compute_calls": calls,
        "transactions": [span.to_dict() for span in spans],
        "transaction_summary": transaction_summary,
        "performance_claimed": False,
    }
    record["worker_hash"] = canonical_hash(record)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.result_json, record)
    print(
        canonical_json(
            {
                "status": record["result"]["status"],
                "run_id": args.run_id,
                "scope_ns": scope_ns,
                "transaction_count": len(spans),
                "mechanism_coverage": (
                    transaction_summary["mechanism_coverage_share"]
                    if transaction_summary is not None
                    else None
                ),
            }
        ),
        flush=True,
    )
    property_workspace.cleanup()


def _selected_workloads(names: Sequence[str] | None) -> tuple[Mapping[str, str], ...]:
    return fsg1._selected_workloads(names)


def _run_worker(
    *,
    abcrown_python: Path,
    abcrown_root: Path,
    benchmark_root: Path,
    workload: Mapping[str, str],
    mode: str,
    repeat_index: int,
    pair_order: str,
    timeout_seconds: int,
    max_iterations: int,
    result_path: Path,
) -> dict[str, Any]:
    run_id = f"{workload['workload_id'].replace(':', '-')}-r{repeat_index}-{mode}"
    command = (
        str(abcrown_python),
        str(Path(__file__).resolve()),
        "worker",
        "--mode",
        mode,
        "--run-id",
        run_id,
        "--workload-id",
        workload["workload_id"],
        "--repeat-index",
        str(repeat_index),
        "--pair-order",
        pair_order,
        "--model",
        str(benchmark_root / workload["model"]),
        "--property",
        str(benchmark_root / workload["property"]),
        "--model-relative-path",
        workload["model"],
        "--property-relative-path",
        workload["property"],
        "--benchmark-root",
        str(benchmark_root),
        "--abcrown-root",
        str(abcrown_root),
        "--result-json",
        str(result_path),
        "--timeout-seconds",
        str(timeout_seconds),
        "--max-iterations",
        str(max_iterations),
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=fsg1._external_env(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds + 180,
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"S0 transaction worker {run_id} failed with {completed.returncode}: "
            f"{completed.stdout[-12000:]}"
        )
    print(completed.stdout.strip()[-2000:], flush=True)
    return _load_json(result_path)


def _without_hash(record: Mapping[str, Any]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "worker_hash"}


def _expected_worker_source(workload_id: str) -> dict[str, object]:
    workloads = {str(workload["workload_id"]): workload for workload in fsg1.WORKLOADS}
    if workload_id not in workloads:
        raise ValueError("S0 transaction worker workload differs")
    workload = workloads[workload_id]
    return {
        "abcrown_commit": fsg1.ABCROWN_COMMIT,
        "auto_lirpa_commit": fsg1.AUTO_LIRPA_COMMIT,
        "vnncomp_commit": fsg1.VNNCOMP_COMMIT,
        "target_blobs": TARGET_BLOBS,
        "model_relative_path": workload["model"],
        "property_relative_path": workload["property"],
        "model_sha256": workload["model_sha256"],
        "property_sha256": workload["property_sha256"],
    }


def _validate_observer_protocol(value: object) -> None:
    expected = {
        "clock": "time.perf_counter_ns",
        "cuda_synchronization_added_by_transaction_observer": False,
        "stack_inspection": False,
        "tensor_reads": False,
        "minimum_mechanism_coverage": 0.97,
        "maximum_median_profile_perturbation": 1.05,
        "target_inventory": _target_inventory(),
    }
    if value != expected:
        raise ValueError("S0 transaction observer protocol differs")


def _validate_worker(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != WORKER_SCHEMA_VERSION:
        raise ValueError("S0 transaction worker schema differs")
    if record.get("configuration_id") != "B0":
        raise ValueError("S0 transaction worker configuration differs")
    if record.get("performance_claimed") is not False:
        raise ValueError("S0 transaction worker cannot claim performance")
    if record.get("worker_hash") != canonical_hash(_without_hash(record)):
        raise ValueError("S0 transaction worker hash differs")
    mode = record.get("mode")
    if mode not in {"control", "profile"}:
        raise ValueError("S0 transaction worker mode differs")
    workload_id = record.get("workload_id")
    if not isinstance(workload_id, str):
        raise TypeError("S0 transaction worker workload identity differs")
    if record.get("source") != _expected_worker_source(workload_id):
        raise ValueError("S0 transaction worker source identity differs")
    _validate_observer_protocol(record.get("observer_protocol"))
    solver_protocol = record.get("solver_protocol")
    if not isinstance(solver_protocol, Mapping):
        raise TypeError("S0 transaction worker solver protocol differs")
    expected_solver_keys = {
        "device",
        "seed",
        "reset_seed_after_precompile",
        "timeout_seconds",
        "max_iterations",
        "alpha_steps",
        "beta_steps",
        "batch_size",
        "auto_enlarge_batch_size",
        "complete_verifier",
        "attack_policy",
        "synchronize_outer_scope",
        "property_cache",
    }
    if set(solver_protocol) != expected_solver_keys:
        raise ValueError("S0 transaction worker solver protocol fields differ")
    if (
        solver_protocol.get("device") != "cuda"
        or solver_protocol.get("seed") != 100
        or solver_protocol.get("reset_seed_after_precompile") is not True
        or solver_protocol.get("auto_enlarge_batch_size") is not False
        or solver_protocol.get("complete_verifier") != "bab"
        or solver_protocol.get("attack_policy") != "skip"
        or solver_protocol.get("synchronize_outer_scope") is not True
        or solver_protocol.get("property_cache") != "cold_isolated_copy"
    ):
        raise ValueError("S0 transaction worker solver protocol policy differs")
    scope_ns = record.get("scope_ns")
    if not isinstance(scope_ns, int) or isinstance(scope_ns, bool) or scope_ns <= 0:
        raise TypeError("S0 transaction worker scope differs")
    calls = record.get("compute_calls")
    transactions = record.get("transactions")
    if not isinstance(calls, list) or not isinstance(transactions, list):
        raise TypeError("S0 transaction worker trace differs")
    if mode == "control":
        if calls or transactions or record.get("transaction_summary") is not None:
            raise ValueError("S0 transaction control contains observer evidence")
        return
    if not calls or not transactions:
        raise ValueError("S0 transaction profile trace is empty")
    spans = tuple(
        host_transaction_span_from_dict(transaction) for transaction in transactions
    )
    expected_summary = summarize_solver_transactions(
        spans, compute_calls=calls, scope_ns=scope_ns
    )
    if record.get("transaction_summary") != expected_summary:
        raise ValueError("S0 transaction worker semantic summary differs")


def _compute_signature(record: Mapping[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "call_id": call["call_id"],
            "parent_call_id": call["parent_call_id"],
            "depth": call["depth"],
            "method": call["method"],
            "phase": call["phase"],
            "external_phase": call["external_phase"],
            "bound_lower": call["bound_lower"],
            "bound_upper": call["bound_upper"],
            "kwargs_keys": call["kwargs_keys"],
        }
        for call in record["compute_calls"]
    ]


def _derive_payloads(
    records: Sequence[Mapping[str, Any]], *, repeats: int
) -> dict[str, object]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for record in records:
        _validate_worker(record)
        key = (str(record["workload_id"]), int(record["repeat_index"]))
        mode = str(record["mode"])
        modes = grouped.setdefault(key, {})
        if mode in modes:
            raise ValueError("S0 transaction pair contains duplicate mode")
        modes[mode] = record
    pairs: list[dict[str, object]] = []
    profiles_by_workload: dict[str, list[Mapping[str, Any]]] = {}
    for (workload_id, repeat_index), modes in sorted(grouped.items()):
        if set(modes) != {"control", "profile"}:
            raise ValueError("S0 transaction pair is incomplete")
        control = modes["control"]
        profile = modes["profile"]
        semantic_fields = (
            "configuration_id",
            "workload_id",
            "repeat_index",
            "pair_order",
            "source",
            "solver_protocol",
            "environment",
            "result",
        )
        semantic_match = all(
            control[field] == profile[field] for field in semantic_fields
        )
        if not semantic_match:
            raise ValueError("S0 transaction control/profile semantics differ")
        profile_summary = profile["transaction_summary"]
        if not isinstance(profile_summary, Mapping):
            raise TypeError("S0 transaction profile summary differs")
        ratio = float(profile["scope_ns"]) / float(control["scope_ns"])
        pairs.append(
            {
                "workload_id": workload_id,
                "repeat_index": repeat_index,
                "pair_order": control["pair_order"],
                "control_run_id": control["run_id"],
                "profile_run_id": profile["run_id"],
                "control_scope_ns": control["scope_ns"],
                "profile_scope_ns": profile["scope_ns"],
                "perturbation_ratio": ratio,
                "mechanism_coverage_share": profile_summary["mechanism_coverage_share"],
                "mechanism_unresolved_share": profile_summary[
                    "mechanism_unresolved_share"
                ],
                "mechanism_admitted": profile_summary["mechanism_admitted"],
                "semantic_match": True,
                "profile_worker_hash": profile["worker_hash"],
                "transaction_summary_hash": profile_summary["summary_hash"],
                "performance_claimed": False,
            }
        )
        profiles_by_workload.setdefault(workload_id, []).append(profile)
    workload_summaries: dict[str, object] = {}
    all_passed = True
    for workload_id, profiles in sorted(profiles_by_workload.items()):
        if len(profiles) != repeats:
            raise ValueError("S0 transaction repeat count differs")
        signatures = [_compute_signature(profile) for profile in profiles]
        compute_signature_exact = all(
            signature == signatures[0] for signature in signatures[1:]
        )
        workload_pairs = [pair for pair in pairs if pair["workload_id"] == workload_id]
        perturbations = [_number(pair, "perturbation_ratio") for pair in workload_pairs]
        coverages = [
            _number(pair, "mechanism_coverage_share") for pair in workload_pairs
        ]
        unresolved = [
            _number(pair, "mechanism_unresolved_share") for pair in workload_pairs
        ]
        perturbation_median = statistics.median(perturbations)
        perturbation_passed = perturbation_median <= 1.05
        mechanism_passed = min(coverages) >= 0.97 and max(unresolved) <= 0.03
        gate = perturbation_passed and mechanism_passed and compute_signature_exact
        all_passed = all_passed and gate
        workload_summaries[workload_id] = {
            "repeat_count": len(profiles),
            "perturbation_ratios": perturbations,
            "median_perturbation_ratio": perturbation_median,
            "maximum_perturbation_ratio": max(perturbations),
            "perturbation_gate": 1.05,
            "perturbation_passed": perturbation_passed,
            "mechanism_coverages": coverages,
            "minimum_mechanism_coverage": min(coverages),
            "maximum_mechanism_unresolved": max(unresolved),
            "mechanism_gate": 0.97,
            "mechanism_passed": mechanism_passed,
            "compute_signature_exact": compute_signature_exact,
            "gate_passed": gate,
        }
    summary: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": (
            "s0-explicit-transactions-admitted"
            if all_passed
            else "s0-explicit-transactions-not-admitted"
        ),
        "configuration_id": "B0",
        "pair_count": len(pairs),
        "profile_count": len(profiles_by_workload) * repeats,
        "workloads": workload_summaries,
        "transaction_target_count": len(TARGET_SPECS),
        "budget_recompute_open": all_passed,
        "s1_performance_gate_open": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return {"pairs.jsonl": pairs, "summary.json": summary, "failure_rows.jsonl": []}


def _protocol(
    repeats: int, workloads: Sequence[Mapping[str, str]]
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "repeats": repeats,
        "workloads": [workload["workload_id"] for workload in workloads],
        "pair_order": "alternating_control_profile",
        "maximum_median_profile_perturbation": 1.05,
        "minimum_mechanism_coverage": 0.97,
        "maximum_mechanism_unresolved": 0.03,
        "target_inventory": _target_inventory(),
        "target_blobs": TARGET_BLOBS,
        "performance_claimed": False,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def _readme() -> str:
    return (
        "# ASPLOS'27 S0 Explicit Solver Transactions\n\n"
        "This artifact records alternating fresh official B0 control/profile pairs. "
        "It admits attribution only when semantic identity, <=1.05 median observer "
        "perturbation, and >=0.97 mechanism coverage all pass. It does not claim "
        "BoundFlow performance.\n"
    )


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "pair_count": summary["pair_count"],
        "budget_recompute_open": summary["budget_recompute_open"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _write_derived_payloads(artifact_dir: Path, payloads: Mapping[str, object]) -> None:
    for name, payload in payloads.items():
        if name.endswith(".jsonl"):
            _write_jsonl(
                artifact_dir / name,
                cast(Sequence[Mapping[str, Any]], payload),
            )
        else:
            _write_json(artifact_dir / name, payload)


def _write_artifact_envelope(
    artifact_dir: Path, summary: Mapping[str, Any]
) -> Mapping[str, Any]:
    replay_result = _replay_result(summary)
    (artifact_dir / "replay_stdout.txt").write_text(
        canonical_json(replay_result) + "\n", encoding="utf-8"
    )
    (artifact_dir / "README.md").write_text(_readme(), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": summary["status"],
        "source_git_head": fsg1._git_revision(ROOT),
        "code_revision": {str(path): file_sha256(ROOT / path) for path in CODE_FILES},
        "files": {name: file_sha256(artifact_dir / name) for name in OUTPUT_FILES},
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact_dir / "manifest.json", manifest)
    return replay_result


def generate_artifact(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.repeats < 1:
        raise ValueError("S0 transaction repeats must be positive")
    artifact_dir = args.artifact_dir.resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    workloads = _selected_workloads(args.workload)
    benchmark_root = args.benchmark_root.resolve()
    abcrown_root = args.abcrown_root.resolve()
    # Preserve the venv launcher identity; resolving its symlink selects bare UV Python.
    abcrown_python = Path(os.path.abspath(args.abcrown_python))
    fsg1._validate_inputs(benchmark_root, abcrown_root, abcrown_python, workloads)
    _verify_external_source(abcrown_root, benchmark_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-s0-txn-parent-") as temporary:
        temporary_root = Path(temporary)
        for workload in workloads:
            for repeat_index in range(args.repeats):
                modes = (
                    ("control", "profile")
                    if repeat_index % 2 == 0
                    else ("profile", "control")
                )
                pair_order = "-".join(modes)
                for mode in modes:
                    result_path = temporary_root / (
                        f"{workload['workload_id'].replace(':', '-')}-"
                        f"r{repeat_index}-{mode}.json"
                    )
                    try:
                        records.append(
                            _run_worker(
                                abcrown_python=abcrown_python,
                                abcrown_root=abcrown_root,
                                benchmark_root=benchmark_root,
                                workload=workload,
                                mode=mode,
                                repeat_index=repeat_index,
                                pair_order=pair_order,
                                timeout_seconds=args.timeout_seconds,
                                max_iterations=args.max_iterations,
                                result_path=result_path,
                            )
                        )
                    except Exception as error:  # pylint: disable=broad-exception-caught
                        failures.append(
                            {
                                "workload_id": workload["workload_id"],
                                "repeat_index": repeat_index,
                                "mode": mode,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                        break
                if failures:
                    break
            if failures:
                break
    _write_jsonl(artifact_dir / "worker_runs.jsonl", records)
    _write_jsonl(artifact_dir / "failure_rows.jsonl", failures)
    if failures:
        raise RuntimeError("S0 transaction worker failed; see failure_rows.jsonl")
    _write_json(artifact_dir / "protocol.json", _protocol(args.repeats, workloads))
    payloads = _derive_payloads(records, repeats=args.repeats)
    _write_derived_payloads(artifact_dir, payloads)
    summary = _load_json(artifact_dir / "summary.json")
    return _write_artifact_envelope(artifact_dir, summary)


def refresh_artifact_from_raw(artifact_dir: Path) -> Mapping[str, Any]:
    """Regenerate derived files and hashes without changing frozen worker raw."""

    artifact_dir = artifact_dir.resolve()
    protocol = _load_json(artifact_dir / "protocol.json")
    repeats = protocol.get("repeats")
    workload_ids = protocol.get("workloads")
    if (
        not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or repeats < 1
        or not isinstance(workload_ids, list)
        or not workload_ids
        or any(not isinstance(item, str) for item in workload_ids)
    ):
        raise ValueError("S0 transaction refresh protocol differs")
    workloads = _selected_workloads(cast(Sequence[str], workload_ids))
    if protocol != _protocol(repeats, workloads):
        raise ValueError("S0 transaction refresh protocol semantics differ")
    records = _read_jsonl(artifact_dir / "worker_runs.jsonl")
    payloads = _derive_payloads(records, repeats=repeats)
    _write_derived_payloads(artifact_dir, payloads)
    summary = _load_json(artifact_dir / "summary.json")
    return _write_artifact_envelope(artifact_dir, summary)


def replay_artifact(artifact_dir: Path) -> Mapping[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
        or manifest.get("manifest_hash") != canonical_hash(semantic_manifest)
    ):
        raise ValueError("S0 transaction manifest envelope differs")
    code_revision = manifest.get("code_revision")
    if not isinstance(code_revision, Mapping) or set(code_revision) != {
        str(path) for path in CODE_FILES
    }:
        raise ValueError("S0 transaction code revision inventory differs")
    for path in CODE_FILES:
        if code_revision[str(path)] != file_sha256(ROOT / path):
            raise ValueError("S0 transaction code revision differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(OUTPUT_FILES):
        raise ValueError("S0 transaction output inventory differs")
    for name in OUTPUT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError("S0 transaction artifact file digest differs")
    protocol = _load_json(artifact_dir / "protocol.json")
    repeats = protocol.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise TypeError("S0 transaction protocol repeats differ")
    workload_ids = protocol.get("workloads")
    if (
        not isinstance(workload_ids, list)
        or not workload_ids
        or any(not isinstance(item, str) for item in workload_ids)
        or len(workload_ids) != len(set(workload_ids))
    ):
        raise TypeError("S0 transaction protocol workloads differ")
    workloads = _selected_workloads(cast(Sequence[str], workload_ids))
    if protocol != _protocol(repeats, workloads):
        raise ValueError("S0 transaction protocol semantics differ")
    records = _read_jsonl(artifact_dir / "worker_runs.jsonl")
    payloads = _derive_payloads(records, repeats=repeats)
    for name, payload in payloads.items():
        if name.endswith(".jsonl"):
            rows = cast(Sequence[Mapping[str, Any]], payload)
            expected = "".join(canonical_json(dict(row)) + "\n" for row in rows)
        else:
            expected = canonical_json(payload, indent=2) + "\n"
        if (artifact_dir / name).read_text(encoding="utf-8") != expected:
            raise ValueError(f"S0 transaction semantic replay differs: {name}")
    summary = _load_json(artifact_dir / "summary.json")
    if manifest.get("status") != summary.get("status") or manifest.get(
        "summary_hash"
    ) != summary.get("summary_hash"):
        raise ValueError("S0 transaction manifest summary projection differs")
    result = _replay_result(summary)
    if (artifact_dir / "replay_stdout.txt").read_text(encoding="utf-8") != (
        canonical_json(result) + "\n"
    ):
        raise ValueError("S0 transaction replay stdout differs")
    if (artifact_dir / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("S0 transaction README differs")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    generate.add_argument("--repeats", type=int, default=5)
    generate.add_argument("--timeout-seconds", type=int, default=60)
    generate.add_argument("--max-iterations", type=int, default=16)
    generate.add_argument("--workload", action="append")
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    refresh = commands.add_parser("refresh-derived")
    refresh.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    worker = commands.add_parser("worker")
    worker.add_argument("--mode", choices=("control", "profile"), required=True)
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--workload-id", required=True)
    worker.add_argument("--repeat-index", type=int, required=True)
    worker.add_argument("--pair-order", required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--model-relative-path", required=True)
    worker.add_argument("--property-relative-path", required=True)
    worker.add_argument("--benchmark-root", type=Path, required=True)
    worker.add_argument("--abcrown-root", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--timeout-seconds", type=int, required=True)
    worker.add_argument("--max-iterations", type=int, required=True)
    worker.add_argument("--alpha-steps", type=int, default=5)
    worker.add_argument("--beta-steps", type=int, default=10)
    worker.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _worker(args)
        return
    if args.command == "generate":
        result = generate_artifact(args)
    elif args.command == "refresh-derived":
        result = refresh_artifact_from_raw(args.artifact_dir)
    else:
        result = replay_artifact(args.artifact_dir)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
