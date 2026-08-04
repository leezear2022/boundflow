#!/usr/bin/env python3
"""Generate or replay the NRIR-22 ancestral-constraint refinement artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,duplicate-code
# pylint: disable=import-outside-toplevel,import-error,wrong-import-position
# pylint: disable=missing-function-docstring,line-too-long,protected-access

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
    canonical_hash,
    file_sha256,
)
from scripts.run_native_intermediate_refinement_artifact import (
    _canonical_json,
    _load_json,
    _mapping,
    _sha256,
    _write_json,
)
from scripts.run_per_child_objective_refinement_artifact import (
    BACKWARD_CHUNK_SIZE,
    CLAUSE_INDICES,
    MAX_DEPTH,
    MAX_NODES,
    TARGETS_PER_RELU,
    TORCH_THREADS,
    WORKER_TIMEOUT_SECONDS,
    _resolved_source as _base_resolved_source,
    _source_payload,
    _validate_worker_result,
    _without_diagnostic_timing,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.ancestral-constraint-refinement-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.ancestral-constraint-refinement-evidence/v1"
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
WORKLOAD_ID = "cifar10_resnet:000"
MODES = ("root_global", "independent_per_child", "ancestral_carry")
MODE_EXECUTION = {
    "root_global": ("root_global", "independent_exact_split_v1"),
    "independent_per_child": ("per_child", "independent_exact_split_v1"),
    "ancestral_carry": ("per_child", "ancestral_constraint_carry_v1"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def _native_code_revision() -> str:
    paths = (
        "boundflow/ir/refinement.py",
        "boundflow/runtime/crown_ibp.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/runtime/native_alpha_beta_optimization_state.py",
        "boundflow/runtime/native_optimized_relu_split_bab_runtime.py",
        "scripts/run_per_child_objective_refinement_artifact.py",
        "scripts/run_ancestral_constraint_refinement_artifact.py",
    )
    return canonical_hash({path: file_sha256(REPO_ROOT / path) for path in paths})


def _resolved_source(benchmark_root: Path) -> dict[str, object]:
    source = _base_resolved_source(benchmark_root)
    source["native_code_revision"] = _native_code_revision()
    return source


def _policy_payload() -> dict[str, object]:
    return {
        "device": "cpu",
        "torch_threads": TORCH_THREADS,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "alpha_steps": 5,
        "max_nodes": MAX_NODES,
        "max_depth": MAX_DEPTH,
        "expansion_batch_size": 2,
        "max_eval_batch_size": 4,
        "targets_per_relu": TARGETS_PER_RELU,
        "backward_chunk_size": BACKWARD_CHUNK_SIZE,
        "candidate_policy_id": "objective_influence_width_per_relu_v1",
        "source_admission": "validated_parent_refinement_execution",
        "source_consumption": "sound_constraint_only",
        "comparison": "root-global-vs-independent-vs-ancestral-carry",
        "performance_claimed": False,
    }


def _worker_command(
    *, mode: str, clause_index: int, source: Mapping[str, object], result: Path
) -> list[str]:
    worker_mode, strategy = MODE_EXECUTION[mode]
    input_shape = cast(tuple[int, ...], source["input_shape"])
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/run_per_child_objective_refinement_artifact.py"),
        "worker",
        "--mode",
        worker_mode,
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
        "--per-child-strategy",
        strategy,
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
            f"NRIR-22 clause {clause_index} {mode} worker failed with "
            f"{completed.returncode}: {completed.stdout[-8000:]}"
        )
    return _load_json(result), completed.stdout, completed.returncode, elapsed_ns


def _comparison(
    clause_index: int,
    root_global: Mapping[str, Any],
    independent: Mapping[str, Any],
    carry: Mapping[str, Any],
) -> dict[str, object]:
    carry_vs_independent = float(carry["worst_leaf_lower"]) - float(
        independent["worst_leaf_lower"]
    )
    carry_vs_root = float(carry["worst_leaf_lower"]) - float(
        root_global["worst_leaf_lower"]
    )
    root_delta = float(carry["root_lower"]) - float(root_global["root_lower"])
    return {
        "clause_index": clause_index,
        "root_global_root_lower": root_global["root_lower"],
        "independent_root_lower": independent["root_lower"],
        "ancestral_root_lower": carry["root_lower"],
        "root_lower_delta": root_delta,
        "root_bound_same": abs(root_delta) <= 1e-5,
        "root_global_worst_leaf_lower": root_global["worst_leaf_lower"],
        "independent_worst_leaf_lower": independent["worst_leaf_lower"],
        "ancestral_worst_leaf_lower": carry["worst_leaf_lower"],
        "ancestral_vs_independent_delta": carry_vs_independent,
        "ancestral_vs_root_global_delta": carry_vs_root,
        "strictly_improves_independent": carry_vs_independent > 0.0,
        "not_weaker_than_root_global": carry_vs_root >= -1e-5,
        "root_global_leaf_lowers": root_global["leaf_lowers"],
        "independent_leaf_lowers": independent["leaf_lowers"],
        "ancestral_leaf_lowers": carry["leaf_lowers"],
        "performance_claimed": False,
    }


def _build_evidence(
    benchmark_root: Path, artifact_dir: Path
) -> tuple[dict[str, object], dict[str, str]]:
    source = _resolved_source(benchmark_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir22-") as temporary:
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
            by_identity[(clause_index, "root_global")],
            by_identity[(clause_index, "independent_per_child")],
            by_identity[(clause_index, "ancestral_carry")],
        )
        for clause_index in CLAUSE_INDICES
    ]
    passed = all(
        bool(item["strictly_improves_independent"])
        and bool(item["not_weaker_than_root_global"])
        and bool(item["root_bound_same"])
        for item in comparisons
    )
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "validated_reduced" if passed else "validated_no_go",
        "claim_boundary": (
            "same-policy, same-seven-node/depth-two CPU tightness comparison of "
            "root-global reuse, independent exact-split recomputation, and "
            "validated ancestral-constraint carry-forward on two fixed VNN-COMP "
            "ResNet clauses; no speedup or property-closure claim"
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
            "The fixed experiment covers two clauses, one model, seven nodes, and depth two; it is not a complete property result.",
            "Ancestral constraints come only from validated native parent refinement executions and are consumed as sound constraints, not child exact states.",
            "Per-node refinement is serial before packed optimizer execution; no latency or speedup claim is made.",
            "The bounded-tree result need not generalize to deeper trees, other clauses, or other workloads.",
            "CUDA, repeated timing, competitor parity, complete closure, and ASPLOS-ready claims remain pending.",
        ],
        "performance_claimed": False,
    }
    return evidence, files


def _validate_ancestral_lineage(result: Mapping[str, Any]) -> None:
    queue = _mapping(result.get("queue_trace"), "NRIR-22 queue")
    if queue.get("per_child_refinement_strategy") != ("ancestral_constraint_carry_v1"):
        raise ValueError("NRIR-22 ancestral strategy identity differs")
    evaluations = cast(list[Mapping[str, Any]], queue["evaluations"])
    records = cast(list[Mapping[str, Any]], queue["per_child_refinements"])
    programs = {
        str(program["node_id"]): program
        for program in cast(list[Mapping[str, Any]], result["refinement_programs"])
    }
    record_by_node = {str(record["node_id"]): record for record in records}
    for evaluation, record in zip(evaluations, records):
        node = _mapping(evaluation["node"], "NRIR-22 node")
        node_id = str(node["node_id"])
        plan = _mapping(programs[node_id]["plan"], "NRIR-22 Plan")
        parent_id = node.get("parent_node_id")
        if parent_id is None:
            if (
                any(
                    key in plan
                    for key in (
                        "source_intermediate_constraints_hash",
                        "source_refinement_plan_hash",
                        "source_refinement_semantic_trace_hash",
                    )
                )
                or "source_parent_node_id" in record
            ):
                raise ValueError("NRIR-22 ancestral root declares a source")
            continue
        parent = record_by_node[str(parent_id)]
        if (
            record.get("source_parent_node_id") != parent_id
            or record.get("source_consumption") != "sound_constraint_only"
            or record.get("parent_refinement_consumed_as_exact") is not False
            or record.get("source_intermediate_constraints_hash")
            != parent.get("final_intermediate_bounds_hash")
            or record.get("source_refinement_plan_hash")
            != parent.get("refinement_plan_hash")
            or record.get("source_refinement_semantic_trace_hash")
            != parent.get("refinement_semantic_trace_hash")
            or plan.get("source_intermediate_constraints_hash")
            != record.get("source_intermediate_constraints_hash")
            or plan.get("source_refinement_plan_hash")
            != record.get("source_refinement_plan_hash")
            or plan.get("source_refinement_semantic_trace_hash")
            != record.get("source_refinement_semantic_trace_hash")
        ):
            raise ValueError("NRIR-22 ancestral source lineage differs")
        task_module = _mapping(programs[node_id]["task"], "NRIR-22 Task module")
        materialize = cast(list[Mapping[str, Any]], task_module["tasks"])[0]
        if "refine.source_intermediate_constraints" not in cast(
            list[str], materialize["input_value_ids"]
        ):
            raise ValueError("NRIR-22 source constraint Task dependency differs")


def _validate_mode_result(mode: str, result: Mapping[str, Any]) -> None:
    _validate_worker_result(result)
    queue = _mapping(result["queue_trace"], "NRIR-22 queue")
    if mode == "root_global":
        if result.get("mode") != "root_global":
            raise ValueError("NRIR-22 root-global worker mode differs")
    elif mode == "independent_per_child":
        if (
            result.get("mode") != "per_child"
            or "per_child_refinement_strategy" in queue
            or any(
                "source_parent_node_id" in item
                for item in cast(
                    list[Mapping[str, Any]], queue["per_child_refinements"]
                )
            )
        ):
            raise ValueError("NRIR-22 independent baseline differs")
    else:
        if result.get("mode") != "per_child":
            raise ValueError("NRIR-22 ancestral worker mode differs")
        _validate_ancestral_lineage(result)


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") not in {"validated_reduced", "validated_no_go"}
        or evidence.get("performance_claimed") is not False
        or "no speedup" not in str(evidence.get("claim_boundary"))
    ):
        raise ValueError("NRIR-22 evidence header differs")
    source = _mapping(evidence.get("source"), "NRIR-22 source")
    if (
        source.get("workload_id") != WORKLOAD_ID
        or not _sha256(source.get("native_code_revision"))
        or not _sha256(source.get("model_sha256"))
        or not _sha256(source.get("property_sha256"))
    ):
        raise ValueError("NRIR-22 source identity differs")
    records = evidence.get("records")
    if not isinstance(records, list):
        raise TypeError("NRIR-22 records must be a list")
    identities: set[tuple[int, str]] = set()
    by_identity: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in records:
        item = _mapping(record, "NRIR-22 record")
        identity = (int(item["clause_index"]), str(item["mode"]))
        result = _mapping(item["result"], "NRIR-22 result")
        _validate_mode_result(identity[1], result)
        if (
            identity in identities
            or identity[0] != int(result["clause_index"])
            or item.get("process_returncode") != 0
            or not isinstance(item.get("e2e_elapsed_ns"), int)
            or int(item["e2e_elapsed_ns"]) <= 0
            or not str(item.get("log_path", "")).startswith("logs/")
            or not _sha256(item.get("log_sha256"))
        ):
            raise ValueError("NRIR-22 execution record differs")
        identities.add(identity)
        by_identity[identity] = result
    expected = {
        (clause_index, mode) for clause_index in CLAUSE_INDICES for mode in MODES
    }
    if identities != expected:
        raise ValueError("NRIR-22 record coverage differs")
    comparisons = evidence.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != len(CLAUSE_INDICES):
        raise ValueError("NRIR-22 comparison coverage differs")
    passed: list[bool] = []
    for comparison in comparisons:
        item = _mapping(comparison, "NRIR-22 comparison")
        clause = int(item["clause_index"])
        root = by_identity[(clause, "root_global")]
        independent = by_identity[(clause, "independent_per_child")]
        carry = by_identity[(clause, "ancestral_carry")]
        independent_delta = float(carry["worst_leaf_lower"]) - float(
            independent["worst_leaf_lower"]
        )
        root_delta = float(carry["worst_leaf_lower"]) - float(root["worst_leaf_lower"])
        root_bound_delta = float(carry["root_lower"]) - float(root["root_lower"])
        condition = independent_delta > 0.0 and root_delta >= -1e-5
        if (
            abs(float(item["ancestral_vs_independent_delta"]) - independent_delta)
            > 1e-7
            or abs(float(item["ancestral_vs_root_global_delta"]) - root_delta) > 1e-7
            or bool(item["strictly_improves_independent"]) != (independent_delta > 0.0)
            or bool(item["not_weaker_than_root_global"]) != (root_delta >= -1e-5)
            or bool(item["root_bound_same"]) != (abs(root_bound_delta) <= 1e-5)
            or item.get("performance_claimed") is not False
        ):
            raise ValueError("NRIR-22 tightness comparison differs")
        passed.append(condition and abs(root_bound_delta) <= 1e-5)
    expected_status = "validated_reduced" if all(passed) else "validated_no_go"
    if evidence.get("status") != expected_status:
        raise ValueError("NRIR-22 closure status differs")
    if (
        not isinstance(evidence.get("limitations"), list)
        or len(cast(list[object], evidence["limitations"])) != 5
    ):
        raise ValueError("NRIR-22 limitation ledger differs")


def _generate(args: argparse.Namespace) -> None:
    evidence, files = _build_evidence(args.benchmark_root, args.artifact_dir)
    validate_evidence_structure(evidence)
    evidence_path = args.artifact_dir / EVIDENCE_FILE
    _write_json(evidence_path, evidence)
    files[EVIDENCE_FILE] = file_sha256(evidence_path)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": evidence["status"],
        "performance_claimed": False,
        "files": dict(sorted(files.items())),
        "evidence_hash": canonical_hash(evidence),
    }
    _write_json(args.artifact_dir / MANIFEST_FILE, manifest)
    print(
        _canonical_json(
            {"status": evidence["status"], "evidence_hash": manifest["evidence_hash"]}
        )
    )


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
        or manifest.get("status") != evidence.get("status")
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != actual_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-22 artifact manifest differs")
    validate_evidence_structure(evidence)
    source = _resolved_source(args.benchmark_root)
    if evidence.get("source") != _source_payload(source):
        raise ValueError("NRIR-22 source replay differs")
    stored = {
        (int(record["clause_index"]), str(record["mode"])): _mapping(
            record["result"], "NRIR-22 stored result"
        )
        for record in cast(list[Mapping[str, Any]], evidence["records"])
    }
    semantic_hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir22-replay-") as temporary:
        temp_root = Path(temporary)
        for clause_index in CLAUSE_INDICES:
            for mode in MODES:
                result_path = temp_root / f"clause-{clause_index}-{mode}.json"
                fresh, _log, _returncode, _elapsed = _run_worker(
                    mode=mode,
                    clause_index=clause_index,
                    source=source,
                    result=result_path,
                )
                _validate_mode_result(mode, fresh)
                deterministic = _without_diagnostic_timing(fresh)
                if deterministic != _without_diagnostic_timing(
                    stored[(clause_index, mode)]
                ):
                    raise ValueError("NRIR-22 source-to-IR semantic replay differs")
                semantic_hashes[f"clause-{clause_index}-{mode}"] = canonical_hash(
                    deterministic
                )
    print(
        _canonical_json(
            {
                "status": "replayed",
                "closure": evidence["status"],
                "evidence_hash": manifest["evidence_hash"],
                "semantic_result_hashes": semantic_hashes,
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
