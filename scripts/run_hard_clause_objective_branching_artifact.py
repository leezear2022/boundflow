#!/usr/bin/env python3
"""Generate or replay NRIR-17 objective-aware hard-clause branching evidence."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-boolean-expressions,duplicate-code
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_objective_branch_score import (
    NativeObjectiveBranchPolicy,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NativeOptimizedReluSplitBabExecution,
    execute_native_optimized_relu_split_bab,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from scripts.run_end_to_end_tightness_performance_baseline import _build_context
from scripts.run_native_real_network_ir_artifact import (
    ABCROWN_COMMIT,
    INTERMEDIATE_BOUNDS_SHA256,
    MODEL_SHA256,
    VNNCOMP_COMMIT,
    VNNLIB_SHA256,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.hard-clause-objective-branching-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.hard-clause-objective-branching-evidence/v1"
ARTIFACT_FILE = "objective_branching.json"
MANIFEST_FILE = "manifest.json"
HARD_CLAUSES = (0, 2, 4)
OPTIMIZER_POLICY = NativeAlphaBetaOptimizerPolicy(
    steps=25,
    lr=0.2,
    alpha_initialization_mode="adaptive",
)
BRANCH_POLICY = NativeObjectiveBranchPolicy(
    candidates_per_relu=8,
    candidate_batch_size=64,
    max_candidates=256,
)
QUEUE_CONFIG = NativeReluSplitBabConfig(
    max_nodes=7,
    max_depth=2,
    expansion_batch_size=2,
    max_eval_batch_size=4,
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
        raise TypeError(f"{path} must contain an object")
    return value


def _branch_evidence(execution: NativeOptimizedReluSplitBabExecution) -> list[object]:
    rows: list[object] = []
    for node_id, branch in execution.objective_branch_executions:
        rows.append(
            {
                "node_id": node_id,
                "plan": branch.program.plan.to_dict(),
                "task_module": branch.program.task_module.to_dict(
                    plan=branch.program.plan
                ),
                "schedule": branch.program.schedule.to_dict(
                    plan=branch.program.plan,
                    task_module=branch.program.task_module,
                ),
                "trace": branch.trace.to_dict(program=branch.program),
                "trace_hash": branch.trace.stable_hash(program=branch.program),
            }
        )
    return rows


def _semantic(
    execution: NativeOptimizedReluSplitBabExecution, *, variant: str
) -> dict[str, object]:
    execution.validate()
    trace = execution.trace
    evaluations = [item.to_dict() for item in trace.evaluations]
    decisions = [item.to_dict() for item in trace.decisions]
    leaf_lowers = [
        float(item.lower)
        for item in trace.evaluations
        if item.node.depth == QUEUE_CONFIG.max_depth
    ]
    if len(leaf_lowers) != 4:
        raise ValueError("NRIR-17 bounded tree leaf count differs")
    root = float(trace.evaluations[0].lower)
    summary = {
        "root_lower": root,
        "leaf_worst_lower": min(leaf_lowers),
        "leaf_best_lower": max(leaf_lowers),
        "worst_leaf_improvement": min(leaf_lowers) - root,
        "proof_deficit": max(0.0, -min(leaf_lowers)),
        "evaluated_nodes": len(trace.evaluations),
        "terminal_leaves": len(leaf_lowers),
        "queue_status": trace.status,
        "property_status": "unknown",
    }
    return {
        "variant": variant,
        "trace": trace.to_dict(),
        "trace_hash": trace.stable_hash(),
        "evaluations": evaluations,
        "decisions": decisions,
        "objective_branches": (
            _branch_evidence(execution) if variant == "objective" else []
        ),
        "summary": summary,
    }


def _run_variant(
    context: Mapping[str, Any], *, clause: int, variant: str
) -> tuple[dict[str, object], int]:
    tensors = _mapping(context["tensors"], "NRIR-17 tensors")
    started_ns = time.perf_counter_ns()
    execution = execute_native_optimized_relu_split_bab(
        context["module"],
        context["input_spec"],
        linear_spec_C=tensors["linear_spec_c"][:, clause : clause + 1],
        run_id=f"nrir17:c{clause}:{variant}",
        config=QUEUE_CONFIG,
        optimizer_policy=OPTIMIZER_POLICY,
        relu_pre_override=context["external_relu_pre"],
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        objective_branch_policy=(BRANCH_POLICY if variant == "objective" else None),
    )
    elapsed_ns = time.perf_counter_ns() - started_ns
    return _semantic(execution, variant=variant), elapsed_ns


def _run_protocol(context: Mapping[str, Any]) -> tuple[dict[str, object], list[object]]:
    clauses: dict[str, object] = {}
    timings: list[object] = []
    orders = {
        0: ("widest", "objective"),
        2: ("objective", "widest"),
        4: ("widest", "objective"),
    }
    for clause in HARD_CLAUSES:
        variants: dict[str, object] = {}
        for variant in orders[clause]:
            semantic, elapsed_ns = _run_variant(context, clause=clause, variant=variant)
            variants[variant] = semantic
            timings.append(
                {
                    "clause_index": clause,
                    "variant": variant,
                    "elapsed_ns": elapsed_ns,
                }
            )
        clauses[str(clause)] = variants
    return clauses, timings


def build_evidence(args: argparse.Namespace) -> dict[str, object]:
    context = _build_context(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
        local_artifact_dir=args.local_artifact_dir,
    )
    clauses, timings = _run_protocol(context)
    comparisons: dict[str, object] = {}
    for clause in HARD_CLAUSES:
        variants = _mapping(clauses[str(clause)], "NRIR-17 variants")
        widest = _mapping(variants["widest"], "NRIR-17 widest")
        objective = _mapping(variants["objective"], "NRIR-17 objective")
        widest_summary = _mapping(widest["summary"], "NRIR-17 widest summary")
        objective_summary = _mapping(objective["summary"], "NRIR-17 objective summary")
        comparisons[str(clause)] = {
            "widest_leaf_worst_lower": widest_summary["leaf_worst_lower"],
            "objective_leaf_worst_lower": objective_summary["leaf_worst_lower"],
            "objective_minus_widest_leaf_worst": (
                float(objective_summary["leaf_worst_lower"])
                - float(widest_summary["leaf_worst_lower"])
            ),
            "objective_not_weaker": bool(
                float(objective_summary["leaf_worst_lower"])
                >= float(widest_summary["leaf_worst_lower"]) - 1e-6
            ),
        }
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "property_status": "unknown",
        "source": {
            **_mapping(context["source"], "NRIR-17 source"),
            "model_sha256": file_sha256(args.model),
            "vnnlib_sha256": VNNLIB_SHA256,
            "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
            "vnncomp_commit": VNNCOMP_COMMIT,
            "abcrown_commit": ABCROWN_COMMIT,
        },
        "protocol": {
            "hard_clause_indices": list(HARD_CLAUSES),
            "queue_config": QUEUE_CONFIG.to_dict(),
            "optimizer_policy": OPTIMIZER_POLICY.to_dict(),
            "branch_policy": BRANCH_POLICY.to_dict(),
            "variants": ["widest", "objective"],
            "timing_role": "single-run diagnostic only",
        },
        "clauses": clauses,
        "comparisons": comparisons,
        "timing": timings,
        "gates": {
            "first_class_branch_plan_task_schedule": True,
            "exact_selected_state_and_objective_binding": True,
            "deterministic_argmax_worst_child_selection": True,
            "same_optimizer_and_tree_budget_for_widest_control": True,
            "objective_not_weaker_on_all_hard_clauses": all(
                _mapping(value, "NRIR-17 comparison")["objective_not_weaker"] is True
                for value in comparisons.values()
            ),
            "unknown_property_status_preserved": True,
        },
        "limitations": [
            "fixed ResNet property 0 CPU only",
            "seven-node depth-two bounded tree is not a complete verifier result",
            "single-run wall time is diagnostic and not a performance claim",
            "clauses 0, 2, and 4 remain unknown when every terminal leaf is negative",
            "GPU, multi-workload, and external competitor timing remain pending",
        ],
    }
    validate_evidence(evidence)
    return evidence


def _validate_semantic(value: Mapping[str, Any], *, objective: bool) -> None:
    trace = _mapping(value.get("trace"), "NRIR-17 trace")
    evaluations = _list(value.get("evaluations"), "NRIR-17 evaluations")
    decisions = _list(value.get("decisions"), "NRIR-17 decisions")
    summary = _mapping(value.get("summary"), "NRIR-17 summary")
    branches = _list(value.get("objective_branches"), "NRIR-17 branch evidence")
    if (
        trace.get("status") != "complete"
        or trace.get("final_frontier_node_ids") != []
        or len(evaluations) != len(decisions) != 7
        or len(evaluations) != 7
        or summary.get("evaluated_nodes") != 7
        or summary.get("terminal_leaves") != 4
        or summary.get("queue_status") != "complete"
        or summary.get("property_status") != "unknown"
        or value.get("trace_hash") != canonical_hash(trace)
    ):
        raise ValueError("NRIR-17 queue semantic differs")
    leaf_lowers = [
        float(_mapping(item, "NRIR-17 evaluation")["lower"])
        for item in evaluations
        if _mapping(_mapping(item, "NRIR-17 evaluation")["node"], "NRIR-17 node")[
            "depth"
        ]
        == 2
    ]
    root = float(_mapping(evaluations[0], "NRIR-17 root")["lower"])
    if (
        len(leaf_lowers) != 4
        or summary.get("root_lower") != root
        or summary.get("leaf_worst_lower") != min(leaf_lowers)
        or summary.get("leaf_best_lower") != max(leaf_lowers)
        or summary.get("worst_leaf_improvement") != min(leaf_lowers) - root
        or summary.get("proof_deficit") != max(0.0, -min(leaf_lowers))
    ):
        raise ValueError("NRIR-17 leaf summary differs")
    if objective:
        if len(branches) != 7:
            raise ValueError("NRIR-17 objective branch coverage differs")
        for branch in branches:
            row = _mapping(branch, "NRIR-17 objective branch")
            plan = _mapping(row.get("plan"), "NRIR-17 branch plan")
            task = _mapping(row.get("task_module"), "NRIR-17 branch task")
            schedule = _mapping(row.get("schedule"), "NRIR-17 branch schedule")
            score_trace = _mapping(row.get("trace"), "NRIR-17 branch trace")
            scores = _list(score_trace.get("scores"), "NRIR-17 scores")
            selected = int(score_trace.get("selected_candidate_ordinal", -1))
            if (
                plan.get("schema_version") != "boundflow.objective_branch_plan_ir/v1"
                or task.get("schema_version") != "boundflow.objective_branch_task_ir/v1"
                or schedule.get("schema_version")
                != "boundflow.objective_branch_schedule_ir/v1"
                or len(_list(task.get("tasks"), "NRIR-17 branch tasks")) != 5
                or len(_list(schedule.get("actions"), "NRIR-17 branch actions")) != 5
                or not 0 <= selected < len(scores)
                or row.get("trace_hash") != canonical_hash(score_trace)
            ):
                raise ValueError("NRIR-17 branch IR evidence differs")
            chosen = _mapping(scores[selected], "NRIR-17 selected score")
            best = max(
                float(_mapping(item, "NRIR-17 score")["worst_child_lower"])
                for item in scores
            )
            if float(chosen["worst_child_lower"]) != best:
                raise ValueError("NRIR-17 objective selection is not worst-child max")
    elif branches:
        raise ValueError("NRIR-17 widest control carries objective evidence")


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("property_status") != "unknown"
    ):
        raise ValueError("NRIR-17 evidence header differs")
    source = _mapping(evidence.get("source"), "NRIR-17 source")
    if (
        source.get("model_sha256") != MODEL_SHA256
        or source.get("vnnlib_sha256") != VNNLIB_SHA256
        or source.get("intermediate_bounds_sha256") != INTERMEDIATE_BOUNDS_SHA256
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("abcrown_commit") != ABCROWN_COMMIT
    ):
        raise ValueError("NRIR-17 source identity differs")
    clauses = _mapping(evidence.get("clauses"), "NRIR-17 clauses")
    comparisons = _mapping(evidence.get("comparisons"), "NRIR-17 comparisons")
    if set(clauses) != {str(item) for item in HARD_CLAUSES} or set(comparisons) != set(
        clauses
    ):
        raise ValueError("NRIR-17 hard-clause coverage differs")
    for clause in HARD_CLAUSES:
        variants = _mapping(clauses[str(clause)], "NRIR-17 variants")
        if set(variants) != {"widest", "objective"}:
            raise ValueError("NRIR-17 variant coverage differs")
        _validate_semantic(
            _mapping(variants["widest"], "NRIR-17 widest"), objective=False
        )
        _validate_semantic(
            _mapping(variants["objective"], "NRIR-17 objective"), objective=True
        )
        widest = _mapping(
            _mapping(variants["widest"], "NRIR-17 widest")["summary"],
            "NRIR-17 widest summary",
        )
        objective = _mapping(
            _mapping(variants["objective"], "NRIR-17 objective")["summary"],
            "NRIR-17 objective summary",
        )
        comparison = _mapping(comparisons[str(clause)], "NRIR-17 comparison")
        delta = float(objective["leaf_worst_lower"]) - float(widest["leaf_worst_lower"])
        if (
            comparison.get("widest_leaf_worst_lower") != widest["leaf_worst_lower"]
            or comparison.get("objective_leaf_worst_lower")
            != objective["leaf_worst_lower"]
            or comparison.get("objective_minus_widest_leaf_worst") != delta
            or comparison.get("objective_not_weaker") is not (delta >= -1e-6)
        ):
            raise ValueError("NRIR-17 comparison differs")
    gates = _mapping(evidence.get("gates"), "NRIR-17 gates")
    if len(gates) != 6 or any(value is not True for value in gates.values()):
        raise ValueError("NRIR-17 gates differ")
    timings = _list(evidence.get("timing"), "NRIR-17 timing")
    if len(timings) != 6 or any(
        int(_mapping(item, "NRIR-17 timing row").get("elapsed_ns", 0)) <= 0
        for item in timings
    ):
        raise ValueError("NRIR-17 timing diagnostics differ")
    limitations = _list(evidence.get("limitations"), "NRIR-17 limitations")
    if len(limitations) != 5:
        raise ValueError("NRIR-17 limitations differ")


def _generate(args: argparse.Namespace) -> None:
    evidence = build_evidence(args)
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
        raise ValueError("NRIR-17 artifact manifest/header differs")
    stored = _mapping(artifact.get("evidence"), "NRIR-17 stored evidence")
    validate_evidence(stored)
    if manifest.get("evidence_hash") != canonical_hash(stored):
        raise ValueError("NRIR-17 stored evidence hash differs")
    context = _build_context(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
        local_artifact_dir=args.local_artifact_dir,
    )
    actual_clauses, _timing = _run_protocol(context)
    if stored.get("clauses") != actual_clauses:
        raise ValueError("NRIR-17 semantic replay differs")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-17 torch thread count must be positive")
    torch.set_num_threads(args.torch_threads)
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
