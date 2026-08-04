#!/usr/bin/env python3
"""Run or replay NRIR-34 ResNet nine-clause global-deadline integration."""

# pylint: disable=too-many-locals,too-many-statements,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

EVIDENCE_SCHEMA_VERSION = "boundflow.sibling-pack-objective-ancestral-full-query/v1"
MANIFEST_SCHEMA_VERSION = (
    "boundflow.sibling-pack-objective-ancestral-full-query-manifest/v1"
)
ARTIFACT_DIR = Path(
    "artifacts/sibling-packed-objective-ancestral-evaluator/"
    "vnncomp21-resnet2b-property0-full-query-cpu-integration-v1"
)
TORCH_THREADS = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "replay"))
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    return parser.parse_args()


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/objective_ancestral_sibling_pack.py",
        "boundflow/runtime/native_objective_ancestral_sibling_pack.py",
        "boundflow/runtime/native_objective_ancestral_sibling_pack_complete_query.py",
        "boundflow/runtime/complete_verifier_query.py",
        "boundflow/runtime/native_candidate_search.py",
        "boundflow/runtime/native_property_verdict.py",
        "scripts/run_sibling_packed_objective_ancestral_full_query.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _semantic_evidence(evidence: Mapping[str, Any]) -> dict[str, object]:
    return {
        "workload": evidence["workload"],
        "protocol": evidence["protocol"],
        "query_trace": evidence["query_trace"],
        "query_trace_hash": evidence["query_trace_hash"],
        "clause_summaries": evidence["clause_summaries"],
        "original_ordinal_accounting_valid": evidence[
            "original_ordinal_accounting_valid"
        ],
        "sound_status": evidence["sound_status"],
        "claim_boundary": evidence["claim_boundary"],
        "performance_claimed": False,
    }


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("source", {}).get("native_code_revision") != _code_revision()
        or evidence.get("performance_claimed") is not False
        or evidence.get("evidence_hash")
        != _canonical_hash(_semantic_evidence(evidence))
    ):
        raise ValueError("NRIR-34 full-query evidence envelope differs")
    protocol = evidence.get("protocol")
    trace = evidence.get("query_trace")
    summaries = evidence.get("clause_summaries")
    if (
        not isinstance(protocol, dict)
        or not isinstance(trace, dict)
        or not isinstance(summaries, list)
    ):
        raise TypeError("NRIR-34 full-query evidence structure differs")
    clause_count = int(protocol.get("original_clause_count", 0))
    completed = tuple(int(item["clause_index"]) for item in summaries)
    unresolved = tuple(int(item) for item in trace.get("unresolved_clause_indices", []))
    pending = tuple(int(item) for item in trace.get("pending_clause_indices", []))
    skipped = tuple(
        int(item) for item in trace.get("skipped_after_unsafe_clause_indices", [])
    )
    all_indices = set(range(clause_count))
    accounting_valid = (
        clause_count == 9
        and completed == tuple(range(len(completed)))
        and set(completed) | set(pending) | set(skipped) == all_indices
        and not (set(completed) & set(pending))
        and not (set(completed) & set(skipped))
        and not (set(pending) & set(skipped))
        and unresolved
        == tuple(
            int(item["clause_index"])
            for item in summaries
            if item["verdict_status"] == "unknown"
        )
    )
    if (
        trace.get("query_policy", {}).get("timeout_ns") != 60_000_000_000
        or trace.get("status") not in {"unknown", "unsafe", "verified"}
        or evidence.get("sound_status") != trace.get("status")
        or evidence.get("query_trace_hash") != _canonical_hash(trace)
        or evidence.get("original_ordinal_accounting_valid") is not accounting_valid
        or evidence.get("original_ordinal_accounting_valid") is not True
        or any(
            int(item["accepted_nodes"]) != 1 + 2 * int(item["sibling_group_count"])
            or item["all_groups_atomic_pairs"] is not True
            or item["performance_claimed"] is not False
            for item in summaries
        )
        or evidence.get("claim_boundary")
        != "sound_full_query_integration_only_no_property_or_performance_upgrade"
    ):
        raise ValueError("NRIR-34 full-query semantic evidence differs")


def _generate(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.complete_verifier_query import CompleteVerifierQueryPolicy
    from boundflow.runtime.native_objective_ancestral_sibling_pack_complete_query import (
        execute_native_objective_ancestral_sibling_pack_complete_query,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig

    torch.set_num_threads(args.torch_threads)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    _query, tensors, module, input_spec = _load_query_runtime(
        Path(str(workload["model"])),
        Path(str(workload["property"])),
        "cifar10_resnet:000",
    )
    search_policy, optimizer_policy = _policies()
    queue_config = NativeReluSplitBabConfig(
        max_nodes=31,
        max_depth=4,
        expansion_batch_size=1,
        max_eval_batch_size=2,
    )
    execution = execute_native_objective_ancestral_sibling_pack_complete_query(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id="nrir34:cifar10_resnet:000:property0:full-query",
        query_policy=CompleteVerifierQueryPolicy(timeout_ns=60_000_000_000),
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )
    summaries = []
    for clause in execution.clauses:
        queue = clause.queue
        frontier = queue.queue.trace.final_frontier_node_ids
        summaries.append(
            {
                "clause_index": clause.trace.clause_index,
                "verdict_status": clause.trace.status,
                "accepted_nodes": len(queue.queue.trace.evaluations),
                "maximum_depth": max(
                    item.node.depth for item in queue.queue.trace.evaluations
                ),
                "frontier_count": len(frontier),
                "sibling_group_count": len(queue.sibling_groups),
                "all_groups_atomic_pairs": all(
                    item.child_branch_values == (-1, 1) for item in queue.sibling_groups
                ),
                "fallback_reason": queue.trace.fallback_reason,
                "discarded_attempt_stage": queue.trace.discarded_attempt_stage,
                "queue_trace_hash": queue.queue.trace.stable_hash(),
                "aggregate_trace_hash": queue.trace.semantic_signature_hash,
                "performance_claimed": False,
            }
        )
    trace = execution.trace.to_dict()
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "source": {"native_code_revision": _code_revision()},
        "workload": _public_workload(workload),
        "protocol": {
            "original_clause_count": int(tensors.linear_spec_c.shape[1]),
            "original_clause_order": "ascending_index",
            "global_timeout_seconds": 60,
            "search_before_each_admitted_clause": True,
            "child_cap": 128,
            "search_budget": {"max_nodes": 31, "max_depth": 4},
            "sibling_group_size": 2,
            "torch_threads": args.torch_threads,
        },
        "query_trace": trace,
        "query_trace_hash": _canonical_hash(trace),
        "clause_summaries": summaries,
        "original_ordinal_accounting_valid": True,
        "sound_status": execution.trace.status,
        "claim_boundary": (
            "sound_full_query_integration_only_no_property_or_performance_upgrade"
        ),
        "performance_claimed": False,
    }
    evidence["evidence_hash"] = _canonical_hash(_semantic_evidence(evidence))
    validate_evidence(evidence)
    artifact_dir = args.artifact_dir.resolve()
    evidence_path = artifact_dir / "evidence.json"
    _write_json(evidence_path, evidence)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": {"evidence.json": _file_sha256(evidence_path)},
        "evidence_hash": evidence["evidence_hash"],
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "status": "ok",
                "query_status": execution.trace.status,
                "completed_clauses": len(execution.clauses),
                "pending_clauses": len(execution.trace.pending_clause_indices),
                "accepted_nodes": [item["accepted_nodes"] for item in summaries],
                "evidence_hash": evidence["evidence_hash"],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    evidence = _load_json(artifact_dir / "evidence.json")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("files", {}).get("evidence.json")
        != _file_sha256(artifact_dir / "evidence.json")
        or manifest.get("evidence_hash") != evidence.get("evidence_hash")
    ):
        raise ValueError("NRIR-34 full-query manifest differs")
    validate_evidence(evidence)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    if evidence.get("workload") != _public_workload(workload):
        raise ValueError("NRIR-34 full-query workload differs")
    print(
        _canonical_json(
            {
                "status": "ok",
                "query_status": evidence["sound_status"],
                "completed_clauses": len(evidence["clause_summaries"]),
                "evidence_hash": evidence["evidence_hash"],
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
