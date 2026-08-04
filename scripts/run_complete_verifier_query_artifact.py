#!/usr/bin/env python3
"""Generate or replay NRIR-14 complete verifier query evidence."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=duplicate-code,line-too-long
# pylint: disable=too-many-arguments,too-few-public-methods

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
from boundflow.runtime.complete_verifier_query import (
    COMPLETE_VERIFIER_QUERY_COMPILER_VERSION,
    COMPLETE_VERIFIER_QUERY_SCHEMA_VERSION,
    CompleteVerifierQueryPolicy,
    execute_complete_verifier_query,
)
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NATIVE_CANDIDATE_SEARCH_SCHEMA_VERSION,
    NativeProjectedGradientSearchPolicy,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NATIVE_REEXECUTION_ATOL,
    NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF,
)
from boundflow.runtime.native_property_verdict import (
    NATIVE_PROPERTY_VERDICT_SCHEMA_VERSION,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec
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

ARTIFACT_SCHEMA_VERSION = "boundflow.complete-verifier-query-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.complete-verifier-query-evidence/v1"
ARTIFACT_FILE = "complete_query.json"
MANIFEST_FILE = "manifest.json"
FIXED_QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir14-complete-query"
FIXED_SEARCH_POLICY = NativeProjectedGradientSearchPolicy(steps=4, step_size=0.002)
FIXED_QUEUE_CONFIG = NativeReluSplitBabConfig(
    max_nodes=1,
    max_depth=0,
    expansion_batch_size=1,
    max_eval_batch_size=1,
)
FIXED_OPTIMIZER_POLICY = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.05)


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


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _toy_module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="complete-query-artifact-toy",
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
        entry_task_id="complete-query-artifact-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0]]),
                "b1": torch.tensor([0.1]),
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


def _toy_query(
    *,
    query_id: str,
    objectives: torch.Tensor,
    thresholds: torch.Tensor,
    search_steps: int,
    query_policy: CompleteVerifierQueryPolicy = CompleteVerifierQueryPolicy(),
    clock_ns=None,
):
    kwargs = {} if clock_ns is None else {"clock_ns": clock_ns}
    return execute_complete_verifier_query(
        _toy_module(),
        _toy_spec(),
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id=query_id,
        query_policy=query_policy,
        search_policy=NativeProjectedGradientSearchPolicy(
            steps=search_steps,
            step_size=0.25,
        ),
        queue_config=NativeReluSplitBabConfig(
            max_nodes=1,
            max_depth=0,
            expansion_batch_size=1,
            max_eval_batch_size=1,
        ),
        optimizer_policy=NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1),
        **kwargs,
    )


class _FakeClock:
    def __init__(self, values: tuple[int, ...]):
        self._values = list(values)
        self._last = values[-1]

    def __call__(self) -> int:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def _serialize_query(execution) -> dict[str, object]:
    clauses: list[dict[str, object]] = []
    for clause in execution.clauses:
        search = clause.search.trace.to_dict()
        queue = clause.queue.trace.to_dict()
        verdict = clause.verdict.trace.to_dict()
        clauses.append(
            {
                "clause_trace": clause.trace.to_dict(),
                "search_trace": search,
                "search_trace_hash": clause.search.trace.stable_hash(),
                "best_input_hash": tensor_content_hash(clause.search.best_input),
                "queue_trace": queue,
                "queue_trace_hash": clause.queue.trace.stable_hash(),
                "verdict_trace": verdict,
                "verdict_trace_hash": clause.verdict.trace.stable_hash(
                    clause.queue.trace
                ),
                "counterexample_input_hash": (
                    None
                    if clause.verdict.counterexample_input is None
                    else tensor_content_hash(clause.verdict.counterexample_input)
                ),
            }
        )
    return {
        "query_trace": execution.trace.to_dict(),
        "query_trace_hash": execution.trace.stable_hash(),
        "clauses": clauses,
    }


def build_complete_verifier_query_evidence(
    *, model: Path, source_artifact_dir: Path
) -> dict[str, object]:
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("NRIR-14 model digest differs")
    source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = _payload_tensors(payload)
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    if tuple(op.op_type for op in module.get_entry_task().ops) != EXPECTED_PRIMAL_OPS:
        raise ValueError("NRIR-14 primal topology differs")
    fixed_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    fixed = execute_complete_verifier_query(
        module,
        fixed_spec,
        linear_spec_C=tensors["linear_spec_c"],
        thresholds=torch.zeros(9, dtype=tensors["linear_spec_c"].dtype),
        query_id=FIXED_QUERY_ID,
        query_policy=CompleteVerifierQueryPolicy(),
        search_policy=FIXED_SEARCH_POLICY,
        queue_config=FIXED_QUEUE_CONFIG,
        optimizer_policy=FIXED_OPTIMIZER_POLICY,
    )
    verified = _toy_query(
        query_id="complete-query-verified-toy",
        objectives=torch.tensor([[[-1.0], [-1.0]]]),
        thresholds=torch.tensor([-2.0, -2.0]),
        search_steps=1,
    )
    unsafe = _toy_query(
        query_id="complete-query-unsafe-toy",
        objectives=torch.tensor([[[-1.0], [-1.0], [-1.0]]]),
        thresholds=torch.tensor([-2.0, -0.5, -2.0]),
        search_steps=4,
    )
    unknown = _toy_query(
        query_id="complete-query-unknown-toy",
        objectives=torch.tensor([[[-1.0], [-1.0]]]),
        thresholds=torch.tensor([-2.0, -0.95]),
        search_steps=1,
    )
    deadline = _toy_query(
        query_id="complete-query-deadline-toy",
        objectives=torch.tensor([[[-1.0], [-1.0]]]),
        thresholds=torch.tensor([-2.0, -2.0]),
        search_steps=0,
        query_policy=CompleteVerifierQueryPolicy(timeout_ns=5),
        clock_ns=_FakeClock((0, 0, 10)),
    )

    fixed_case = _serialize_query(fixed)
    verified_case = _serialize_query(verified)
    unsafe_case = _serialize_query(unsafe)
    unknown_case = _serialize_query(unknown)
    deadline_case = _serialize_query(deadline)
    fixed_best_values = tuple(
        clause.search.trace.best_objective_value for clause in fixed.clauses
    )
    fixed_native_diffs = tuple(
        clause.queue.trace.native_stacks[0].selected_native_lower_max_abs_diff
        for clause in fixed.clauses
    )
    gates = {
        "multi_clause_conjunction_requires_every_clause_verified": bool(
            verified.trace.status == "verified"
            and len(verified.clauses) == 2
            and all(item.trace.status == "verified" for item in verified.clauses)
        ),
        "pgd_candidate_is_replayed_before_unsafe_short_circuit": bool(
            unsafe.trace.status == "unsafe"
            and unsafe.trace.unsafe_clause_index == 1
            and unsafe.trace.skipped_after_unsafe_clause_indices == (2,)
            and unsafe.clauses[-1].search.trace.counterexample_found
            and unsafe.clauses[-1].verdict.trace.counterexample is not None
        ),
        "attack_not_found_never_upgrades_unknown_to_verified": bool(
            unknown.trace.status == "unknown"
            and unknown.trace.unresolved_clause_indices == (1,)
            and not unknown.clauses[-1].search.trace.counterexample_found
        ),
        "cooperative_deadline_exposes_all_pending_clauses": bool(
            deadline.trace.status == "unknown"
            and deadline.trace.reason == "query_deadline_exhausted"
            and deadline.trace.pending_clause_indices == (0, 1)
            and not deadline.clauses
        ),
        "fixed_resnet_executes_nine_real_property_clauses": bool(
            len(fixed.clauses) == 9
            and tuple(item.trace.clause_index for item in fixed.clauses)
            == tuple(range(9))
            and fixed.trace.status == "unknown"
            and fixed.trace.unresolved_clause_indices == tuple(range(9))
        ),
        "fixed_resnet_pgd_search_finds_no_false_counterexample": bool(
            all(value > 0.0 for value in fixed_best_values)
            and not any(
                item.search.trace.counterexample_found for item in fixed.clauses
            )
        ),
        "scale_aware_execution_accepts_only_runtime_allclose_traces": bool(
            max(fixed_native_diffs) <= NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF
            and max(fixed_native_diffs) > NATIVE_REEXECUTION_ATOL
        ),
        "fixed_source_and_property_are_digest_bound": bool(
            source_manifest.get("schema_version")
            == "boundflow.native-real-network-ir-artifact/v1"
            and file_sha256(model) == MODEL_SHA256
        ),
        "all_complete_queries_are_correctness_only": bool(
            all(
                _mapping(case.get("query_trace"), "query trace").get(
                    "performance_claimed"
                )
                is False
                for case in (
                    fixed_case,
                    verified_case,
                    unsafe_case,
                    unknown_case,
                    deadline_case,
                )
            )
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-14 gates failed: {gates}")

    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "compiler_version": COMPLETE_VERIFIER_QUERY_COMPILER_VERSION,
        "performance_claimed": False,
        "property_status": "complete_query_control_validated_reduced",
        "claim_boundary": (
            "multi-clause conjunction control with deterministic candidate search, "
            "sound witness replay, unsafe short-circuit, and cooperative deadline; "
            "fixed ResNet remains unresolved and no performance is claimed"
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
            "best_objective_values": list(fixed_best_values),
            "native_lower_max_abs_diffs": list(fixed_native_diffs),
        },
        "toy_verified": verified_case,
        "toy_unsafe": unsafe_case,
        "toy_unknown": unknown_case,
        "toy_deadline": deadline_case,
        "gates": gates,
        "limitations": [
            "fixed ResNet scalarized native bounds remain too loose and all nine clauses are unknown",
            "candidate search is deterministic center-start single-restart PGD",
            "deadline enforcement is cooperative between stages and cannot preempt an active kernel",
            "single input box and conjunction aggregation only",
            "CPU correctness/control evidence only; no latency, memory, CUDA, or speedup claim",
        ],
    }
    validate_complete_verifier_query_evidence(evidence)
    return evidence


def _validate_query_case(
    value: object,
    *,
    expected_status: str,
    expected_reason: str,
) -> Mapping[str, Any]:
    case = _mapping(value, "NRIR-14 query case")
    query = _mapping(case.get("query_trace"), "NRIR-14 query trace")
    clauses = _list(case.get("clauses"), "NRIR-14 clauses")
    if (
        query.get("schema_version") != COMPLETE_VERIFIER_QUERY_SCHEMA_VERSION
        or query.get("compiler_version") != COMPLETE_VERIFIER_QUERY_COMPILER_VERSION
        or query.get("status") != expected_status
        or query.get("reason") != expected_reason
        or query.get("performance_claimed") is not False
        or case.get("query_trace_hash") != canonical_hash(query)
        or len(clauses) != len(_list(query.get("completed_clauses"), "completed"))
    ):
        raise ValueError("NRIR-14 query case header/hash differs")
    for index, value_clause in enumerate(clauses):
        clause = _mapping(value_clause, "NRIR-14 clause")
        clause_trace = _mapping(clause.get("clause_trace"), "clause trace")
        search = _mapping(clause.get("search_trace"), "search trace")
        queue = _mapping(clause.get("queue_trace"), "queue trace")
        verdict = _mapping(clause.get("verdict_trace"), "verdict trace")
        if (
            clause_trace.get("clause_index") != index
            or search.get("schema_version") != NATIVE_CANDIDATE_SEARCH_SCHEMA_VERSION
            or verdict.get("schema_version") != NATIVE_PROPERTY_VERDICT_SCHEMA_VERSION
            or clause.get("search_trace_hash") != canonical_hash(search)
            or clause.get("queue_trace_hash") != canonical_hash(queue)
            or clause.get("verdict_trace_hash") != canonical_hash(verdict)
            or clause_trace.get("search_trace_hash") != clause.get("search_trace_hash")
            or clause_trace.get("queue_trace_hash") != clause.get("queue_trace_hash")
            or clause_trace.get("verdict_trace_hash")
            != clause.get("verdict_trace_hash")
            or clause_trace.get("status") != verdict.get("status")
            or search.get("proof_claimed") is not False
        ):
            raise ValueError("NRIR-14 clause pipeline identity differs")
    return case


def validate_complete_verifier_query_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("compiler_version") != COMPLETE_VERIFIER_QUERY_COMPILER_VERSION
        or evidence.get("performance_claimed") is not False
        or evidence.get("property_status") != "complete_query_control_validated_reduced"
        or not isinstance(evidence.get("claim_boundary"), str)
    ):
        raise ValueError("NRIR-14 evidence header differs")
    source = _mapping(evidence.get("source"), "NRIR-14 source")
    if (
        source.get("model_sha256") != MODEL_SHA256
        or source.get("vnnlib_sha256") != VNNLIB_SHA256
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("abcrown_commit") != ABCROWN_COMMIT
    ):
        raise ValueError("NRIR-14 source identity differs")
    fixed = _validate_query_case(
        evidence.get("fixed_resnet"),
        expected_status="unknown",
        expected_reason="one_or_more_clauses_unresolved",
    )
    verified = _validate_query_case(
        evidence.get("toy_verified"),
        expected_status="verified",
        expected_reason="all_clauses_verified",
    )
    unsafe = _validate_query_case(
        evidence.get("toy_unsafe"),
        expected_status="unsafe",
        expected_reason="concrete_counterexample_clause",
    )
    unknown = _validate_query_case(
        evidence.get("toy_unknown"),
        expected_status="unknown",
        expected_reason="one_or_more_clauses_unresolved",
    )
    deadline = _validate_query_case(
        evidence.get("toy_deadline"),
        expected_status="unknown",
        expected_reason="query_deadline_exhausted",
    )
    fixed_query = _mapping(fixed["query_trace"], "fixed query")
    fixed_clauses = _list(fixed["clauses"], "fixed clauses")
    best_values = _list(fixed.get("best_objective_values"), "fixed best values")
    native_diffs = _list(fixed.get("native_lower_max_abs_diffs"), "native diffs")
    if (
        len(fixed_clauses) != 9
        or fixed_query.get("unresolved_clause_indices") != list(range(9))
        or len(best_values) != 9
        or not all(float(value) > 0.0 for value in best_values)
        or len(native_diffs) != 9
        or max(float(value) for value in native_diffs)
        > NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF
        or max(float(value) for value in native_diffs) <= NATIVE_REEXECUTION_ATOL
    ):
        raise ValueError("NRIR-14 fixed ResNet clause boundary differs")
    verified_query = _mapping(verified["query_trace"], "verified query")
    if (
        len(_list(verified["clauses"], "verified clauses")) != 2
        or verified_query.get("unresolved_clause_indices") != []
        or verified_query.get("pending_clause_indices") != []
    ):
        raise ValueError("NRIR-14 verified aggregation differs")
    unsafe_query = _mapping(unsafe["query_trace"], "unsafe query")
    unsafe_clauses = _list(unsafe["clauses"], "unsafe clauses")
    unsafe_last = _mapping(unsafe_clauses[-1], "unsafe last clause")
    if (
        unsafe_query.get("unsafe_clause_index") != 1
        or unsafe_query.get("skipped_after_unsafe_clause_indices") != [2]
        or unsafe_last.get("counterexample_input_hash") is None
    ):
        raise ValueError("NRIR-14 unsafe short-circuit differs")
    unknown_query = _mapping(unknown["query_trace"], "unknown query")
    if unknown_query.get("unresolved_clause_indices") != [1]:
        raise ValueError("NRIR-14 attack-not-proof boundary differs")
    deadline_query = _mapping(deadline["query_trace"], "deadline query")
    if (
        deadline_query.get("pending_clause_indices") != [0, 1]
        or deadline.get("clauses") != []
        or deadline_query.get("deadline_is_cooperative") is not True
    ):
        raise ValueError("NRIR-14 cooperative deadline differs")
    gates = _mapping(evidence.get("gates"), "NRIR-14 gates")
    expected_gates = {
        "multi_clause_conjunction_requires_every_clause_verified",
        "pgd_candidate_is_replayed_before_unsafe_short_circuit",
        "attack_not_found_never_upgrades_unknown_to_verified",
        "cooperative_deadline_exposes_all_pending_clauses",
        "fixed_resnet_executes_nine_real_property_clauses",
        "fixed_resnet_pgd_search_finds_no_false_counterexample",
        "scale_aware_execution_accepts_only_runtime_allclose_traces",
        "fixed_source_and_property_are_digest_bound",
        "all_complete_queries_are_correctness_only",
    }
    if set(gates) != expected_gates or any(
        value is not True for value in gates.values()
    ):
        raise ValueError("NRIR-14 gates differ")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 5:
        raise ValueError("NRIR-14 limitations differ")


def _generate(args: argparse.Namespace) -> None:
    evidence = build_complete_verifier_query_evidence(
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
        raise ValueError("NRIR-14 artifact manifest/header differs")
    stored = _mapping(artifact.get("evidence"), "NRIR-14 stored evidence")
    validate_complete_verifier_query_evidence(stored)
    if manifest.get("evidence_hash") != canonical_hash(stored):
        raise ValueError("NRIR-14 stored evidence hash differs")
    actual = build_complete_verifier_query_evidence(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
    )
    if stored != actual:
        raise ValueError("NRIR-14 replay differs from frozen evidence")
    print(_canonical_json({"status": "ok", "evidence_hash": canonical_hash(actual)}))


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
