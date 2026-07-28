"""Generate or replay the deterministic IR-4D compiler-query artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import cast, Sequence

import torch

from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.ir.plan import PlanCost, StateAction, StateCandidate
from boundflow.planner.materialization import BoundMethod
from boundflow.runtime.bab_query import BoundQueryRequest, make_bound_query
from boundflow.runtime.bab_query_runtime import (
    CompilerSameSolverQueryRuntime,
    SameSolverRuntimeConfig,
)
from boundflow.runtime.bound_state_store import BoundRuntimeStateStore
from boundflow.runtime.compiler_query_runtime import (
    CompilerBoundQueryRequest,
    TypedCompilerQueryPayload,
    TypedCompilerQueryRuntime,
)
from scripts.run_plan_ir_v1_reference_artifact import (
    ReferenceSmokeWorkload,
    build_reference_smoke_workload,
)

ARTIFACT_SCHEMA = "boundflow.compiler-query-runtime-artifact/v2"


def _state_cost(latency: float) -> PlanCost:
    return PlanCost(
        predicted_latency_ms=latency,
        predicted_peak_bytes=0,
        compile_cost_ms=0.0,
        setup_cost_ms=0.0,
        confidence=1.0,
        risk_tags=("ir4d_artifact",),
    )


def _build_workload() -> ReferenceSmokeWorkload:
    workload = build_reference_smoke_workload()
    middle_op = workload.bound_module.graph.ops[1]
    candidates: list[StateCandidate] = []
    for index, value_id in enumerate(middle_op.outputs):
        value = next(
            item
            for item in workload.bound_module.graph.values
            if item.value_id == value_id
        )
        state_id = f"artifact-middle-output:{index}"
        if any(dim is None for dim in value.tensor_type.shape):
            raise ValueError("IR-4E artifact requires static state shapes")
        static_shape = cast(tuple[int, ...], value.tensor_type.shape)
        size_bytes = int(torch.empty(static_shape).numel() * 4)
        for action, latency in (
            (StateAction.REUSE, 0.0),
            (StateAction.CACHE, 0.1),
        ):
            candidates.append(
                StateCandidate(
                    candidate_id=f"state:{state_id}:{action.value}",
                    state_id=state_id,
                    source_value_id=value_id,
                    action=action,
                    state_version=value.state_version or "",
                    size_bytes=size_bytes,
                    static_legal=True,
                    rejection_reasons=(),
                    cost=_state_cost(latency),
                )
            )
    return replace(
        workload,
        template=replace(workload.template, state_candidates=tuple(candidates)),
    )


def _request(
    workload: ReferenceSmokeWorkload, query_id: str, sequence_number: int
) -> CompilerBoundQueryRequest:
    query, dynamic_payload = make_bound_query(
        module=workload.task_module,
        query_id=query_id,
        parent_query_id=None,
        sequence_number=sequence_number,
        example_idx=0,
        input_spec=workload.input_spec,
        linear_spec_c=None,
        split_by_relu_input={},
        warm_alpha_by_relu_input={},
        warm_beta_by_relu_input={},
        bound_method=BoundMethod.CROWN,
        execution_options={},
    )
    return CompilerBoundQueryRequest(
        query_request=BoundQueryRequest(query=query, payload=dynamic_payload),
        compiler_payload=TypedCompilerQueryPayload(
            legacy_task_module=workload.task_module,
            bound_module=workload.bound_module,
            template=workload.template,
            input_spec=workload.input_spec,
            relu_pre=workload.relu_pre,
        ),
    )


def build_artifact() -> dict[str, object]:
    """Run cache→reuse in one fresh process and return canonical evidence."""

    workload = _build_workload()
    compiler = TypedCompilerQueryRuntime(
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
        state_store=BoundRuntimeStateStore(),
    )
    runtime = CompilerSameSolverQueryRuntime(
        SameSolverRuntimeConfig(
            max_batch_size=2,
            memory_budget_bytes=1 << 20,
            minimum_fill_ratio=1.0,
        ),
        compiler,
    )
    paired_results = runtime.execute(
        (
            _request(workload, "artifact:cache", 7),
            _request(workload, "artifact:reuse", 2),
        ),
        now_us=100,
    )
    results = [result for _query_id, result in paired_results]
    runtime_audit = runtime.audit()
    compiler_audit = runtime_audit["compiler"]
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "bound_module_hash": workload.bound_module.stable_hash(),
        "template_hash": workload.template.stable_hash(
            bound_module=workload.bound_module
        ),
        "query_ids": [result.query_id for result in results],
        "sequence_numbers": [result.sequence_number for result in results],
        "result_hashes": [
            {
                "lower": tensor_content_hash(result.bounds.lower),
                "upper": tensor_content_hash(result.bounds.upper),
                "trace": result.trace.stable_hash(),
                "state_reused_tasks": sum(
                    event.backend_candidate_id == "state-reuse"
                    for event in result.trace.events
                ),
            }
            for result in results
        ],
        "audit": {
            "submitted_queries": runtime_audit["submitted_queries"],
            "emitted_queries": runtime_audit["emitted_queries"],
            "completed_queries": runtime_audit["completed_queries"],
            "emitted_batches": runtime_audit["emitted_batches"],
            "compatibility_dispatches": runtime_audit["compatibility_dispatches"],
            "legacy_executor_dispatches": runtime_audit["legacy_executor_dispatches"],
            "compiler": compiler_audit,
        },
    }


def _canonical(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a canonical artifact or verify it by fresh execution."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--out", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)

    actual = build_artifact()
    encoded = _canonical(actual) + "\n"
    if args.command == "generate":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
        return 0
    expected = args.artifact.read_text(encoding="utf-8")
    if expected != encoded:
        raise ValueError("IR-4E compiler-query artifact replay mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
