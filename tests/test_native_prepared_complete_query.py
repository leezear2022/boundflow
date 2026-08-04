"""Prepared production complete-query root-path contracts."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.complete_verifier_query import (
    CompleteVerifierQueryPolicy,
    execute_complete_verifier_query,
)
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
)
from boundflow.runtime.native_prepared_complete_query import (
    execute_native_prepared_complete_query,
    prepare_native_root_complete_query,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="prepared-complete-query-toy",
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
        entry_task_id="prepared-complete-query-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0]]),
                "b1": torch.tensor([0.1]),
                "W2": torch.tensor([[1.0]]),
                "b2": torch.tensor([0.0]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-1.0]]),
        upper=torch.tensor([[1.0]]),
    )


def _optimizer_policy() -> NativeAlphaBetaOptimizerPolicy:
    return NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)


def _search_policy(*, steps: int) -> NativeProjectedGradientSearchPolicy:
    return NativeProjectedGradientSearchPolicy(steps=steps, step_size=0.25)


def _queue_config() -> NativeReluSplitBabConfig:
    return NativeReluSplitBabConfig(
        max_nodes=1,
        max_depth=0,
        expansion_batch_size=1,
        max_eval_batch_size=1,
    )


def test_prepared_root_query_matches_audit_status_and_bounds() -> None:
    module = _module()
    spec = _spec()
    objectives = torch.tensor([[[-1.0], [-1.0]]])
    thresholds = torch.tensor([-2.0, -0.95])
    search = _search_policy(steps=1)
    optimizer = _optimizer_policy()
    prepared = prepare_native_root_complete_query(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="prepared-complete-query-toy",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    production = execute_native_prepared_complete_query(
        prepared,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
    )
    audit = execute_complete_verifier_query(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="audit-complete-query-toy",
        query_policy=CompleteVerifierQueryPolicy(),
        search_policy=search,
        queue_config=_queue_config(),
        optimizer_policy=optimizer,
    )

    assert production.trace.status == audit.trace.status == "unknown"
    assert production.trace.unresolved_clause_indices == (1,)
    assert tuple(item.status for item in production.trace.completed_clauses) == (
        "verified",
        "unknown",
    )
    assert tuple(item.lower for item in production.trace.completed_clauses) == tuple(
        item.queue.trace.evaluations[0].lower for item in audit.clauses
    )
    assert all(
        not item.audit_hash_chain_constructed and not item.selected_native_reexecution
        for item in production.trace.completed_clauses
    )


def test_prepared_root_query_replays_unsafe_and_short_circuits() -> None:
    module = _module()
    spec = _spec()
    objectives = torch.tensor([[[-1.0], [-1.0], [-1.0]]])
    thresholds = torch.tensor([-2.0, -0.5, -2.0])
    prepared = prepare_native_root_complete_query(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="prepared-complete-query-unsafe",
        search_policy=_search_policy(steps=4),
        optimizer_policy=_optimizer_policy(),
    )
    production = execute_native_prepared_complete_query(
        prepared,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
    )

    assert production.trace.status == "unsafe"
    assert production.trace.unsafe_clause_index == 1
    assert production.trace.skipped_after_unsafe_clause_indices == (2,)
    assert production.trace.completed_clauses[-1].counterexample_input_hash is not None
    assert len(production.searches) == len(production.optimizer_results) == 2


def test_prepared_root_query_identity_and_trace_tampering_fail_closed() -> None:
    module = _module()
    spec = _spec()
    objectives = torch.tensor([[[-1.0], [-1.0]]])
    thresholds = torch.tensor([-2.0, -0.95])
    prepared = prepare_native_root_complete_query(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="prepared-complete-query-drift",
        search_policy=_search_policy(steps=1),
        optimizer_policy=_optimizer_policy(),
    )
    with pytest.raises(ValueError, match="exact identity differs"):
        execute_native_prepared_complete_query(
            prepared,
            module,
            spec,
            linear_spec_C=-objectives,
            thresholds=thresholds,
        )
    execution = execute_native_prepared_complete_query(
        prepared,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
    )
    first = execution.trace.completed_clauses[0]
    with pytest.raises(ValueError, match="clause status"):
        replace(first, status="unknown").validate()
    with pytest.raises(ValueError, match="query trace"):
        replace(execution.trace, unresolved_clause_indices=(0, 1)).validate()
