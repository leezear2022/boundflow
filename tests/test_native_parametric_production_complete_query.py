"""Parametric complete-query parity and query-scoped cache tests."""

# pylint: disable=missing-function-docstring,duplicate-code

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
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
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="parametric-query-toy",
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
        entry_task_id="parametric-query-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [-0.25, 0.75]]),
                "b1": torch.tensor([0.1, -0.2]),
                "W2": torch.tensor([[0.75, -1.0], [-0.5, 0.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )


def _policies():
    return (
        CompleteVerifierQueryPolicy(),
        NativeProjectedGradientSearchPolicy(steps=2, step_size=0.01),
        NativeReluSplitBabConfig(
            max_nodes=7,
            max_depth=2,
            expansion_batch_size=2,
            max_eval_batch_size=4,
        ),
        NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1),
    )


def test_parametric_complete_query_matches_v1_and_compiles_once() -> None:
    module = _module()
    spec = _spec()
    objectives = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    thresholds = torch.tensor([-1e6, -1e6])
    query_policy, search_policy, queue_config, optimizer_policy = _policies()
    baseline = execute_native_production_complete_verifier_query(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="parametric-complete-query",
        query_policy=query_policy,
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )
    parametric = execute_native_parametric_production_complete_verifier_query(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="parametric-complete-query",
        query_policy=query_policy,
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )

    assert parametric.trace.status == baseline.trace.status
    assert parametric.trace.reason == baseline.trace.reason
    assert parametric.trace.unresolved_clause_indices == (
        baseline.trace.unresolved_clause_indices
    )
    assert (
        parametric.trace.pending_clause_indices == baseline.trace.pending_clause_indices
    )
    assert [clause.trace.status for clause in parametric.clauses] == [
        clause.trace.status for clause in baseline.clauses
    ]
    assert [
        (
            item.lower,
            item.upper,
            item.selected_state_hash,
        )
        for clause in parametric.clauses
        for item in clause.queue.trace.evaluations
    ] == [
        (
            item.lower,
            item.upper,
            item.selected_state_hash,
        )
        for clause in baseline.clauses
        for item in clause.queue.trace.evaluations
    ]
    cache = parametric.compiler_cache_trace.to_dict()
    assert cache["template_count"] == 1
    assert cache["instance_count"] == 2
    assert cache["miss_count"] == 1
    assert cache["hit_count"] == 1


def test_parametric_complete_query_cache_trace_tamper_fails_closed() -> None:
    query_policy, search_policy, queue_config, optimizer_policy = _policies()
    execution = execute_native_parametric_production_complete_verifier_query(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]]),
        thresholds=torch.tensor([-1e6, -1e6]),
        query_id="parametric-complete-query-tamper",
        query_policy=query_policy,
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )
    events = list(execution.compiler_cache_trace.events)
    events[1] = replace(events[1], template_hash="f" * 64)

    with pytest.raises(ValueError, match="cache trace target differs"):
        replace(
            execution.compiler_cache_trace,
            events=tuple(events),
        ).validate()
