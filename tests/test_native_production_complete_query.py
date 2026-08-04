"""Production complete-query parity and disclosure tests."""

# pylint: disable=missing-function-docstring,duplicate-code

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
from boundflow.runtime.native_production_complete_query import (
    execute_native_production_complete_verifier_query,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="production-query-toy",
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
        entry_task_id="production-query-toy",
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


def test_production_complete_query_matches_audit_verdict_and_clause_bounds() -> None:
    module = _module()
    spec = _spec()
    objectives = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    thresholds = torch.tensor([-1e6, -1e6])
    query_policy, search_policy, queue_config, optimizer_policy = _policies()
    audit = execute_complete_verifier_query(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="production-complete-query",
        query_policy=query_policy,
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )
    production = execute_native_production_complete_verifier_query(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="production-complete-query",
        query_policy=query_policy,
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )

    assert production.trace.status == audit.trace.status == "verified"
    assert production.trace.execution_mode == "production_prepared"
    assert all(
        clause.trace.queue_execution_mode == "production_prepared"
        and clause.queue.trace.audit_hash_chain_constructed is False
        and clause.queue.trace.selected_native_reexecution is False
        for clause in production.clauses
    )
    assert [
        (item.lower, item.upper)
        for clause in production.clauses
        for item in clause.queue.trace.evaluations
    ] == [
        (item.lower, item.upper)
        for clause in audit.clauses
        for item in clause.queue.trace.evaluations
    ]


def test_audit_complete_query_payload_keeps_historical_default_shape() -> None:
    module = _module()
    query_policy, search_policy, queue_config, optimizer_policy = _policies()
    audit = execute_complete_verifier_query(
        module,
        _spec(),
        linear_spec_C=torch.tensor([[[1.0, -1.0]]]),
        thresholds=torch.tensor([-1e6]),
        query_id="audit-default-payload",
        query_policy=query_policy,
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )

    payload = audit.trace.to_dict()
    assert "execution_mode" not in payload
    assert "queue_execution_mode" not in payload["completed_clauses"][0]
