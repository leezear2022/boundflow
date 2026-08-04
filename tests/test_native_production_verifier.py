"""Production prepared verifier parity and fail-closed tests."""

# pylint: disable=missing-function-docstring,duplicate-code

from dataclasses import replace

import pytest
import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    execute_native_optimized_relu_split_bab,
)
from boundflow.runtime.native_production_verifier import (
    execute_native_production_relu_split_bab,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="production-verifier-toy",
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
        entry_task_id="production-verifier-toy",
        bindings={
            "params": {
                "W1": torch.tensor(
                    [[1.0, -0.5, 0.25], [-0.25, 0.75, 1.0], [0.5, 0.5, -1.0]]
                ),
                "b1": torch.tensor([0.1, -0.2, 0.05]),
                "W2": torch.tensor([[0.75, -1.0, 0.5], [-0.5, 0.25, 1.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6, -0.1]]),
        upper=torch.tensor([[0.7, 0.4, 0.9]]),
    )


def _policy() -> NativeAlphaBetaOptimizerPolicy:
    return NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)


def _config() -> NativeReluSplitBabConfig:
    return NativeReluSplitBabConfig(
        max_nodes=7,
        max_depth=2,
        expansion_batch_size=2,
        max_eval_batch_size=4,
        threshold=1e6,
    )


def test_production_queue_matches_audit_bounds_states_and_logical_queue() -> None:
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0, -1.0]])
    audit = execute_native_optimized_relu_split_bab(
        module,
        spec,
        linear_spec_C=objective,
        run_id="production-parity",
        config=_config(),
        optimizer_policy=_policy(),
    )
    production = execute_native_production_relu_split_bab(
        module,
        spec,
        linear_spec_C=objective,
        run_id="production-parity",
        config=_config(),
        optimizer_policy=_policy(),
    )

    assert production.trace.logical_queue_signature() == (
        audit.trace.logical_queue_signature()
    )
    assert production.trace.status == audit.trace.status
    assert production.trace.final_frontier_node_ids == (
        audit.trace.final_frontier_node_ids
    )
    assert all(
        actual.lower == expected.lower
        and actual.upper == expected.upper
        and actual.selected_state_hash == expected.selected_state_hash
        for actual, expected in zip(
            production.trace.evaluations, audit.trace.evaluations
        )
    )
    assert production.trace.audit_hash_chain_constructed is False
    assert production.trace.selected_native_reexecution is False
    assert all(
        len(batch.actions) == 4
        and batch.audit_hash_chain_constructed is False
        and batch.selected_native_reexecution is False
        for batch in production.trace.batches
    )


def test_production_queue_external_semantics_matches_audit() -> None:
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0, -1.0]])
    _interval_env, external = _forward_ibp_trace_mlp(module, spec)
    audit = execute_native_optimized_relu_split_bab(
        module,
        spec,
        linear_spec_C=objective,
        run_id="production-external-parity",
        config=_config(),
        optimizer_policy=_policy(),
        relu_pre_override=external,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
    )
    production = execute_native_production_relu_split_bab(
        module,
        spec,
        linear_spec_C=objective,
        run_id="production-external-parity",
        config=_config(),
        optimizer_policy=_policy(),
        relu_pre_override=external,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
    )

    assert production.trace.logical_queue_signature() == (
        audit.trace.logical_queue_signature()
    )
    assert [item.lower for item in production.trace.evaluations] == [
        item.lower for item in audit.trace.evaluations
    ]


def test_production_queue_identity_and_schedule_tampering_fail_closed() -> None:
    execution = execute_native_production_relu_split_bab(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0, -1.0]]),
        run_id="production-tamper",
        config=_config(),
        optimizer_policy=_policy(),
    )
    trace = execution.trace
    first = trace.batches[0]
    with pytest.raises(ValueError, match="Schedule IR"):
        replace(
            trace,
            batches=(
                replace(first, plan=replace(first.plan, objective_hash="f" * 64)),
                *trace.batches[1:],
            ),
        ).validate()
    actions = list(first.actions)
    actions[0] = replace(actions[0], task_id="wrong-task")
    with pytest.raises(ValueError, match="runtime/Schedule order"):
        replace(first, actions=tuple(actions)).validate()


def test_production_queue_rejects_intermediate_provenance_mismatch() -> None:
    with pytest.raises(ValueError, match="semantics/provenance differ"):
        execute_native_production_relu_split_bab(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[1.0, -1.0]]),
            run_id="production-invalid-provenance",
            config=_config(),
            optimizer_policy=_policy(),
            intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        )
