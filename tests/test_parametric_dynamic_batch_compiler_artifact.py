"""NRIR-28 artifact compiler-evidence replay tests."""

# pylint: disable=missing-function-docstring,duplicate-code

import copy

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
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec
from scripts.run_parametric_dynamic_batch_compiler_artifact import (
    MODE_ORDERS,
    _validate_compiler_payload,
)


def _execution():
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="parametric-artifact-toy",
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
        entry_task_id="parametric-artifact-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [-0.25, 0.75]]),
                "b1": torch.tensor([0.1, -0.2]),
                "W2": torch.tensor([[0.75, -1.0], [-0.5, 0.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )
    return execute_native_parametric_production_complete_verifier_query(
        module,
        spec,
        linear_spec_C=torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]]),
        thresholds=torch.tensor([-1e6, -1e6]),
        query_id="parametric-artifact-toy",
        query_policy=CompleteVerifierQueryPolicy(),
        search_policy=NativeProjectedGradientSearchPolicy(steps=1, step_size=0.01),
        queue_config=NativeReluSplitBabConfig(
            max_nodes=3,
            max_depth=1,
            expansion_batch_size=1,
            max_eval_batch_size=2,
        ),
        optimizer_policy=NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1),
    )


def _compiler_payload() -> dict[str, object]:
    execution = _execution()
    return {
        "cache": execution.compiler_cache_trace.to_dict(),
        "batches": [
            batch.to_dict()
            for clause in execution.clauses
            for batch in clause.queue.compiler_batches
        ],
    }


def test_parametric_artifact_recomputes_compiler_ir_digests() -> None:
    payload = _compiler_payload()
    _validate_compiler_payload(payload)

    tampered = copy.deepcopy(payload)
    tampered["batches"][0]["instance"]["objective_hash"] = "f" * 64
    with pytest.raises(ValueError, match="event/instance linkage differs"):
        _validate_compiler_payload(tampered)


def test_parametric_artifact_protocol_alternates_both_modes() -> None:
    assert len(MODE_ORDERS) == 3
    assert all(
        set(order) == {"production_v1", "parametric_v2"} for order in MODE_ORDERS
    )
    assert MODE_ORDERS[0] != MODE_ORDERS[1]
