"""Complete multi-clause verifier query and candidate-search tests."""

# pylint: disable=missing-function-docstring,redefined-outer-name,duplicate-code
# pylint: disable=too-few-public-methods

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.complete_verifier_query import (
    CompleteVerifierQueryPolicy,
    execute_complete_verifier_query,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
    search_native_box_counterexample,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import (
    InputSpec,
    execute_task_module_concrete,
)


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="complete-query-toy",
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
        entry_task_id="complete-query-toy",
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


def _queue_config() -> NativeReluSplitBabConfig:
    return NativeReluSplitBabConfig(
        max_nodes=1,
        max_depth=0,
        expansion_batch_size=1,
        max_eval_batch_size=1,
    )


def _execute(
    objectives: torch.Tensor,
    thresholds: torch.Tensor,
    *,
    search_steps: int,
    timeout_ns: int | None = None,
    clock_ns=None,
):
    kwargs = {}
    if clock_ns is not None:
        kwargs["clock_ns"] = clock_ns
    return execute_complete_verifier_query(
        _module(),
        _spec(),
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="complete-query-toy",
        query_policy=CompleteVerifierQueryPolicy(timeout_ns=timeout_ns),
        search_policy=NativeProjectedGradientSearchPolicy(
            steps=search_steps,
            step_size=0.25,
        ),
        queue_config=_queue_config(),
        optimizer_policy=_optimizer_policy(),
        **kwargs,
    )


def test_concrete_executor_preserves_autograd_for_candidate_search() -> None:
    candidate = torch.tensor([[0.0]], requires_grad=True)
    execution = execute_task_module_concrete(
        _module(),
        candidate,
        preserve_gradients=True,
    )
    gradient = torch.autograd.grad(execution.output.sum(), candidate)[0]

    assert execution.gradients_preserved is True
    assert execution.output.requires_grad is True
    assert torch.equal(execution.output, torch.tensor([[0.1]]))
    assert torch.equal(gradient, torch.tensor([[1.0]]))


def test_projected_gradient_search_finds_but_does_not_prove() -> None:
    search = search_native_box_counterexample(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[-1.0]]),
        threshold=-0.5,
        policy=NativeProjectedGradientSearchPolicy(steps=4, step_size=0.25),
    )

    assert search.trace.counterexample_found is True
    assert search.trace.early_stopped is True
    assert search.trace.steps_executed == 2
    assert search.trace.objective_values == pytest.approx((-0.1, -0.35, -0.6))
    assert search.trace.gradient_l1_values == pytest.approx((1.0, 1.0))
    assert torch.equal(search.best_input, torch.tensor([[0.5]]))
    assert search.trace.to_dict()["proof_claimed"] is False


def test_multi_clause_verified_requires_every_clause_closed() -> None:
    result = _execute(
        torch.tensor([[[-1.0], [-1.0]]]),
        torch.tensor([-2.0, -2.0]),
        search_steps=1,
    )

    assert result.trace.status == "verified"
    assert result.trace.reason == "all_clauses_verified"
    assert tuple(item.status for item in result.trace.completed_clauses) == (
        "verified",
        "verified",
    )
    assert not result.trace.unresolved_clause_indices
    assert not result.trace.pending_clause_indices


def test_candidate_search_drives_unsafe_short_circuit() -> None:
    result = _execute(
        torch.tensor([[[-1.0], [-1.0], [-1.0]]]),
        torch.tensor([-2.0, -0.5, -2.0]),
        search_steps=4,
    )

    assert result.trace.status == "unsafe"
    assert result.trace.reason == "concrete_counterexample_clause"
    assert result.trace.unsafe_clause_index == 1
    assert result.trace.skipped_after_unsafe_clause_indices == (2,)
    assert tuple(item.status for item in result.trace.completed_clauses) == (
        "verified",
        "unsafe",
    )
    unsafe = result.clauses[-1]
    assert unsafe.search.trace.counterexample_found is True
    assert unsafe.verdict.trace.counterexample is not None
    assert unsafe.verdict.trace.counterexample.objective_value == pytest.approx(-0.6)


def test_attack_not_found_keeps_unresolved_clause_unknown() -> None:
    result = _execute(
        torch.tensor([[[-1.0], [-1.0]]]),
        torch.tensor([-2.0, -0.95]),
        search_steps=1,
    )

    assert result.trace.status == "unknown"
    assert result.trace.reason == "one_or_more_clauses_unresolved"
    assert result.trace.unresolved_clause_indices == (1,)
    assert tuple(item.status for item in result.trace.completed_clauses) == (
        "verified",
        "unknown",
    )
    assert result.clauses[1].search.trace.counterexample_found is False
    assert result.clauses[1].search.trace.best_objective_value == pytest.approx(-0.35)


class _FakeClock:
    def __init__(self, values: tuple[int, ...]):
        self._values = list(values)
        self._last = values[-1]

    def __call__(self) -> int:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def test_cooperative_deadline_returns_pending_unknown() -> None:
    result = _execute(
        torch.tensor([[[1.0], [-1.0]]]),
        torch.tensor([0.0, -2.0]),
        search_steps=0,
        timeout_ns=5,
        clock_ns=_FakeClock((0, 0, 10)),
    )

    assert result.trace.status == "unknown"
    assert result.trace.reason == "query_deadline_exhausted"
    assert result.trace.elapsed_ns == 10
    assert not result.trace.completed_clauses
    assert result.trace.pending_clause_indices == (0, 1)


def test_query_trace_and_objective_tampering_fail_closed() -> None:
    result = _execute(
        torch.tensor([[[-1.0], [-1.0]]]),
        torch.tensor([-2.0, -2.0]),
        search_steps=1,
    )
    with pytest.raises(ValueError, match="verified query"):
        replace(
            result.trace,
            completed_clauses=result.trace.completed_clauses[:1],
            pending_clause_indices=(1,),
        ).validate()
    with pytest.raises(ValueError, match="coverage"):
        result.validate_against(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[[1.0], [-1.0]]]),
        )


def test_complete_query_binds_external_intermediate_semantics() -> None:
    module = _module()
    spec = _spec()
    _interval_env, external = _forward_ibp_trace_mlp(module, spec)
    result = execute_complete_verifier_query(
        module,
        spec,
        linear_spec_C=torch.tensor([[[-1.0]]]),
        thresholds=torch.tensor([-2.0]),
        query_id="complete-query-external",
        query_policy=CompleteVerifierQueryPolicy(),
        search_policy=NativeProjectedGradientSearchPolicy(steps=1, step_size=0.25),
        queue_config=_queue_config(),
        optimizer_policy=NativeAlphaBetaOptimizerPolicy(
            steps=1,
            lr=0.1,
            alpha_initialization_mode="adaptive",
        ),
        relu_pre_override=external,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
    )

    assert result.trace.status == "verified"
    assert len(result.clauses) == 1
    assert result.clauses[0].queue.trace.native_stack_count == 1


def test_complete_query_external_provenance_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="semantics/provenance differ"):
        execute_complete_verifier_query(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[[-1.0]]]),
            thresholds=torch.tensor([-2.0]),
            query_id="complete-query-external-missing",
            query_policy=CompleteVerifierQueryPolicy(),
            search_policy=NativeProjectedGradientSearchPolicy(steps=1, step_size=0.25),
            queue_config=_queue_config(),
            optimizer_policy=_optimizer_policy(),
            intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        )
