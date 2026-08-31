"""S4-1D/2 active-state evaluator and 10/9 trajectory gates."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals
# pylint: disable=import-error,import-outside-toplevel,duplicate-code

from __future__ import annotations

import torch

from boundflow.runtime.asplos27_s4_all_state_evaluator import (
    PreparedS4AllStateEvaluatorV1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    _evaluate_full_region,
)
from tests.test_asplos27_s4_gradient_phase import _fixture

SITES = (17, 19, 23, 25, 28, 31)


def _dense_mutable_state(executor):  # type: ignore[no-untyped-def]
    tensors = list(executor.tensors)
    by_name = {
        spec.name: ordinal for ordinal, spec in enumerate(executor.plan.tensor_specs)
    }
    alphas = []
    for site in SITES:
        ordinal = by_name[f"relu/{site}/alpha"]
        tensors[ordinal] = tensors[ordinal].detach().clone().requires_grad_(True)
        alphas.append(tensors[ordinal])
    beta_ordinal = by_name["relu/31/beta"]
    tensors[beta_ordinal] = tensors[beta_ordinal].detach().clone().requires_grad_(True)
    return tuple(tensors), alphas, tensors[beta_ordinal]


def test_s4_all_state_ten_nine_trajectory_matches_dense_autograd() -> None:
    _device, stream, candidate_executor, buffers, _ = _fixture()
    evaluator = PreparedS4AllStateEvaluatorV1(
        candidate_executor,
        buffers,
        exact_call_id="s4-all-state-trajectory",
        stream=stream,
    )
    resources = buffers._resources
    assert resources is not None
    candidate_parameters = list(resources._parameters)
    candidate_optimizer = torch.optim.Adam(
        (
            {"params": candidate_parameters[:6], "lr": 0.01},
            {"params": [candidate_parameters[6]], "lr": 0.05},
        )
    )
    candidate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        candidate_optimizer, gamma=0.98
    )

    _, _, dense_executor, _, _ = _fixture()
    dense_tensors, dense_alpha, dense_beta = _dense_mutable_state(dense_executor)
    dense_parameters = [*dense_alpha, dense_beta]
    dense_optimizer = torch.optim.Adam(
        (
            {"params": dense_alpha, "lr": 0.01},
            {"params": [dense_beta], "lr": 0.05},
        )
    )
    dense_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        dense_optimizer, gamma=0.98
    )

    for ordinal in range(10):
        with torch.cuda.stream(stream):
            candidate = evaluator.evaluate(ordinal, terminal=ordinal == 9)
        dense_lower = _evaluate_full_region(dense_executor.plan, dense_tensors).reshape(
            6
        )
        dense_gradients = torch.autograd.grad(-dense_lower.sum(), dense_parameters)
        stream.synchronize()
        assert torch.allclose(candidate.lower, dense_lower, atol=2e-4, rtol=2e-4)
        assert torch.equal(torch.sign(candidate.lower), torch.sign(dense_lower))
        expected_gradients = tuple(
            gradient[0, 0] for gradient in dense_gradients[:6]
        ) + (dense_gradients[6],)
        for observed, expected in zip(candidate.gradients, expected_gradients):
            assert torch.allclose(observed, expected, atol=2e-5, rtol=2e-5)
            assert torch.equal(torch.sign(observed), torch.sign(expected))

        if ordinal < 9:
            with torch.cuda.stream(stream):
                candidate_optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(
                    candidate_parameters, candidate.gradients
                ):
                    parameter.grad = gradient
                candidate_optimizer.step()
                with torch.no_grad():
                    for parameter in candidate_parameters[:6]:
                        parameter.clamp_(0.0, 1.0)
                    candidate_parameters[6].clamp_(min=0.0)
                candidate_scheduler.step()

            dense_optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(dense_parameters, dense_gradients):
                parameter.grad = gradient
            dense_optimizer.step()
            with torch.no_grad():
                for parameter in dense_alpha:
                    parameter.clamp_(0.0, 1.0)
                dense_beta.clamp_(min=0.0)
            dense_scheduler.step()
            stream.synchronize()
            for observed, expected in zip(candidate_parameters[:6], dense_alpha):
                assert torch.allclose(observed, expected[0, 0], atol=2e-5, rtol=2e-5)
            assert torch.allclose(
                candidate_parameters[6], dense_beta, atol=2e-5, rtol=2e-5
            )
        else:
            assert candidate.terminal_lease is not None
            terminal_las = candidate.terminal_lease.consume(evaluation_generation=10)
            assert len(terminal_las) == 6
            assert all(torch.isfinite(value).all() for value in terminal_las)


def test_s4_all_state_driver_has_ten_nine_counters() -> None:
    from boundflow.runtime.asplos27_s4_optimizer_driver import (
        execute_s4_optimizer_v1,
    )

    _, stream, executor, buffers, _ = _fixture()
    evaluator = PreparedS4AllStateEvaluatorV1(
        executor,
        buffers,
        exact_call_id="s4-all-state-driver",
        stream=stream,
    )
    result = execute_s4_optimizer_v1(evaluator)
    assert result.evaluation_count == 10
    assert result.optimizer_mutation_count == 9
    assert result.scheduler_call_count == 10
    assert result.value_graph_submission_count == 10
    assert result.compact_coefficient_launch_count == 180
    assert result.fallback_count == 0
    assert result.performance_claimed is False
