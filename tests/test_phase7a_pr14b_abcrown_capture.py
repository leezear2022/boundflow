"""Contracts for capturing one process-local external fixed replay."""

import torch

from boundflow.runtime.abcrown_adapter import ABCrownInitialCrownCapture


class _Box:  # pylint: disable=too-few-public-methods,invalid-name
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        self.x_L = lower
        self.x_U = upper


class _BoundedInput(torch.Tensor):
    ptb: _Box


def _bounded_input(lower: torch.Tensor, upper: torch.Tensor) -> _BoundedInput:
    value = ((lower + upper) / 2.0).as_subclass(_BoundedInput)
    value.ptb = _Box(lower, upper)
    return value


class _FakeBoundedModule:  # pylint: disable=too-few-public-methods
    def compute_bounds(  # pylint: disable=invalid-name
        self,
        *,
        x: tuple[torch.Tensor, ...],
        C: torch.Tensor,
        method: str,
        bound_upper: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return a deterministic fake lower/upper pair."""

        center = x[0].reshape(int(x[0].shape[0]), -1)
        lower = torch.bmm(C, center.unsqueeze(2)).squeeze(2)
        upper = lower + 1.0 if method.lower() == "crown" and bound_upper else None
        return lower, upper


def test_capture_owns_box_spec_and_replays_exact_call() -> None:
    """Captured tensors survive mutation and the closure preserves call kwargs."""

    lower = torch.tensor([[-1.0, 0.0]])
    upper = torch.tensor([[0.5, 2.0]])
    bounded_input = _bounded_input(lower, upper)
    linear_spec = torch.tensor([[[1.0, -2.0]]])
    module = _FakeBoundedModule()
    capture = ABCrownInitialCrownCapture(
        phase_resolver=lambda: "alpha_crown_initialization"
    )

    with capture.instrument(module):
        module.compute_bounds(
            x=(bounded_input,),
            C=linear_spec,
            method="CROWN",
            bound_upper=False,
        )

    assert "compute_bounds" not in module.__dict__
    assert capture.captured is not None
    lower.add_(10.0)
    linear_spec.zero_()
    assert torch.equal(capture.captured.input_lower, torch.tensor([[-1.0, 0.0]]))
    assert torch.equal(capture.captured.linear_spec_c, torch.tensor([[[1.0, -2.0]]]))
    assert capture.captured.bound_lower_requested
    assert not capture.captured.bound_upper_requested
    replay_lower, replay_upper = capture.captured.replay_external()
    assert replay_lower.shape == (1, 1)
    assert replay_upper is None


def test_capture_ignores_optimized_and_activation_phase_calls() -> None:
    """PR-14B must not silently expand into α/β or activation-BaB queries."""

    value = _bounded_input(torch.zeros(1, 2), torch.ones(1, 2))
    module = _FakeBoundedModule()
    phase = "alpha_crown_initialization"
    capture = ABCrownInitialCrownCapture(phase_resolver=lambda: phase)

    with capture.instrument(module):
        module.compute_bounds(
            x=(value,), C=torch.ones(1, 1, 2), method="CROWN-Optimized"
        )
        phase = "activation_bab_bound"
        module.compute_bounds(x=(value,), C=torch.ones(1, 1, 2), method="CROWN")

    assert capture.captured is None
