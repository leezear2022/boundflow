"""Contracts for capturing one process-local external fixed replay."""

import pytest
import torch

from boundflow.domains.interval import IntervalState
from boundflow.runtime.abcrown_adapter import (
    ABCrownInitialCrownCapture,
    CapturedIntermediateBound,
    bind_intermediate_bounds,
    bind_captured_intermediate_bounds,
    deserialize_intermediate_bounds,
    intermediate_bounds_sha256,
    serialize_intermediate_bounds,
)


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
    def __init__(self) -> None:
        preactivation = type("ExternalPreactivation", (), {})()
        preactivation.name = "/pre"
        preactivation.lower = torch.tensor([[-0.25, 0.1]])
        preactivation.upper = torch.tensor([[0.75, 0.2]])
        relu = type("BoundRelu", (), {})()
        relu.name = "/relu"
        relu.inputs = [preactivation]
        self._nodes = [relu]

    def nodes(self):
        return self._nodes

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
    assert len(capture.captured.intermediate_bounds) == 1
    assert capture.captured.relu_lower_slope_policy == "adaptive"
    assert len(capture.captured.intermediate_bounds_hash) == 64
    local = {"h1": IntervalState(lower=torch.zeros(1, 2), upper=torch.ones(1, 2))}
    rebound = bind_captured_intermediate_bounds(capture.captured, local)
    torch.testing.assert_close(rebound["h1"].lower, torch.tensor([[-0.25, 0.1]]))
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


def test_capture_intermediate_binding_fails_closed_on_topology_or_shape_drift() -> None:
    value = _bounded_input(torch.zeros(1, 2), torch.ones(1, 2))
    capture = ABCrownInitialCrownCapture(
        phase_resolver=lambda: "alpha_crown_initialization"
    )
    module = _FakeBoundedModule()
    with capture.instrument(module):
        module.compute_bounds(x=(value,), C=torch.ones(1, 1, 2), method="CROWN")
    assert capture.captured is not None

    with pytest.raises(ValueError, match="count mismatch"):
        bind_captured_intermediate_bounds(capture.captured, {})
    with pytest.raises(ValueError, match="shape mismatch"):
        bind_captured_intermediate_bounds(
            capture.captured,
            {"h1": IntervalState(lower=torch.zeros(1, 3), upper=torch.ones(1, 3))},
        )


def test_intermediate_bounds_portable_payload_round_trip(tmp_path) -> None:
    """Frozen external intervals remain loadable with PyTorch's safe loader."""

    bounds = (
        CapturedIntermediateBound(
            ordinal=0,
            external_relu_name="/relu",
            external_preactivation_name="/pre",
            lower=torch.tensor([[-0.25, 0.1]]),
            upper=torch.tensor([[0.75, 0.2]]),
        ),
    )
    payload = serialize_intermediate_bounds(bounds)
    artifact = tmp_path / "payload.pt"
    torch.save({"external_intermediate_bounds": payload}, artifact)

    loaded = torch.load(artifact, map_location="cpu", weights_only=True)
    restored = deserialize_intermediate_bounds(loaded["external_intermediate_bounds"])

    assert intermediate_bounds_sha256(restored) == intermediate_bounds_sha256(bounds)
    assert restored[0].external_relu_name == "/relu"
    assert restored[0].external_preactivation_name == "/pre"
    torch.testing.assert_close(restored[0].lower, bounds[0].lower)
    local = {"h1": IntervalState(lower=torch.zeros(1, 2), upper=torch.ones(1, 2))}
    rebound = bind_intermediate_bounds(restored, local)
    torch.testing.assert_close(rebound["h1"].upper, bounds[0].upper)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload["records"][0].update({"ordinal": 1}), "ordinals"),
        (
            lambda payload: payload["records"][0]["lower"].add_(1.0),
            "lower digest",
        ),
        (
            lambda payload: payload["records"][0].update({"shape": [1, 3]}),
            "recorded shape",
        ),
    ),
)
def test_intermediate_bounds_portable_payload_rejects_tamper(
    mutation, message: str
) -> None:
    bounds = (
        CapturedIntermediateBound(
            ordinal=0,
            external_relu_name="/relu",
            external_preactivation_name="/pre",
            lower=torch.tensor([[-0.25, 0.1]]),
            upper=torch.tensor([[0.75, 0.2]]),
        ),
    )
    payload = serialize_intermediate_bounds(bounds)
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        deserialize_intermediate_bounds(payload)
