"""Provider-neutral helpers for the FSG3 real same-solver worker."""

# pylint: disable=missing-function-docstring,protected-access

from types import SimpleNamespace

import pytest
import torch

from boundflow.runtime.fsg3_same_solver_timing import FSG3Configuration
from scripts import run_fsg3_same_solver_timing as runner
from scripts import run_rvir_v4_live_return_capture as live_runner


def test_upper_projection_encodes_only_positive_infinity() -> None:
    shape, values, mask = runner._upper_values(
        torch.tensor([[float("inf"), 2.0]], dtype=torch.float32)
    )
    assert shape == (1, 2)
    assert values == (0.0, 2.0)
    assert mask == (True, False)
    with pytest.raises(ValueError, match="NaN or negative"):
        runner._upper_values(torch.tensor([float("-inf")]))


def test_live_executor_requires_paired_precompiled_inputs() -> None:
    with pytest.raises(ValueError, match="must be paired"):
        live_runner._LiveExecutor(
            model=runner.Path("model.onnx"),
            torch_module=torch,
            arguments_module=SimpleNamespace(),
            precompiled_program=object(),
            capture_payloads=False,
        )
    executor = live_runner._LiveExecutor(
        model=runner.Path("model.onnx"),
        torch_module=torch,
        arguments_module=SimpleNamespace(),
        precompiled_program=object(),
        precompiled_module=object(),
        capture_payloads=False,
    )
    assert executor.capture_payloads is False
    assert executor.last_core_result is None
    assert executor.last_post_result is None


def test_profile_recorder_emits_non_overlapping_cuda_span() -> None:
    class FakeEvent:
        def record(self, _stream: object) -> None:
            return None

        def elapsed_time(self, _other: object) -> float:
            return 0.25

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=lambda **_kwargs: FakeEvent(),
            current_stream=lambda: object(),
        )
    )
    recorder = runner._ProfileRecorder(fake_torch)
    with recorder.span(
        scope="core",
        name="provider_core",
        stack_layer="solver/provider",
        solver_phase="official_update_bounds_core",
        resource="host+cuda",
        cache_state="process-hit",
    ):
        pass
    spans = recorder.finalize()
    assert len(spans) == 1
    assert spans[0].wall_ns > 0
    assert spans[0].gpu_ns == 250_000


def test_protocol_identity_is_common_across_configurations() -> None:
    identities = {
        runner._protocol_identity(configuration) for configuration in FSG3Configuration
    }
    assert len(identities) == 1
