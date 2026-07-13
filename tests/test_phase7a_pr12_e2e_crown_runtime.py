"""End-to-end PR-12 fused-region integration and fallback gates."""

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.fused_crown import (
    FusedCrownExecutionContext,
    FusedReluAffineRequest,
    TVMFusedCrownExecutor,
    TorchDenseFusedCrownReference,
    plan_fused_crown_regions,
)
from boundflow.runtime.task_executor import InputSpec


def _chain_module(device: torch.device) -> BFTaskModule:
    task = BoundTask(
        task_id="chain",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp("linear", "linear1", ["input", "w1", "b1"], ["h1"]),
            TaskOp("relu", "relu1", ["h1"], ["r1"]),
            TaskOp("linear", "linear2", ["r1", "w2", "b2"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="chain",
        bindings={
            "params": {
                "w1": torch.randn(7, 5, device=device),
                "b1": torch.randn(7, device=device),
                "w2": torch.randn(3, 7, device=device),
                "b2": torch.randn(3, device=device),
            }
        },
    )


def _chain_cnn_module(device: torch.device) -> BFTaskModule:
    task = BoundTask(
        task_id="cnn",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                "conv2d",
                "conv1",
                ["input", "w1", "b1"],
                ["h1"],
                {"stride": (1, 1), "padding": (1, 1), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp("relu", "relu1", ["h1"], ["r1"]),
            TaskOp(
                "conv2d",
                "conv2",
                ["r1", "w2", "b2"],
                ["h2"],
                {"stride": (2, 2), "padding": (1, 1), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp("relu", "relu2", ["h2"], ["r2"]),
            TaskOp(
                "flatten", "flatten", ["r2"], ["flat"], {"start_dim": 1, "end_dim": -1}
            ),
            TaskOp("linear", "head", ["flat", "w3", "b3"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="cnn",
        bindings={
            "params": {
                "w1": torch.randn(2, 1, 3, 3, device=device),
                "b1": torch.randn(2, device=device),
                "w2": torch.randn(3, 2, 3, 3, device=device),
                "b2": torch.randn(3, device=device),
                "w3": torch.randn(4, 27, device=device),
                "b3": torch.randn(4, device=device),
            }
        },
    )


def _residual_cnn_module(device: torch.device, *, downsample: bool) -> BFTaskModule:
    stride = (2, 2) if downsample else (1, 1)
    spatial = 4 if downsample else 7
    task = BoundTask(
        task_id="residual",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                "conv2d",
                "main1",
                ["input", "wm1", "bm1"],
                ["hm1"],
                {"stride": stride, "padding": (1, 1), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp("relu", "main_relu", ["hm1"], ["rm1"]),
            TaskOp(
                "conv2d",
                "main2",
                ["rm1", "wm2", "bm2"],
                ["hm2"],
                {"stride": (1, 1), "padding": (1, 1), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(
                "conv2d",
                "skip",
                ["input", "ws", "bs"],
                ["hs"],
                {"stride": stride, "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp("add", "merge", ["hm2", "hs"], ["sum"]),
            TaskOp("relu", "out_relu", ["sum"], ["rout"]),
            TaskOp(
                "flatten",
                "flatten",
                ["rout"],
                ["flat"],
                {"start_dim": 1, "end_dim": -1},
            ),
            TaskOp("linear", "head", ["flat", "wh", "bh"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="residual",
        bindings={
            "params": {
                "wm1": torch.randn(3, 2, 3, 3, device=device),
                "bm1": torch.randn(3, device=device),
                "wm2": torch.randn(3, 3, 3, 3, device=device),
                "bm2": torch.randn(3, device=device),
                "ws": torch.randn(3, 2, 1, 1, device=device),
                "bs": torch.randn(3, device=device),
                "wh": torch.randn(4, 3 * spatial * spatial, device=device),
                "bh": torch.randn(4, device=device),
            }
        },
    )


class _RejectingExecutor(TorchDenseFusedCrownReference):
    def __init__(self) -> None:
        self.support_calls = 0
        self.run_calls = 0

    def supports(
        self,
        request: FusedReluAffineRequest,
        context: FusedCrownExecutionContext,
    ) -> bool:
        del request, context
        self.support_calls += 1
        return False

    def run(self, request: FusedReluAffineRequest, *, stream=None):  # type: ignore[no-untyped-def]
        self.run_calls += 1
        return super().run(request, stream=stream)


@pytest.mark.parametrize("family", ["linear", "conv2d"])
def test_fused_reference_matches_dense_final_bounds(family: str) -> None:
    torch.manual_seed(1212)
    device = torch.device("cpu")
    module = _chain_module(device) if family == "linear" else _chain_cnn_module(device)
    shape = (2, 5) if family == "linear" else (2, 1, 5, 5)
    center = torch.randn(*shape, device=device)
    spec = InputSpec.linf(value_name="input", center=center, eps=0.08)
    steps = plan_fused_crown_regions(module.get_entry_task().ops)

    dense = run_crown_ibp_mlp(module, spec)
    fused = run_crown_ibp_mlp(
        module,
        spec,
        fused_crown_executor=TorchDenseFusedCrownReference(),
        fused_crown_steps=steps,
    )

    torch.testing.assert_close(fused.lower, dense.lower, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(fused.upper, dense.upper, rtol=2e-5, atol=2e-5)
    assert torch.isfinite(fused.lower).all() and torch.isfinite(fused.upper).all()
    assert (fused.lower <= fused.upper).all()


def test_unsupported_executor_falls_back_without_running_candidate() -> None:
    torch.manual_seed(1213)
    module = _chain_module(torch.device("cpu"))
    spec = InputSpec.linf(value_name="input", center=torch.randn(2, 5), eps=0.05)
    rejecting = _RejectingExecutor()

    expected = run_crown_ibp_mlp(module, spec)
    actual = run_crown_ibp_mlp(
        module,
        spec,
        fused_crown_executor=rejecting,
        fused_crown_steps=plan_fused_crown_regions(module.get_entry_task().ops),
    )

    torch.testing.assert_close(actual.lower, expected.lower)
    torch.testing.assert_close(actual.upper, expected.upper)
    assert rejecting.support_calls == 1
    assert rejecting.run_calls == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("family", ["linear", "conv2d"])
def test_tvm_fused_runtime_matches_dense_final_bounds(family: str) -> None:
    torch.manual_seed(1214)
    device = torch.device("cuda")
    module = _chain_module(device) if family == "linear" else _chain_cnn_module(device)
    shape = (2, 5) if family == "linear" else (2, 1, 5, 5)
    spec = InputSpec.linf(
        value_name="input", center=torch.randn(*shape, device=device), eps=0.06
    )

    dense = run_crown_ibp_mlp(module, spec)
    fused = run_crown_ibp_mlp(
        module,
        spec,
        fused_crown_executor=TVMFusedCrownExecutor(),
        fused_crown_steps=plan_fused_crown_regions(module.get_entry_task().ops),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fused.lower, dense.lower, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(fused.upper, dense.upper, rtol=5e-5, atol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("downsample", [False, True])
def test_tvm_fused_runtime_matches_residual_final_bounds(downsample: bool) -> None:
    torch.manual_seed(1215 + int(downsample))
    device = torch.device("cuda")
    module = _residual_cnn_module(device, downsample=downsample)
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(2, 2, 7, 7, device=device),
        eps=0.04,
    )

    dense = run_crown_ibp_mlp(module, spec)
    fused = run_crown_ibp_mlp(
        module,
        spec,
        fused_crown_executor=TVMFusedCrownExecutor(),
        fused_crown_steps=plan_fused_crown_regions(module.get_entry_task().ops),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fused.lower, dense.lower, rtol=8e-5, atol=8e-5)
    torch.testing.assert_close(fused.upper, dense.upper, rtol=8e-5, atol=8e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tvm_dlpack_adapter_aliases_torch_storage() -> None:
    import tvm

    source = torch.randn(16, device="cuda")
    tvm_view = tvm.runtime.from_dlpack(source)
    round_trip = torch.from_dlpack(tvm_view)

    assert round_trip.data_ptr() == source.data_ptr()
    round_trip.add_(1.0)
    torch.testing.assert_close(round_trip, source)
