"""End-to-end PR-12 fused-region integration and fallback gates."""

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.fused_crown import (
    BoundaryRepresentation,
    FusedCrownExecutionContext,
    FusedCrownExecutionStep,
    FusedReluAffineDescriptor,
    FusedReluAffineRequest,
    InternalMaterializationPolicy,
    TVMFusedCrownExecutor,
    TorchDenseFusedCrownReference,
    fused_crown_graph_fingerprint,
    plan_fused_crown_regions,
    validate_fused_crown_execution_steps,
)
from boundflow.runtime.materialization import trace_materializations
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


def _fanout_mlp_module(device: torch.device) -> BFTaskModule:
    task = BoundTask(
        task_id="fanout",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp("linear", "shared", ["input", "w1", "b1"], ["h"]),
            TaskOp("relu", "relu", ["h"], ["r"]),
            TaskOp("linear", "direct", ["h", "wd", "bd"], ["d"]),
            TaskOp("linear", "relu_path", ["r", "wr", "br"], ["q"]),
            TaskOp("add", "merge", ["d", "q"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="fanout",
        bindings={
            "params": {
                "w1": torch.randn(4, 3, device=device),
                "b1": torch.randn(4, device=device),
                "wd": torch.randn(2, 4, device=device),
                "bd": torch.randn(2, device=device),
                "wr": torch.randn(2, 4, device=device),
                "br": torch.randn(2, device=device),
            }
        },
    )


def _mini_resnet_module(device: torch.device) -> BFTaskModule:
    conv = {"stride": (1, 1), "padding": (1, 1), "dilation": (1, 1), "groups": 1}
    down = {"stride": (2, 2), "padding": (1, 1), "dilation": (1, 1), "groups": 1}
    projection = {"stride": (2, 2), "padding": (0, 0), "dilation": (1, 1), "groups": 1}
    task = BoundTask(
        task_id="mini_resnet",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp("conv2d", "stem", ["input", "ws", "bs"], ["hs"], conv),
            TaskOp("relu", "stem_relu", ["hs"], ["r0"]),
            TaskOp("conv2d", "b1c1", ["r0", "w11", "b11"], ["h11"], conv),
            TaskOp("relu", "b1_relu", ["h11"], ["r11"]),
            TaskOp("conv2d", "b1c2", ["r11", "w12", "b12"], ["h12"], conv),
            TaskOp("add", "b1_add", ["r0", "h12"], ["s1"]),
            TaskOp("relu", "b1_out", ["s1"], ["r1"]),
            TaskOp("conv2d", "b2c1", ["r1", "w21", "b21"], ["h21"], down),
            TaskOp("relu", "b2_relu", ["h21"], ["r21"]),
            TaskOp("conv2d", "b2c2", ["r21", "w22", "b22"], ["h22"], conv),
            TaskOp("conv2d", "b2_skip", ["r1", "wp", "bp"], ["hp"], projection),
            TaskOp("add", "b2_add", ["h22", "hp"], ["s2"]),
            TaskOp("relu", "b2_out", ["s2"], ["r2"]),
            TaskOp(
                "flatten", "flatten", ["r2"], ["flat"], {"start_dim": 1, "end_dim": -1}
            ),
            TaskOp("linear", "head", ["flat", "wh", "bh"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="mini_resnet",
        bindings={
            "params": {
                "ws": torch.randn(4, 3, 3, 3, device=device),
                "bs": torch.randn(4, device=device),
                "w11": torch.randn(4, 4, 3, 3, device=device),
                "b11": torch.randn(4, device=device),
                "w12": torch.randn(4, 4, 3, 3, device=device),
                "b12": torch.randn(4, device=device),
                "w21": torch.randn(8, 4, 3, 3, device=device),
                "b21": torch.randn(8, device=device),
                "w22": torch.randn(8, 8, 3, 3, device=device),
                "b22": torch.randn(8, device=device),
                "wp": torch.randn(8, 4, 1, 1, device=device),
                "bp": torch.randn(8, device=device),
                "wh": torch.randn(3, 8 * 4 * 4, device=device),
                "bh": torch.randn(3, device=device),
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


class _TrackingReference(TorchDenseFusedCrownReference):
    def __init__(self) -> None:
        self.descriptor_calls = 0
        self.support_calls = 0
        self.run_calls = 0

    def supports_descriptor(
        self,
        descriptor: FusedReluAffineDescriptor,
        context: FusedCrownExecutionContext,
    ) -> bool:
        self.descriptor_calls += 1
        return super().supports_descriptor(descriptor, context)

    def supports(
        self,
        request: FusedReluAffineRequest,
        context: FusedCrownExecutionContext,
    ) -> bool:
        self.support_calls += 1
        return super().supports(request, context)

    def run(self, request: FusedReluAffineRequest, *, stream=None):  # type: ignore[no-untyped-def]
        self.run_calls += 1
        return super().run(request, stream=stream)


class _TrackingTVM(TVMFusedCrownExecutor):
    def __init__(self) -> None:
        self.run_calls = 0

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


def test_fanout_affine_is_not_fused_and_remains_sound() -> None:
    torch.manual_seed(99)
    module = _fanout_mlp_module(torch.device("cpu"))
    task = module.get_entry_task()
    center = torch.randn(1, 3)
    spec = InputSpec.linf(value_name="input", center=center, eps=0.1)
    executor = _TrackingReference()

    steps = plan_fused_crown_regions(task.ops)
    assert steps == ()
    dense = run_crown_ibp_mlp(module, spec)
    fused = run_crown_ibp_mlp(
        module,
        spec,
        fused_crown_executor=executor,
        fused_crown_steps=steps,
    )

    torch.testing.assert_close(fused.lower, dense.lower)
    torch.testing.assert_close(fused.upper, dense.upper)
    assert executor.run_calls == 0

    params = module.bindings["params"]
    samples = center.unsqueeze(0) + (torch.rand(2048, 1, 3) * 2.0 - 1.0) * 0.1
    h = samples @ params["w1"].t() + params["b1"]
    outputs = (
        h @ params["wd"].t()
        + params["bd"]
        + torch.relu(h) @ params["wr"].t()
        + params["br"]
    )
    assert (outputs >= fused.lower.unsqueeze(0) - 1e-5).all()
    assert (outputs <= fused.upper.unsqueeze(0) + 1e-5).all()


def test_runtime_rejects_forged_fanout_step_defensively() -> None:
    torch.manual_seed(12131)
    module = _fanout_mlp_module(torch.device("cpu"))
    task = module.get_entry_task()
    forged = FusedCrownExecutionStep(
        kind="fused_relu_linear",
        relu_op_index=1,
        affine_op_index=0,
        consumed_outputs=("r", "h"),
        graph_fingerprint=fused_crown_graph_fingerprint(task.ops),
    )
    assert validate_fused_crown_execution_steps(task.ops, (forged,)) == ()
    executor = _TrackingReference()
    spec = InputSpec.linf(value_name="input", center=torch.randn(1, 3), eps=0.05)

    expected = run_crown_ibp_mlp(module, spec)
    actual = run_crown_ibp_mlp(
        module,
        spec,
        fused_crown_executor=executor,
        fused_crown_steps=(forged,),
    )

    torch.testing.assert_close(actual.lower, expected.lower)
    torch.testing.assert_close(actual.upper, expected.upper)
    assert executor.run_calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("graph_fingerprint", "stale"),
        ("kind", "fused_relu_conv2d"),
        ("backend", "pytorch_eager"),
        ("boundary_representation", BoundaryRepresentation.STRUCTURED),
        ("internal_materialization", InternalMaterializationPolicy.EAGER),
    ],
)
def test_invalid_execution_step_contract_falls_back(field: str, value: object) -> None:
    torch.manual_seed(12132)
    module = _chain_module(torch.device("cpu"))
    task = module.get_entry_task()
    valid = plan_fused_crown_regions(task.ops)[0]
    invalid = replace(valid, **{field: value})
    executor = _TrackingReference()
    spec = InputSpec.linf(value_name="input", center=torch.randn(2, 5), eps=0.05)

    expected = run_crown_ibp_mlp(module, spec)
    actual = run_crown_ibp_mlp(
        module,
        spec,
        fused_crown_executor=executor,
        fused_crown_steps=(invalid,),
    )

    torch.testing.assert_close(actual.lower, expected.lower)
    torch.testing.assert_close(actual.upper, expected.upper)
    assert executor.run_calls == 0


@pytest.mark.parametrize("mode", ["alpha", "beta", "grad", "split"])
def test_unsupported_solver_modes_never_invoke_executor(mode: str) -> None:
    torch.manual_seed(12133)
    module = _chain_module(torch.device("cpu"))
    center = torch.randn(2, 5)
    kwargs: dict[str, object] = {}
    if mode == "alpha":
        kwargs["relu_alpha"] = {"h1": torch.full((7,), 0.4)}
    elif mode == "beta":
        kwargs["relu_pre_add_coeff_u"] = {"h1": torch.zeros(7)}
    elif mode == "grad":
        center.requires_grad_(True)
    else:
        kwargs["relu_split_state"] = {"h1": torch.zeros(2, 7, dtype=torch.int64)}
    spec = InputSpec.linf(value_name="input", center=center, eps=0.05)
    executor = _TrackingReference()

    expected = run_crown_ibp_mlp(module, spec, **kwargs)  # type: ignore[arg-type]
    actual = run_crown_ibp_mlp(
        module,
        spec,
        fused_crown_executor=executor,
        fused_crown_steps=plan_fused_crown_regions(module.get_entry_task().ops),
        **kwargs,  # type: ignore[arg-type]
    )

    torch.testing.assert_close(actual.lower, expected.lower)
    torch.testing.assert_close(actual.upper, expected.upper)
    assert executor.descriptor_calls == 0
    assert executor.run_calls == 0


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
def test_tvm_fused_runtime_matches_multiblock_mini_resnet() -> None:
    torch.manual_seed(1217)
    device = torch.device("cuda")
    module = _mini_resnet_module(device)
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(1, 3, 8, 8, device=device),
        eps=0.025,
    )
    steps = plan_fused_crown_regions(module.get_entry_task().ops)
    assert len(steps) == 3
    executor = _TrackingTVM()

    dense = run_crown_ibp_mlp(module, spec)
    fused = run_crown_ibp_mlp(
        module,
        spec,
        fused_crown_executor=executor,
        fused_crown_steps=steps,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fused.lower, dense.lower, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(fused.upper, dense.upper, rtol=1e-4, atol=1e-4)
    assert torch.isfinite(fused.lower).all() and torch.isfinite(fused.upper).all()
    assert (fused.lower <= fused.upper).all()
    assert executor.run_calls == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unsupported_grouped_conv_falls_back_before_fused_materialization() -> None:
    torch.manual_seed(1218)
    device = torch.device("cuda")
    task = BoundTask(
        task_id="grouped",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                "conv2d",
                "grouped_conv",
                ["input", "w", "b"],
                ["h"],
                {"stride": (1, 1), "padding": (1, 1), "dilation": (1, 1), "groups": 2},
            ),
            TaskOp("relu", "relu", ["h"], ["r"]),
            TaskOp(
                "flatten", "flatten", ["r"], ["flat"], {"start_dim": 1, "end_dim": -1}
            ),
            TaskOp("linear", "head", ["flat", "wh", "bh"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    module = BFTaskModule(
        tasks=[task],
        entry_task_id="grouped",
        bindings={
            "params": {
                "w": torch.randn(4, 1, 3, 3, device=device),
                "b": torch.randn(4, device=device),
                "wh": torch.randn(3, 4 * 5 * 5, device=device),
                "bh": torch.randn(3, device=device),
            }
        },
    )
    spec = InputSpec.linf(
        value_name="input", center=torch.randn(1, 2, 5, 5, device=device), eps=0.03
    )
    executor = _TrackingTVM()

    expected = run_crown_ibp_mlp(module, spec)
    with trace_materializations(
        run_id="unsupported-grouped",
        query_id="q0",
        bound_method="CROWN",
        spec_batch=3,
        domain_batch=1,
    ) as trace:
        actual = run_crown_ibp_mlp(
            module,
            spec,
            fused_crown_executor=executor,
            fused_crown_steps=plan_fused_crown_regions(task.ops),
        )

    torch.testing.assert_close(actual.lower, expected.lower)
    torch.testing.assert_close(actual.upper, expected.upper)
    assert executor.run_calls == 0
    assert all(event.reason != "fused_region_dense_boundary" for event in trace.events)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tvm_executor_honors_non_default_torch_stream_without_global_sync() -> None:
    torch.manual_seed(1219)
    device = torch.device("cuda")
    request = FusedReluAffineRequest(
        kind="linear",
        A_u=torch.randn(1, 2, 16, device=device),
        A_l=torch.randn(1, 2, 16, device=device),
        alpha_u=torch.rand(1, 16, device=device),
        alpha_l=torch.rand(1, 16, device=device),
        beta_u=torch.rand(1, 16, device=device),
        beta_l=torch.rand(1, 16, device=device),
        weight=torch.randn(16, 8, device=device),
        bias=torch.randn(16, device=device),
        input_shape=(8,),
        output_shape=(16,),
        attrs={},
    )
    executor = TVMFusedCrownExecutor()
    expected = TorchDenseFusedCrownReference().run(request)
    executor.run(request)
    torch.cuda.synchronize()

    default = torch.cuda.default_stream(device)
    custom = torch.cuda.Stream(device=device)
    with torch.cuda.stream(default):
        torch.cuda._sleep(200_000_000)
    with torch.cuda.stream(custom):
        actual = executor.run(request, stream=custom)
        snapshot = actual.A_prev_u.clone()
    custom.synchronize()

    torch.testing.assert_close(snapshot, expected.A_prev_u, rtol=2e-5, atol=2e-5)
    default.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tvm_ffi_stream_id_matches_torch_custom_stream() -> None:
    import tvm_ffi

    stream = torch.cuda.Stream()
    device = tvm_ffi.device("cuda:0")
    with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
        assert torch.cuda.current_stream().cuda_stream == stream.cuda_stream
        assert tvm_ffi.get_raw_stream(device) == stream.cuda_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tvm_dlpack_adapter_aliases_torch_storage() -> None:
    import tvm

    source = torch.randn(16, device="cuda")
    tvm_view = tvm.runtime.from_dlpack(source)
    round_trip = torch.from_dlpack(tvm_view)

    assert round_trip.data_ptr() == source.data_ptr()
    round_trip.add_(1.0)
    torch.testing.assert_close(round_trip, source)
