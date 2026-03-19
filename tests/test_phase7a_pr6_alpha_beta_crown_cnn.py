import torch
import torch.nn.functional as F

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_beta_crown import check_first_layer_infeasible_split, run_alpha_beta_crown_mlp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


def _make_small_cnn_module() -> BFTaskModule:
    w1 = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    b1 = torch.zeros((1,), dtype=torch.float32)
    w2 = torch.ones((1, 4), dtype=torch.float32)
    b2 = torch.zeros((1,), dtype=torch.float32)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="flatten", name="flatten", inputs=["r1"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear1", inputs=["flat", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}},
    )


def _make_contradictory_first_layer_conv_module() -> BFTaskModule:
    w1 = torch.tensor([[[[1.0]]], [[[-1.0]]]], dtype=torch.float32)
    b1 = torch.tensor([-0.2, -0.2], dtype=torch.float32)
    w2 = torch.ones((1, 2), dtype=torch.float32)
    b2 = torch.zeros((1,), dtype=torch.float32)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="flatten", name="flatten1", inputs=["r1"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear1", inputs=["flat", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}},
    )


def _make_two_conv_module() -> BFTaskModule:
    w1 = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    b1 = torch.zeros((1,), dtype=torch.float32)
    w2 = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    b2 = torch.zeros((1,), dtype=torch.float32)
    w3 = torch.ones((1, 4), dtype=torch.float32)
    b3 = torch.zeros((1,), dtype=torch.float32)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(
                op_type="conv2d",
                name="conv2",
                inputs=["r1", "W2", "b2"],
                outputs=["h2"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu2", inputs=["h2"], outputs=["r2"]),
            TaskOp(op_type="flatten", name="flatten1", inputs=["r2"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear1", inputs=["flat", "W3", "b3"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3}},
    )


def _eval_small_cnn(xs: torch.Tensor) -> torch.Tensor:
    flat_x = xs.flatten(0, 1)
    h1 = F.conv2d(flat_x, torch.ones((1, 1, 1, 1), dtype=xs.dtype, device=xs.device), bias=None)
    r1 = torch.relu(h1)
    flat = r1.flatten(1)
    out = flat.matmul(torch.ones((4, 1), dtype=xs.dtype, device=xs.device)).reshape(xs.shape[0], xs.shape[1], 1)
    return out


def _sample_points_with_first_pixel_active(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    xs = x0.unsqueeze(0) + float(eps) * (torch.rand((n,) + tuple(x0.shape), dtype=x0.dtype, device=x0.device) * 2.0 - 1.0)
    xs[:, :, 0, 0, 0] = torch.rand((n, int(x0.shape[0])), dtype=x0.dtype, device=x0.device) * float(eps)
    return xs


def test_phase7a_pr6_conv_split_structured_matches_flat_and_branch_choices_are_flat() -> None:
    module = _make_small_cnn_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)
    split_struct = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.int8)
    split_flat = split_struct.flatten()

    bounds_struct, alpha_struct, beta_struct, stats_struct = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state={"h1": split_struct},
        steps=0,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
        objective="lower",
    )
    bounds_flat, alpha_flat, beta_flat, stats_flat = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state={"h1": split_flat},
        steps=0,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
        objective="lower",
    )

    assert torch.allclose(bounds_struct.lower, bounds_flat.lower, atol=1e-6, rtol=1e-6)
    assert torch.allclose(bounds_struct.upper, bounds_flat.upper, atol=1e-6, rtol=1e-6)
    assert tuple(alpha_struct.alpha_by_relu_input["h1"].shape) == (1, 2, 2)
    assert tuple(beta_struct.beta_by_relu_input["h1"].shape) == (1, 2, 2)
    assert tuple(alpha_flat.alpha_by_relu_input["h1"].shape) == (1, 2, 2)
    assert tuple(beta_flat.beta_by_relu_input["h1"].shape) == (1, 2, 2)
    assert stats_struct.feasibility == "unknown"
    assert stats_struct.branch_choices == [("h1", 1)]
    assert stats_flat.branch_choices == stats_struct.branch_choices


def test_phase7a_pr6_conv_alpha_beta_improves_lower_and_is_sound() -> None:
    module = _make_small_cnn_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)
    split = {"h1": torch.tensor([[[1, 0], [0, 0]]], dtype=torch.int8)}

    b0, _a0, _beta0, _s0 = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state=split,
        steps=0,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
        objective="lower",
    )
    b1, a1, beta1, _s1 = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state=split,
        steps=20,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
        objective="lower",
    )

    assert float(b1.lower.item()) >= float(b0.lower.item()) - 1e-6
    assert tuple(a1.alpha_by_relu_input["h1"].shape) == (1, 2, 2)
    assert tuple(beta1.beta_by_relu_input["h1"].shape) == (1, 2, 2)

    torch.manual_seed(0)
    xs = _sample_points_with_first_pixel_active(x0=x0, eps=1.0, n=256)
    ys = _eval_small_cnn(xs)
    assert (ys >= b1.lower.unsqueeze(0) - 1e-5).all()
    assert (ys <= b1.upper.unsqueeze(0) + 1e-5).all()


def test_phase7a_pr6_conv_beta_grad_nonzero_and_finite() -> None:
    module = _make_small_cnn_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)
    split = {"h1": torch.tensor([[[1, 0], [0, 0]]], dtype=torch.int8)}
    beta = torch.nn.Parameter(torch.full((1, 2, 2), 0.1, dtype=torch.float32))

    bounds = run_crown_ibp_mlp(module, spec, relu_split_state=split, relu_pre_add_coeff_l={"h1": -beta})
    loss = -bounds.lower.mean()
    loss.backward()

    assert beta.grad is not None
    assert torch.isfinite(beta.grad).all()
    assert float(beta.grad.abs().sum().item()) > 0.0


def test_phase7a_pr6_conv_per_batch_params_and_warm_start() -> None:
    module = _make_small_cnn_module()
    x0 = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)
    split = {
        "h1": torch.tensor(
            [
                [[[1, 0], [0, 0]]],
                [[[0, 1], [0, 0]]],
            ],
            dtype=torch.int8,
        )
    }

    b0, a0, beta0, _ = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state=split,
        steps=0,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
        objective="lower",
        per_batch_params=True,
    )
    b1, a1, beta1, _ = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state=split,
        steps=6,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
        objective="lower",
        per_batch_params=True,
        warm_start_alpha=a0,
        warm_start_beta=beta0,
    )

    assert tuple(a0.alpha_by_relu_input["h1"].shape) == (2, 1, 2, 2)
    assert tuple(beta0.beta_by_relu_input["h1"].shape) == (2, 1, 2, 2)
    assert tuple(a1.alpha_by_relu_input["h1"].shape) == (2, 1, 2, 2)
    assert tuple(beta1.beta_by_relu_input["h1"].shape) == (2, 1, 2, 2)
    assert float(b1.lower.mean().item()) >= float(b0.lower.mean().item()) - 1e-6


def test_phase7a_pr6_first_layer_conv_infeasible_split_detected() -> None:
    module = _make_contradictory_first_layer_conv_module()
    x0 = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)
    split = {"h1": torch.tensor([[[1]], [[1]]], dtype=torch.int8)}

    stats = check_first_layer_infeasible_split(module, spec, relu_split_state=split)

    assert stats.feasibility == "infeasible"
    assert stats.infeasible_certificate is not None
    assert float(stats.infeasible_certificate["max_value"]) < 0.0


def test_phase7a_pr6_deeper_conv_split_skips_detector_but_oracle_runs() -> None:
    module = _make_two_conv_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)
    split = {"h2": torch.tensor([[[1, 0], [0, 0]]], dtype=torch.int8)}

    stats = check_first_layer_infeasible_split(module, spec, relu_split_state=split)
    bounds, _alpha, _beta, oracle_stats = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state=split,
        steps=0,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
        objective="lower",
    )

    assert stats.feasibility == "unknown"
    assert "first-layer split halfspaces" in stats.reason
    assert tuple(bounds.lower.shape) == (1, 1)
    assert oracle_stats.feasibility == "unknown"
