import torch

from boundflow.domains.interval import IntervalState
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_beta_crown import run_alpha_beta_crown_mlp
from boundflow.runtime.alpha_crown import run_alpha_crown_mlp
from boundflow.runtime.bab import BabConfig, solve_bab_mlp
from boundflow.runtime.crown_ibp import (
    AffineBackwardState,
    _backprop_relu_step,
    _backprop_relu_step_dense_reference,
    _forward_ibp_trace_mlp,
    _relu_backward_mode,
    run_crown_ibp_mlp,
)
from boundflow.runtime.linear_operator import (
    AddLinearOperator,
    DenseLinearOperator,
    SignSplitLinearOperator,
)
from boundflow.runtime.task_executor import InputSpec


def _flat_case() -> (
    tuple[IntervalState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    pre = IntervalState(
        lower=torch.tensor(
            [[0.2, -2.0, -1.0, -0.5], [0.1, -1.5, -0.8, -0.2]], dtype=torch.float64
        ),
        upper=torch.tensor(
            [[1.0, -0.1, 2.0, 0.5], [0.9, -0.2, 1.2, 0.7]], dtype=torch.float64
        ),
    )
    torch.manual_seed(3)
    A_u = torch.randn(2, 3, 4, dtype=torch.float64)
    A_l = torch.randn(2, 3, 4, dtype=torch.float64)
    b_u = torch.randn(2, 3, dtype=torch.float64)
    b_l = torch.randn(2, 3, dtype=torch.float64)
    return pre, A_u, A_l, b_u, b_l


def _make_relu_mlp() -> BFTaskModule:
    torch.manual_seed(11)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="linear",
                name="linear1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(
                op_type="linear",
                name="linear2",
                inputs=["r1", "W2", "b2"],
                outputs=["out"],
            ),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={
            "params": {
                "W1": torch.randn(6, 4, dtype=torch.float64),
                "b1": torch.randn(6, dtype=torch.float64),
                "W2": torch.randn(3, 6, dtype=torch.float64),
                "b2": torch.randn(3, dtype=torch.float64),
            }
        },
    )


def test_dense_relu_reference_matches_current_wrapper_with_alpha_and_beta_coefficients() -> (
    None
):
    pre, A_u, A_l, b_u, b_l = _flat_case()
    alpha = torch.tensor([0.2, 0.3, 0.6, 0.8], dtype=torch.float64)
    add_u = torch.tensor(
        [[0.0, 0.0, 0.1, -0.2], [0.0, 0.0, 0.3, -0.4]], dtype=torch.float64
    )
    add_l = -add_u

    reference = _backprop_relu_step_dense_reference(
        A_u,
        A_l,
        b_u,
        b_l,
        pre=pre,
        x_name="h1",
        relu_alpha={"h1": alpha},
        relu_pre_add_coeff_u={"h1": add_u},
        relu_pre_add_coeff_l={"h1": add_l},
        device=A_u.device,
        dtype=A_u.dtype,
        caller="test",
    )
    with _relu_backward_mode("structured"):
        wrapped = _backprop_relu_step(
            AffineBackwardState(
                A_u=DenseLinearOperator(A_u),
                A_l=DenseLinearOperator(A_l),
                b_u=b_u,
                b_l=b_l,
            ),
            pre=pre,
            x_name="h1",
            relu_alpha={"h1": alpha},
            relu_pre_add_coeff_u={"h1": add_u},
            relu_pre_add_coeff_l={"h1": add_l},
            device=A_u.device,
            dtype=A_u.dtype,
            caller="test",
        )

    assert torch.equal(wrapped.A_u.to_dense(), reference.A_u)
    assert torch.equal(wrapped.A_l.to_dense(), reference.A_l)
    assert torch.equal(wrapped.b_u, reference.b_u)
    assert torch.equal(wrapped.b_l, reference.b_l)
    assert isinstance(wrapped.A_u, AddLinearOperator)
    assert isinstance(wrapped.A_u.lhs, SignSplitLinearOperator)
    assert isinstance(wrapped.A_l, AddLinearOperator)
    assert isinstance(wrapped.A_l.lhs, SignSplitLinearOperator)


def test_dense_relu_reference_alpha_gradient_matches_sign_selection_formula() -> None:
    pre, A_u, A_l, b_u, b_l = _flat_case()
    alpha = torch.tensor([0.2, 0.3, 0.6, 0.8], dtype=torch.float64, requires_grad=True)
    result = _backprop_relu_step_dense_reference(
        A_u,
        A_l,
        b_u,
        b_l,
        pre=pre,
        x_name="h1",
        relu_alpha={"h1": alpha},
        relu_pre_add_coeff_u=None,
        relu_pre_add_coeff_l=None,
        device=A_u.device,
        dtype=A_u.dtype,
        caller="test",
    )
    (
        result.A_u.sum() + result.A_l.sum() + result.b_u.sum() + result.b_l.sum()
    ).backward()

    ambiguous = (pre.lower < 0) & (pre.upper > 0)
    expected = (
        torch.where((A_u < 0) & ambiguous.unsqueeze(1), A_u, torch.zeros_like(A_u))
        + torch.where((A_l >= 0) & ambiguous.unsqueeze(1), A_l, torch.zeros_like(A_l))
    ).sum(dim=(0, 1))
    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert torch.allclose(alpha.grad, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(alpha.grad[:2], torch.zeros(2, dtype=torch.float64))
    assert float(alpha.grad[2:].abs().sum()) > 0.0


def test_dense_relu_reference_supports_batched_nchw_alpha_gradient() -> None:
    torch.manual_seed(5)
    pre = IntervalState(
        lower=-torch.rand(2, 2, 2, 2, dtype=torch.float64),
        upper=torch.rand(2, 2, 2, 2, dtype=torch.float64),
    )
    A_u = torch.randn(2, 4, 8, dtype=torch.float64)
    A_l = torch.randn(2, 4, 8, dtype=torch.float64)
    alpha = torch.full((2, 2, 2), 0.5, dtype=torch.float64, requires_grad=True)
    result = _backprop_relu_step_dense_reference(
        A_u,
        A_l,
        torch.zeros(2, 4, dtype=torch.float64),
        torch.zeros(2, 4, dtype=torch.float64),
        pre=pre,
        x_name="conv_pre",
        relu_alpha={"conv_pre": alpha},
        relu_pre_add_coeff_u=None,
        relu_pre_add_coeff_l=None,
        device=A_u.device,
        dtype=A_u.dtype,
        caller="test",
    )
    loss = result.A_l.sum() + result.b_l.sum()
    loss.backward()

    assert tuple(result.A_u.shape) == (2, 4, 8)
    assert tuple(result.A_l.shape) == (2, 4, 8)
    assert alpha.grad is not None
    assert tuple(alpha.grad.shape) == (2, 2, 2)
    assert torch.isfinite(alpha.grad).all()
    assert float(alpha.grad.abs().sum()) > 0.0


def test_structured_relu_alpha_gradient_matches_dense_reference_path() -> None:
    pre, A_u, A_l, b_u, b_l = _flat_case()
    dense_alpha = torch.tensor(
        [0.2, 0.3, 0.6, 0.8], dtype=torch.float64, requires_grad=True
    )
    structured_alpha = dense_alpha.detach().clone().requires_grad_(True)
    state = AffineBackwardState(
        A_u=DenseLinearOperator(A_u),
        A_l=DenseLinearOperator(A_l),
        b_u=b_u,
        b_l=b_l,
    )

    with _relu_backward_mode("dense"):
        dense = _backprop_relu_step(
            state,
            pre=pre,
            x_name="h1",
            relu_alpha={"h1": dense_alpha},
            relu_pre_add_coeff_u=None,
            relu_pre_add_coeff_l=None,
            device=A_u.device,
            dtype=A_u.dtype,
            caller="test",
        )
    with _relu_backward_mode("structured"):
        structured = _backprop_relu_step(
            state,
            pre=pre,
            x_name="h1",
            relu_alpha={"h1": structured_alpha},
            relu_pre_add_coeff_u=None,
            relu_pre_add_coeff_l=None,
            device=A_u.device,
            dtype=A_u.dtype,
            caller="test",
        )
    dense_loss = dense.A_l.to_dense().sum() + dense.b_l.sum()
    structured_loss = structured.A_l.to_dense().sum() + structured.b_l.sum()
    dense_loss.backward()
    structured_loss.backward()

    assert isinstance(structured.A_u, SignSplitLinearOperator)
    assert isinstance(structured.A_l, SignSplitLinearOperator)
    assert torch.allclose(
        structured.A_u.to_dense(), dense.A_u.to_dense(), atol=1e-12, rtol=1e-12
    )
    assert torch.allclose(
        structured.A_l.to_dense(), dense.A_l.to_dense(), atol=1e-12, rtol=1e-12
    )
    assert structured_alpha.grad is not None and dense_alpha.grad is not None
    assert torch.allclose(
        structured_alpha.grad, dense_alpha.grad, atol=1e-12, rtol=1e-12
    )


def test_full_crown_and_multistep_alpha_match_dense_reference_mode() -> None:
    module = _make_relu_mlp()
    spec = InputSpec.linf(
        value_name="input", center=torch.randn(2, 4, dtype=torch.float64), eps=0.1
    )
    linear_spec = torch.randn(2, 5, 3, dtype=torch.float64)

    with _relu_backward_mode("dense"):
        dense_crown = run_crown_ibp_mlp(module, spec, linear_spec_C=linear_spec)
        dense_alpha, dense_state, _ = run_alpha_crown_mlp(
            module,
            spec,
            linear_spec_C=linear_spec,
            steps=3,
            lr=0.1,
        )
    with _relu_backward_mode("structured"):
        structured_crown = run_crown_ibp_mlp(module, spec, linear_spec_C=linear_spec)
        structured_alpha, structured_state, _ = run_alpha_crown_mlp(
            module,
            spec,
            linear_spec_C=linear_spec,
            steps=3,
            lr=0.1,
        )

    assert torch.allclose(
        structured_crown.lower, dense_crown.lower, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        structured_crown.upper, dense_crown.upper, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        structured_alpha.lower, dense_alpha.lower, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        structured_alpha.upper, dense_alpha.upper, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        structured_state.alpha_by_relu_input["h1"],
        dense_state.alpha_by_relu_input["h1"],
        atol=1e-10,
        rtol=1e-10,
    )


def test_alpha_beta_and_bab_match_dense_reference_mode() -> None:
    module = _make_relu_mlp()
    spec = InputSpec.linf(
        value_name="input", center=torch.zeros(1, 4, dtype=torch.float64), eps=1.0
    )
    _interval, relu_pre = _forward_ibp_trace_mlp(module, spec)
    pre = relu_pre["h1"]
    ambiguous = ((pre.lower[0] < 0) & (pre.upper[0] > 0)).nonzero()
    assert int(ambiguous.numel()) > 0
    split = torch.zeros(6, dtype=torch.int8)
    split[int(ambiguous[0].item())] = 1
    split_state = {"h1": split}

    with _relu_backward_mode("dense"):
        dense_bounds, dense_alpha, dense_beta, _ = run_alpha_beta_crown_mlp(
            module,
            spec,
            relu_split_state=split_state,
            steps=2,
            lr=0.1,
            beta_init=0.1,
        )
    with _relu_backward_mode("structured"):
        structured_bounds, structured_alpha, structured_beta, _ = (
            run_alpha_beta_crown_mlp(
                module,
                spec,
                relu_split_state=split_state,
                steps=2,
                lr=0.1,
                beta_init=0.1,
            )
        )

    assert torch.allclose(
        structured_bounds.lower, dense_bounds.lower, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        structured_bounds.upper, dense_bounds.upper, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        structured_alpha.alpha_by_relu_input["h1"],
        dense_alpha.alpha_by_relu_input["h1"],
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        structured_beta.beta_by_relu_input["h1"],
        dense_beta.beta_by_relu_input["h1"],
        atol=1e-10,
        rtol=1e-10,
    )

    config = BabConfig(
        max_nodes=8,
        oracle="alpha_beta",
        node_batch_size=2,
        enable_node_eval_cache=False,
        alpha_steps=2,
        alpha_lr=0.1,
        threshold=0.0,
    )
    with _relu_backward_mode("dense"):
        dense_bab = solve_bab_mlp(module, spec, config=config)
    with _relu_backward_mode("structured"):
        structured_bab = solve_bab_mlp(module, spec, config=config)

    assert structured_bab.status == dense_bab.status
    assert structured_bab.nodes_visited == dense_bab.nodes_visited
    assert structured_bab.nodes_evaluated == dense_bab.nodes_evaluated
    assert structured_bab.nodes_expanded == dense_bab.nodes_expanded
    assert abs(structured_bab.best_lower - dense_bab.best_lower) <= 1e-10
    assert abs(structured_bab.best_upper - dense_bab.best_upper) <= 1e-10
