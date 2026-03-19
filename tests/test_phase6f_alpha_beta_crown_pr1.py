import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_beta_crown import run_alpha_beta_crown_mlp
from boundflow.runtime.alpha_crown import run_alpha_crown_mlp
from boundflow.runtime.bab import BabConfig, solve_bab_mlp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


def _make_first_layer_two_neuron_module(*, w: torch.Tensor, b: torch.Tensor) -> BFTaskModule:
    # input -> linear -> relu -> linear(out=identity on hidden) -> out
    hidden, in_dim = int(w.shape[0]), int(w.shape[1])
    assert tuple(b.shape) == (hidden,)
    w2 = torch.eye(hidden, dtype=w.dtype)
    b2 = torch.zeros(hidden, dtype=w.dtype)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["r1", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w, "b1": b, "W2": w2, "b2": b2}},
    )


def test_phase6f_beta_grad_is_nonzero_and_finite() -> None:
    # 1D toy: beta participates in the bound computation graph via split-constraint encoding.
    w = torch.tensor([[1.0]], dtype=torch.float32)
    b = torch.tensor([0.0], dtype=torch.float32)
    module = _make_first_layer_two_neuron_module(w=w, b=b)
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[0.0]], dtype=torch.float32), eps=1.0)
    split = {"h1": torch.tensor([+1], dtype=torch.int8)}

    beta = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
    # split: s*z>=0 with s=+1 => Lagrangian term is -beta*s*z = -beta*z
    bounds = run_crown_ibp_mlp(module, spec, relu_split_state=split, relu_pre_add_coeff_l={"h1": -beta})
    loss = -bounds.lower.mean()
    loss.backward()
    assert beta.grad is not None
    assert torch.isfinite(beta.grad).all()
    assert float(beta.grad.abs().sum().item()) > 0.0


def test_phase6f_nontrivial_empty_domain_detected_as_infeasible() -> None:
    # 2D Linf box x in [-1,1]^2.
    # Two first-layer neurons:
    #   z1 = x1 + x2 - 1.2
    #   z2 = -x1 - x2 - 1.2
    # Split both as active => z1>=0 and z2>=0 => (z1+z2)=-2.4 contradiction => infeasible.
    w = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
    b = torch.tensor([-1.2, -1.2], dtype=torch.float32)
    module = _make_first_layer_two_neuron_module(w=w, b=b)
    spec = InputSpec.linf(value_name="input", center=torch.zeros(1, 2, dtype=torch.float32), eps=1.0)
    split = {"h1": torch.tensor([+1, +1], dtype=torch.int8)}

    _bounds, _alpha, _beta, stats = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state=split,
        steps=0,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
    )
    assert stats.feasibility == "infeasible"
    assert stats.infeasible_certificate is not None


def test_phase6f_empty_domain_detected_for_non_pairwise_certificate() -> None:
    # 2D box x in [-1,1]^2.
    # Three constraints (all split active):
    #   z1 = x1 - 0.6 >= 0
    #   z2 = x2 - 0.6 >= 0
    #   z3 = -x1 - x2 - 0.6 >= 0
    # Sum => -1.8 >= 0 contradiction. No opposite pair is required.
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
    b = torch.tensor([-0.6, -0.6, -0.6], dtype=torch.float32)
    module = _make_first_layer_two_neuron_module(w=w, b=b)
    spec = InputSpec.linf(value_name="input", center=torch.zeros(1, 2, dtype=torch.float32), eps=1.0)
    split = {"h1": torch.tensor([+1, +1, +1], dtype=torch.int8)}

    _bounds, _alpha, _beta, stats = run_alpha_beta_crown_mlp(module, spec, relu_split_state=split, steps=0, lr=0.2)
    assert stats.feasibility == "infeasible"
    assert stats.infeasible_certificate is not None


def test_phase6f_empty_domain_detected_for_three_directions_l2_ball() -> None:
    # Clean non-pairwise certificate:
    # unit L2 ball centered at 0, and 3 constraints with directions 0/120/240 degrees.
    #
    # If all are split active:
    #   a_i·x - t >= 0  for i=1..3
    # but a1+a2+a3 = 0, so summing yields 0 - 3t >= 0 contradiction for t>0.
    #
    # This is a pure convex-combo (Farkas-style) certificate and cannot be caught by an "opposite pair" check.
    t = 0.49
    a1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    a2 = torch.tensor([-0.5, 0.8660254], dtype=torch.float32)  # sqrt(3)/2
    a3 = torch.tensor([-0.5, -0.8660254], dtype=torch.float32)
    w = torch.stack([a1, a2, a3], dim=0)  # [3,2]
    b = torch.full((3,), -t, dtype=torch.float32)
    module = _make_first_layer_two_neuron_module(w=w, b=b)
    spec = InputSpec.l2(value_name="input", center=torch.zeros(1, 2, dtype=torch.float32), eps=1.0)
    split = {"h1": torch.tensor([+1, +1, +1], dtype=torch.int8)}

    _bounds, _alpha, _beta, stats = run_alpha_beta_crown_mlp(module, spec, relu_split_state=split, steps=0, lr=0.2)
    assert stats.feasibility == "infeasible"
    assert stats.infeasible_certificate is not None
    assert float(stats.infeasible_certificate["max_value"]) < 0.0


def test_phase6f_beta_best_of_not_worse_than_step0() -> None:
    # DoD: best-of includes step=0, so optimization cannot regress.
    w = torch.tensor([[1.0]], dtype=torch.float32)
    b = torch.tensor([0.0], dtype=torch.float32)
    module = _make_first_layer_two_neuron_module(w=w, b=b)
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[0.0]], dtype=torch.float32), eps=1.0)
    split = {"h1": torch.tensor([+1], dtype=torch.int8)}

    b0, _a0, _beta0, s0 = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state=split,
        steps=0,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
        objective="lower",
    )
    assert s0.feasibility == "unknown"
    b1, _a1, _beta1, s1 = run_alpha_beta_crown_mlp(
        module,
        spec,
        relu_split_state=split,
        steps=10,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.1,
        objective="lower",
    )
    assert s1.feasibility == "unknown"
    assert float(b1.lower.item()) >= float(b0.lower.item()) - 1e-7


def test_phase6f_bab_alpha_beta_removes_need_for_1d_patch() -> None:
    # 1D toy: y = ReLU(x), x in [-1,1]. Property y>=0 is true.
    #
    # alpha-only BaB without the 1D input-restriction patch is NOT complete here:
    # after splitting active (x>=0), the leaf bound is still computed over x in [-1,1],
    # so the leaf lower bound can be negative and BaB may return UNSAFE.
    #
    # alpha-beta BaB can make split constraints first-class via beta encoding, recovering completeness
    # without relying on the patch.
    w1 = torch.tensor([[1.0]], dtype=torch.float32)
    b1 = torch.tensor([0.0], dtype=torch.float32)
    w2 = torch.tensor([[1.0]], dtype=torch.float32)
    b2 = torch.tensor([0.0], dtype=torch.float32)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["r1", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    module = BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}})
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[0.0]], dtype=torch.float32), eps=1.0)

    cfg_alpha = BabConfig(max_nodes=16, oracle="alpha", alpha_steps=0, alpha_init=0.5, threshold=0.0, tol=1e-8)
    res_alpha = solve_bab_mlp(module, spec, config=cfg_alpha)
    assert res_alpha.status == "unsafe"

    cfg_ab = BabConfig(max_nodes=16, oracle="alpha_beta", alpha_steps=40, alpha_lr=0.2, alpha_init=0.5, threshold=0.0, tol=1e-8)
    res_ab = solve_bab_mlp(module, spec, config=cfg_ab)
    assert res_ab.status == "proven"
