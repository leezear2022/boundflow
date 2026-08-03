"""Exact-box contracts required by VNNLIB fixed replay."""

from itertools import product

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner.materialization import BoundMethod
from boundflow.runtime.bab_query import make_bound_query
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.linear_operator import DenseLinearOperator
from boundflow.runtime.perturbation import BoxPerturbation
from boundflow.runtime.task_executor import InputSpec


def _linear_module(weight: torch.Tensor, bias: torch.Tensor) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="linear",
                name="linear0",
                inputs=["input", "weight", "bias"],
                outputs=["output"],
            )
        ],
        input_values=["input"],
        output_values=["output"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"weight": weight, "bias": bias}},
    )


def _box_vertices(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    rows = [
        [
            float(upper[0, index] if bit else lower[0, index])
            for index, bit in enumerate(bits)
        ]
        for bits in product((0, 1), repeat=int(lower.shape[1]))
    ]
    return torch.tensor(rows, dtype=lower.dtype, device=lower.device)


def test_box_concretize_matches_all_vertices_for_dense_and_lazy_affine() -> None:
    """Dense and lazy forms must equal exhaustive box-corner extrema."""

    lower = torch.tensor([[-1.0, 0.2, 2.0]], dtype=torch.float32)
    upper = torch.tensor([[0.5, 0.9, 2.1]], dtype=torch.float32)
    center = (lower + upper) / 2.0
    weight = torch.tensor([[1.0, -2.0, 0.5], [-0.3, 4.0, -1.0]], dtype=torch.float32)
    bias = torch.tensor([0.7, -0.2], dtype=torch.float32)
    perturbation = BoxPerturbation(lower=lower, upper=upper)

    expected = _box_vertices(lower, upper).matmul(weight.t()) + bias
    dense_lower, dense_upper = perturbation.concretize_matmul(
        center=center, weight=weight, bias=bias
    )
    operator = DenseLinearOperator(weight.unsqueeze(0))
    lazy_lower, lazy_upper = perturbation.concretize_affine(
        center=center, A=operator, b=bias.unsqueeze(0)
    )

    assert torch.allclose(dense_lower, expected.amin(dim=0, keepdim=True))
    assert torch.allclose(dense_upper, expected.amax(dim=0, keepdim=True))
    assert torch.equal(dense_lower, lazy_lower)
    assert torch.equal(dense_upper, lazy_upper)


def test_box_batched_matmul_matches_independent_boxes() -> None:
    """Each batch row keeps its own lower/upper vectors."""

    lower = torch.tensor([[-1.0, 0.0], [2.0, -3.0]], dtype=torch.float32)
    upper = torch.tensor([[0.0, 2.0], [2.5, 1.0]], dtype=torch.float32)
    center = (lower + upper) / 2.0
    weight = torch.tensor([[[1.0, -2.0]], [[-3.0, 0.5]]], dtype=torch.float32)
    perturbation = BoxPerturbation(lower=lower, upper=upper)

    actual_lower, actual_upper = perturbation.concretize_matmul(
        center=center, weight=weight
    )

    assert torch.allclose(actual_lower, torch.tensor([[-5.0], [-9.0]]))
    assert torch.allclose(actual_upper, torch.tensor([[0.0], [-5.5]]))


def test_box_owns_caller_bounds() -> None:
    """Later caller mutation must not change a frozen replay region."""

    lower = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)
    upper = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    perturbation = BoxPerturbation(lower=lower, upper=upper)
    lower.add_(10.0)
    upper.zero_()

    owned_lower, owned_upper = perturbation.bounding_box(
        torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    )
    assert torch.equal(owned_lower, torch.tensor([[-1.0, 0.0]]))
    assert torch.equal(owned_upper, torch.tensor([[1.0, 2.0]]))


def test_input_spec_box_runs_plain_crown_without_uniform_epsilon() -> None:
    """The plain-CROWN executor accepts non-uniform input radii."""

    weight = torch.tensor([[1.0, -2.0], [-3.0, 0.5]], dtype=torch.float32)
    bias = torch.tensor([0.1, -0.4], dtype=torch.float32)
    module = _linear_module(weight, bias)
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-1.0, 0.0]], dtype=torch.float32),
        upper=torch.tensor([[0.0, 2.0]], dtype=torch.float32),
    )

    bounds = run_crown_ibp_mlp(module, spec)

    assert torch.allclose(bounds.lower, torch.tensor([[-4.9, -0.4]]))
    assert torch.allclose(bounds.upper, torch.tensor([[0.1, 3.6]]))


def test_box_query_identity_changes_when_only_width_distribution_changes() -> None:
    """Same center/shape with different per-element radii is not reusable."""

    module = _linear_module(torch.eye(2), torch.zeros(2))
    lower_a = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
    upper_a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    lower_b = torch.tensor([[-0.5, -2.5]], dtype=torch.float32)
    upper_b = torch.tensor([[0.5, 2.5]], dtype=torch.float32)

    def build(spec: InputSpec, query_id: str):
        return make_bound_query(
            module=module,
            query_id=query_id,
            parent_query_id=None,
            sequence_number=0,
            example_idx=0,
            input_spec=spec,
            linear_spec_c=None,
            split_by_relu_input={},
            warm_alpha_by_relu_input={},
            warm_beta_by_relu_input={},
            bound_method=BoundMethod.CROWN,
            execution_options={},
        )[0]

    query_a = build(
        InputSpec.box(value_name="input", lower=lower_a, upper=upper_a), "a"
    )
    query_b = build(
        InputSpec.box(value_name="input", lower=lower_b, upper=upper_b), "b"
    )

    assert (
        query_a.compatibility_key.input_shape == query_b.compatibility_key.input_shape
    )
    assert query_a.input_region_hash != query_b.input_region_hash
    assert (
        query_a.compatibility_key.perturbation_signature
        != query_b.compatibility_key.perturbation_signature
    )


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (torch.zeros(1, 2), torch.zeros(1, 3), "shape mismatch"),
        (torch.tensor([[1.0]]), torch.tensor([[0.0]]), "lower <= upper"),
        (torch.tensor([[float("nan")]]), torch.tensor([[1.0]]), "finite"),
    ],
)
def test_box_rejects_invalid_bounds(
    lower: torch.Tensor, upper: torch.Tensor, message: str
) -> None:
    """Malformed boxes fail before query execution."""

    with pytest.raises(ValueError, match=message):
        BoxPerturbation(lower=lower, upper=upper)
