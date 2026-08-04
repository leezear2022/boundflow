"""Regression tests for box perturbations crossing input shape-only ops."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import itertools

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


def _flatten_first_module() -> BFTaskModule:
    task = BoundTask(
        task_id="flatten-first",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="flatten",
                name="flatten",
                inputs=["input"],
                outputs=["flat"],
                attrs={"start_dim": 1, "end_dim": -1},
            ),
            TaskOp(
                op_type="linear",
                name="linear1",
                inputs=["flat", "W1", "b1"],
                outputs=["hidden"],
            ),
            TaskOp(
                op_type="relu",
                name="relu1",
                inputs=["hidden"],
                outputs=["active"],
            ),
            TaskOp(
                op_type="linear",
                name="linear2",
                inputs=["active", "W2", "b2"],
                outputs=["output"],
            ),
        ],
        input_values=["input"],
        output_values=["output"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id=task.task_id,
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5, 0.25, 0.75], [-0.2, 0.4, 0.8, -1.0]]),
                "b1": torch.tensor([0.1, -0.2]),
                "W2": torch.tensor([[0.7, -0.3], [-0.4, 0.9]]),
                "b2": torch.tensor([0.05, -0.1]),
            }
        },
    )


def test_box_perturbation_is_shape_transformed_across_input_flatten() -> None:
    module = _flatten_first_module()
    lower = torch.tensor([[[-1.0, 0.0], [0.5, -0.25]]])
    upper = torch.tensor([[[0.5, 1.0], [1.5, 0.75]]])
    result = run_crown_ibp_mlp(
        module, InputSpec.box(value_name="input", lower=lower, upper=upper)
    )

    parameters = module.bindings["params"]
    assert isinstance(parameters, dict)
    concrete = []
    for choices in itertools.product((0, 1), repeat=4):
        flat = torch.tensor(
            [
                upper.reshape(-1)[index] if choice else lower.reshape(-1)[index]
                for index, choice in enumerate(choices)
            ]
        ).unsqueeze(0)
        hidden = flat @ parameters["W1"].t() + parameters["b1"]
        output = torch.relu(hidden) @ parameters["W2"].t() + parameters["b2"]
        concrete.append(output)
    values = torch.cat(concrete, dim=0)
    assert bool((values >= result.lower - 1e-6).all())
    assert bool((values <= result.upper + 1e-6).all())
