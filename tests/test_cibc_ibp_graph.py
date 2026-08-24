"""Contract tests for prepared CIBC IBP CUDA graph plans."""

# pylint: disable=missing-function-docstring

from contextlib import contextmanager

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.cibc_ibp_graph import (
    CIBCIBPCUDAGraphPlanV1,
    run_cibc_ibp_graph_once_v1,
)


def test_cibc_ibp_cuda_graph_plan_is_exported() -> None:
    assert CIBCIBPCUDAGraphPlanV1.__name__ == "CIBCIBPCUDAGraphPlanV1"


def test_cibc_ibp_operator_context_is_additive_and_opt_in() -> None:
    task = BoundTask(
        task_id="marker-test",
        kind=TaskKind.INTERVAL_IBP,
        ops=[TaskOp("relu", "relu0", ["input"], ["output"])],
        input_values=["input"],
        output_values=["output"],
    )
    module = BFTaskModule(tasks=[task], entry_task_id=task.task_id)
    observed: list[tuple[str, int, str]] = []

    @contextmanager
    def marker(ordinal: int, op_type: str):
        observed.append(("enter", ordinal, op_type))
        yield
        observed.append(("exit", ordinal, op_type))

    lower = torch.tensor([-1.0, 2.0])
    upper = torch.tensor([1.0, 3.0])
    plain, plain_launches = run_cibc_ibp_graph_once_v1(
        module,
        input_value="input",
        input_lower=lower,
        input_upper=upper,
        threads_per_block=None,
    )
    marked, marked_launches = run_cibc_ibp_graph_once_v1(
        module,
        input_value="input",
        input_lower=lower,
        input_upper=upper,
        threads_per_block=None,
        op_context_factory=marker,
    )
    assert observed == [("enter", 0, "relu"), ("exit", 0, "relu")]
    assert plain_launches == marked_launches == 0
    assert torch.equal(plain["output"].lower, marked["output"].lower)
    assert torch.equal(plain["output"].upper, marked["output"].upper)
