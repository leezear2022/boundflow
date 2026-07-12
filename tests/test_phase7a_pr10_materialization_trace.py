import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.materialization import trace_materializations
from boundflow.runtime.task_executor import InputSpec


def _make_relu_mlp() -> BFTaskModule:
    torch.manual_seed(0)
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
                "W1": torch.randn(5, 4),
                "b1": torch.randn(5),
                "W2": torch.randn(3, 5),
                "b2": torch.randn(3),
            }
        },
    )


def test_relu_backward_materialization_trace_records_reason_bytes_and_lifetime() -> (
    None
):
    module = _make_relu_mlp()
    x0 = torch.randn(2, 4)

    with trace_materializations() as trace:
        bounds = run_crown_ibp_mlp(
            module, InputSpec.linf(value_name="input", center=x0, eps=0.1)
        )

    assert tuple(bounds.lower.shape) == (2, 3)
    assert len(trace.events) == 2
    assert {event.site for event in trace.events} == {"h1:upper", "h1:lower"}
    assert {event.reason for event in trace.events} == {"relu_sign_split"}
    assert {event.lifetime for event in trace.events} == {"relu_backward_step"}
    assert {event.logical_shape for event in trace.events} == {(2, 3, 5)}
    assert {event.dense_bytes for event in trace.events} == {2 * 3 * 5 * 4}

    summary = trace.summary()
    assert summary["count"] == 2
    assert summary["dense_bytes"] == 2 * 2 * 3 * 5 * 4
    assert summary["by_reason"] == {
        "relu_sign_split": {"count": 2, "dense_bytes": 2 * 2 * 3 * 5 * 4}
    }


def test_materialization_trace_scope_does_not_leak_between_runs() -> None:
    module = _make_relu_mlp()
    spec = InputSpec.linf(value_name="input", center=torch.randn(1, 4), eps=0.1)

    with trace_materializations() as first:
        run_crown_ibp_mlp(module, spec)
    with trace_materializations() as second:
        run_crown_ibp_mlp(module, spec)

    assert len(first.events) == 2
    assert len(second.events) == 2
    assert first is not second
