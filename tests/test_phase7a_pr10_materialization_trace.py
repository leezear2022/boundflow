import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.materialization import (
    TRACE_SCHEMA_VERSION,
    trace_materializations,
)
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

    with trace_materializations(
        run_id="unit-run",
        query_id="query-0",
        bound_method="CROWN",
        spec_batch=3,
        domain_batch=2,
    ) as trace:
        bounds = run_crown_ibp_mlp(
            module, InputSpec.linf(value_name="input", center=x0, eps=0.1)
        )

    assert tuple(bounds.lower.shape) == (2, 3)
    assert len(trace.events) == 2
    assert {event.operator_site for event in trace.events} == {"h1:upper", "h1:lower"}
    assert {event.reason for event in trace.events} == {"relu_sign_split"}
    assert {event.persistent_or_ephemeral for event in trace.events} == {"persistent"}
    assert {event.logical_lifetime_begin for event in trace.events} == {
        "relu_backward_step"
    }
    assert {event.logical_lifetime_end for event in trace.events} == {"backward_end"}
    assert {event.shape for event in trace.events} == {(2, 3, 5)}
    assert {event.logical_bytes for event in trace.events} == {2 * 3 * 5 * 4}
    assert {event.event_id for event in trace.events} == {0, 1}
    assert {event.operator_tree_depth for event in trace.events} == {2}
    assert {event.operator_node_count for event in trace.events} == {2}

    summary = trace.summary()
    assert summary["event_count"] == 2
    assert summary["logical_materialized_bytes"] == 2 * 2 * 3 * 5 * 4
    assert summary["observed_allocation_delta_bytes"] is None
    assert summary["by_reason"] == {
        "relu_sign_split": {"count": 2, "logical_bytes": 2 * 2 * 3 * 5 * 4}
    }

    record = trace.to_record()
    assert record["schema_version"] == TRACE_SCHEMA_VERSION
    assert record["run_id"] == "unit-run"
    assert record["query_id"] == "query-0"
    assert record["bound_method"] == "CROWN"
    assert record["spec_batch"] == 3
    assert record["domain_batch"] == 2
    assert set(record["state_bytes"]) == {
        "alpha_state_bytes",
        "beta_state_bytes",
        "intermediate_bound_bytes",
        "weight_bytes",
        "operator_state_bytes",
    }
    event_record = record["events"][0]
    assert event_record["schema_version"] == TRACE_SCHEMA_VERSION
    for key in (
        "operator_site",
        "source_value",
        "source_primal_op",
        "operator_tree_depth",
        "operator_node_count",
        "logical_bytes",
        "persistent_or_ephemeral",
        "requires_grad",
        "alpha_related",
        "beta_related",
    ):
        assert key in event_record


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
