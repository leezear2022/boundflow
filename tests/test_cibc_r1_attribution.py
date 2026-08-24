"""Contract, replay, and re-signed tamper tests for CIBC R1 attribution."""

# pylint: disable=missing-function-docstring,consider-using-dict-items

from __future__ import annotations

import copy
from typing import Any, cast

import pytest

from boundflow.ir.task import BoundTask, TaskKind, TaskOp
from boundflow.runtime.cibc_r1_attribution import (
    CalibrationTriplet,
    R1AttributionMethod,
    R1OpType,
    R1OwnedEvent,
    R1OwnerLedger,
    R1RouteReceipt,
    R1Scope,
    R1ScopedOpportunity,
    R1SpeedupSource,
    R1TargetContract,
    R1TimingLedger,
    canonical_hash,
    clock_calibration_from_dict,
    derive_clock_calibration,
    owner_ledger_from_dict,
    projected_b0_query_ratio,
    required_region_speedup,
    route_receipt_from_dict,
    target_contract_from_dict,
    timing_ledger_from_dict,
    topology_from_task,
    topology_ledger_from_dict,
)


def _task() -> BoundTask:
    return BoundTask(
        task_id="r1-test",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp("conv2d", "conv0", ["input", "weight", "bias"], ["conv"]),
            TaskOp("relu", "relu0", ["conv"], ["relu"]),
            TaskOp("flatten", "flatten0", ["relu"], ["flat"]),
            TaskOp("linear", "linear0", ["flat", "head_w", "head_b"], ["output"]),
        ],
        input_values=["input"],
        output_values=["output"],
        params=["weight", "bias", "head_w", "head_b"],
    )


def _topology(*, single_stream: bool = True):
    task = _task()
    shapes = {
        "input": (1, 3, 4, 4),
        "weight": (2, 3, 3, 3),
        "bias": (2,),
        "conv": (1, 2, 2, 2),
        "relu": (1, 2, 2, 2),
        "flat": (1, 8),
        "head_w": (3, 8),
        "head_b": (3,),
        "output": (1, 3),
    }
    dtypes = {name: "torch.float32" for name in shapes}
    devices = {name: "cuda:0" for name in shapes}
    return topology_from_task(
        task,
        external_values=("input", "weight", "bias", "head_w", "head_b"),
        value_shapes=shapes,
        value_dtypes=dtypes,
        value_devices=devices,
        single_stream=single_stream,
    )


def _triplets() -> tuple[CalibrationTriplet, ...]:
    rows = []
    for phase, base in (("before", 1_000_000_000), ("after", 2_000_000_000)):
        for ordinal in range(64):
            gpu = base + ordinal * 100_000
            rows.append(
                CalibrationTriplet(
                    phase=phase,
                    ordinal=ordinal,
                    host_before_ns=gpu + 99_500,
                    gpu_timestamp_ns=gpu,
                    host_after_ns=gpu + 100_500,
                )
            )
    return tuple(rows)


def _owner_ledger(*, single_stream: bool = True) -> R1OwnerLedger:
    topology = _topology(single_stream=single_stream)
    events = []
    for ordinal, node in enumerate(topology.nodes):
        events.append(
            R1OwnedEvent(
                event_ordinal=ordinal,
                name=f"kernel-{ordinal}",
                owner_marker=topology.marker_for(ordinal),
                op_type=node.op_type,
                op_ordinal=ordinal,
                start_host_ns=1_000 + ordinal * 100,
                end_host_ns=1_080 + ordinal * 100,
                stream_id=7,
                correlation_id=100 + ordinal,
                attribution_method=R1AttributionMethod.CORRELATION_PARENT,
                shape_source="correlation_parent",
                parent_digest=canonical_hash(node.to_dict()),
            )
        )
    return R1OwnerLedger(
        topology=topology,
        scope_start_host_ns=900,
        scope_end_host_ns=1_500,
        events=tuple(events),
    )


def _timing() -> R1TimingLedger:
    buckets = {item: 0 for item in R1OpType}
    buckets[R1OpType.CIBC_CONV] = 300_000
    buckets[R1OpType.LINEAR] = 200_000
    buckets[R1OpType.RELU] = 100_000
    buckets[R1OpType.FLATTEN_VIEW] = 10_000
    buckets[R1OpType.GRAPH_RUNTIME_SYNC] = 390_000
    return R1TimingLedger(
        graph_wall_ns=1_000_000,
        kernel_sum_ns=550_000,
        exclusive_by_bucket_ns=buckets,
        critical_path_ns=1_000_000,
        overlap_adjusted_wall_ns=1_000_000,
        overlap_interval_count=0,
        single_stream=True,
    )


def _route() -> R1RouteReceipt:
    return R1RouteReceipt(
        current_b3_to_b0_ratio=0.91,
        opportunities=(
            R1ScopedOpportunity(
                scope=R1Scope.COMPLETE_QUERY,
                op_type=R1OpType.CIBC_CONV,
                share=0.36,
                region_speedup=2.0,
                speedup_source=R1SpeedupSource.QUERY_LOCAL,
                admitted=True,
            ),
            R1ScopedOpportunity(
                scope=R1Scope.COMPLETE_QUERY,
                op_type=R1OpType.LINEAR,
                share=0.10,
                region_speedup=1.0,
                speedup_source=R1SpeedupSource.UNAVAILABLE,
                admitted=False,
            ),
        ),
    )


def test_target_contract_replays_and_scope_targets_do_not_mix() -> None:
    contract = target_contract_from_dict(R1TargetContract().to_dict())
    assert contract.target_for(R1Scope.COMPLETE_QUERY, research=False) == 1.0
    assert contract.target_for(R1Scope.COMPLETE_QUERY, research=True) == 1.15
    assert contract.target_for(R1Scope.QUEUE_BAB, research=True) == 1.20
    with pytest.raises(ValueError, match="target scope"):
        contract.target_for(R1Scope.WHOLE_GRAPH, research=True)


def test_target_contract_rejects_changed_target_and_claim() -> None:
    payload = R1TargetContract().to_dict()
    payload["query_research"] = 1.10
    with pytest.raises(ValueError, match="target contract differs"):
        target_contract_from_dict(payload)
    payload = R1TargetContract().to_dict()
    payload["performance_claimed"] = True
    with pytest.raises(ValueError, match="target contract differs"):
        target_contract_from_dict(payload)


def test_topology_round_trip_binds_ordinals_shapes_and_markers() -> None:
    topology = _topology()
    rebuilt = topology_ledger_from_dict(topology.to_dict())
    assert rebuilt.topology_hash == topology.topology_hash
    assert [node.op_type for node in rebuilt.nodes] == [
        R1OpType.CIBC_CONV,
        R1OpType.RELU,
        R1OpType.FLATTEN_VIEW,
        R1OpType.LINEAR,
    ]
    assert rebuilt.marker_for(0).startswith("boundflow.r1/graph/0/cibc_conv/")


@pytest.mark.parametrize("field", ["ordinal", "output_shapes", "shape_source"])
def test_topology_rejects_resigned_source_identity_tamper(field: str) -> None:
    payload = _topology().to_dict()
    node = payload["nodes"][0]
    if field == "ordinal":
        node[field] = 2
    elif field == "output_shapes":
        node[field] = [[1, 2, 3, 3]]
    else:
        node[field] = "temporal_guess"
    identity = {
        key: payload[key] for key in payload if key not in {"topology_hash", "markers"}
    }
    payload["topology_hash"] = canonical_hash(identity)
    with pytest.raises(ValueError):
        topology_ledger_from_dict(payload)


def test_topology_rejects_unsupported_op_before_profiling() -> None:
    task = _task()
    task.ops[0].op_type = "average_pool2d"
    with pytest.raises(ValueError, match="op unsupported"):
        topology_from_task(
            task,
            external_values=("input", "weight", "bias", "head_w", "head_b"),
            value_shapes={},
            value_dtypes={},
            value_devices={},
            single_stream=True,
        )


def test_clock_calibration_round_trip_is_formally_admitted() -> None:
    receipt = derive_clock_calibration(_triplets(), nsys_anchor_errors_ns=(0, 100, 200))
    assert receipt.cupti_admitted
    assert receipt.formal_admitted
    rebuilt = clock_calibration_from_dict(receipt.to_dict())
    assert rebuilt.to_dict() == receipt.to_dict()


@pytest.mark.parametrize(
    ("field", "value"),
    [("slope", 1.5), ("offset_ns", 0.0), ("formal_admitted", False)],
)
def test_clock_calibration_rejects_resigned_summary_tamper(
    field: str, value: object
) -> None:
    payload = derive_clock_calibration(
        _triplets(), nsys_anchor_errors_ns=(0, 100, 200)
    ).to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match="calibration derivation"):
        clock_calibration_from_dict(payload)


def test_clock_calibration_without_nsys_is_smoke_only() -> None:
    receipt = derive_clock_calibration(_triplets())
    assert receipt.cupti_admitted
    assert not receipt.formal_admitted


def test_clock_calibration_rejects_nonmonotonic_timestamps() -> None:
    rows = list(_triplets())
    rows[2] = CalibrationTriplet(
        phase="before",
        ordinal=2,
        host_before_ns=rows[1].host_before_ns - 10,
        gpu_timestamp_ns=rows[1].gpu_timestamp_ns - 10,
        host_after_ns=rows[1].host_after_ns - 10,
    )
    with pytest.raises(ValueError, match="timestamps do not advance"):
        derive_clock_calibration(rows, nsys_anchor_errors_ns=(0, 100, 200))


def test_owner_ledger_round_trip_and_resigned_parent_tamper() -> None:
    ledger = _owner_ledger()
    payload = ledger.to_dict()
    assert owner_ledger_from_dict(payload).to_dict() == payload
    events = cast(list[dict[str, Any]], payload["events"])
    events[0]["owner_marker"] = ledger.topology.marker_for(1)
    unsigned = {
        key: value for key, value in payload.items() if key != "owner_ledger_hash"
    }
    payload["owner_ledger_hash"] = canonical_hash(unsigned)
    with pytest.raises(ValueError, match="source ownership"):
        owner_ledger_from_dict(payload)


@pytest.mark.parametrize("field", ["unowned_event_count", "temporal_fallback_count"])
def test_owner_ledger_rejects_unowned_or_temporal_fallback(field: str) -> None:
    payload = _owner_ledger().to_dict()
    payload[field] = 1
    unsigned = {
        key: value for key, value in payload.items() if key != "owner_ledger_hash"
    }
    payload["owner_ledger_hash"] = canonical_hash(unsigned)
    with pytest.raises(ValueError, match="admission differs"):
        owner_ledger_from_dict(payload)


def test_owner_ledger_rejects_multiple_streams_when_single_stream() -> None:
    ledger = _owner_ledger()
    events = list(ledger.events)
    events[1] = R1OwnedEvent(**{**events[1].__dict__, "stream_id": 8})
    with pytest.raises(ValueError, match="single-stream"):
        R1OwnerLedger(
            topology=ledger.topology,
            scope_start_host_ns=ledger.scope_start_host_ns,
            scope_end_host_ns=ledger.scope_end_host_ns,
            events=tuple(events),
        ).validate()


def test_single_stream_timing_round_trip_and_no_overlap_degeneration() -> None:
    payload = _timing().to_dict()
    assert timing_ledger_from_dict(payload).to_dict() == payload
    payload["overlap_adjusted_wall_ns"] = 950_000
    with pytest.raises(ValueError, match="single-stream degeneration"):
        timing_ledger_from_dict(payload)


def test_query_projection_uses_only_disjoint_query_local_speedups() -> None:
    route = _route()
    expected = 0.91 / ((1.0 - 0.46) + 0.36 / 2.0 + 0.10)
    assert route.projected_b0_query_ratio == pytest.approx(expected)
    assert projected_b0_query_ratio(
        current_b3_to_b0_ratio=0.91, opportunities=route.opportunities
    ) == pytest.approx(expected)


def test_query_projection_rejects_graph_scope_and_historical_speedup() -> None:
    with pytest.raises(ValueError, match="independent graph speedup"):
        R1ScopedOpportunity(
            scope=R1Scope.COMPLETE_QUERY,
            op_type=R1OpType.CIBC_CONV,
            share=0.36,
            region_speedup=2.45631,
            speedup_source=R1SpeedupSource.HISTORICAL_INDEPENDENT,
            admitted=False,
        ).validate()
    graph_row = R1ScopedOpportunity(
        scope=R1Scope.WHOLE_GRAPH,
        op_type=R1OpType.CIBC_CONV,
        share=0.36,
        region_speedup=2.45631,
        speedup_source=R1SpeedupSource.HISTORICAL_INDEPENDENT,
        admitted=False,
    )
    with pytest.raises(ValueError, match="projection scope"):
        projected_b0_query_ratio(
            current_b3_to_b0_ratio=0.91, opportunities=(graph_row,)
        )


def test_unavailable_candidate_is_forced_to_unit_speedup() -> None:
    with pytest.raises(ValueError, match="must be conservative"):
        R1ScopedOpportunity(
            scope=R1Scope.COMPLETE_QUERY,
            op_type=R1OpType.LINEAR,
            share=0.10,
            region_speedup=1.1,
            speedup_source=R1SpeedupSource.UNAVAILABLE,
            admitted=False,
        ).validate()


def test_route_receipt_replays_and_rejects_outer_resigned_verdict() -> None:
    payload = _route().to_dict()
    assert route_receipt_from_dict(payload).to_dict() == payload
    payload["qualification_go"] = not payload["qualification_go"]
    unsigned = {
        key: value for key, value in payload.items() if key != "route_receipt_hash"
    }
    payload["route_receipt_hash"] = canonical_hash(unsigned)
    with pytest.raises(ValueError, match="route derivation"):
        route_receipt_from_dict(payload)


def test_required_region_speedup_reports_mathematical_impossibility() -> None:
    assert required_region_speedup(share=0.36, target=1.15) == pytest.approx(
        0.36 / (1.0 / 1.15 - 0.64)
    )
    assert required_region_speedup(share=0.07, target=1.20) is None


def test_unknown_field_is_rejected_even_when_payload_is_resigned() -> None:
    payload = copy.deepcopy(_route().to_dict())
    payload["unexpected"] = "resigned"
    unsigned = {
        key: value for key, value in payload.items() if key != "route_receipt_hash"
    }
    payload["route_receipt_hash"] = canonical_hash(unsigned)
    with pytest.raises(ValueError, match="fields differ"):
        route_receipt_from_dict(payload)
