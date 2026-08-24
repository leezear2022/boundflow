"""Nsight Systems SQLite replay for CIBC R1-A graph-node attribution."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,line-too-long

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Mapping, Sequence

from boundflow.runtime.cibc_r1_attribution import (
    R1AttributionMethod,
    R1OpType,
    R1OwnedEvent,
    R1OwnerLedger,
    R1TimingLedger,
    canonical_hash,
    calibration_triplet_from_dict,
    clock_calibration_from_dict,
    derive_clock_calibration,
    topology_ledger_from_dict,
)

NSYS_EXPORT_SCHEMA = "boundflow.cibc-r1-nsys-export/v1"
EXPECTED_GRAPH_NODES_BY_OP = {
    R1OpType.CIBC_CONV: 1,
    R1OpType.LINEAR: 10,
    R1OpType.RELU: 2,
    R1OpType.RESIDUAL_ADD: 2,
    R1OpType.FLATTEN_VIEW: 0,
}


@dataclass(frozen=True)
class R1NsightExportReceipt:
    """Frozen counts and ownership evidence rebuilt from one Nsight export."""

    anchor_errors_ns: tuple[int, ...]
    graph_node_count: int
    cloned_graph_node_count: int
    profile_group_count: int
    replay_count: int
    kernel_count: int
    memcpy_count: int
    runtime_api_count: int
    graph_launch_count: int
    stream_ids: tuple[int, ...]
    unowned_event_count: int
    temporal_fallback_count: int
    graph_node_owner_hash: str
    formal_admitted: bool
    schema_version: str = NSYS_EXPORT_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != NSYS_EXPORT_SCHEMA
            or len(self.anchor_errors_ns) < 3
            or any(value < 0 for value in self.anchor_errors_ns)
            or self.graph_node_count != 42
            or self.cloned_graph_node_count < 42
            or self.profile_group_count != 20
            or self.replay_count != 100
            or self.kernel_count != 4_200
            or self.memcpy_count != 200
            or self.runtime_api_count <= 0
            or self.graph_launch_count != 100
            or len(self.stream_ids) != 1
            or self.unowned_event_count != 0
            or self.temporal_fallback_count != 0
            or len(self.graph_node_owner_hash) != 64
            or not isinstance(self.formal_admitted, bool)
        ):
            raise ValueError("R1 Nsight export receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "anchor_errors_ns": list(self.anchor_errors_ns),
            "graph_node_count": self.graph_node_count,
            "cloned_graph_node_count": self.cloned_graph_node_count,
            "profile_group_count": self.profile_group_count,
            "replay_count": self.replay_count,
            "kernel_count": self.kernel_count,
            "memcpy_count": self.memcpy_count,
            "runtime_api_count": self.runtime_api_count,
            "graph_launch_count": self.graph_launch_count,
            "stream_ids": list(self.stream_ids),
            "unowned_event_count": self.unowned_event_count,
            "temporal_fallback_count": self.temporal_fallback_count,
            "graph_node_owner_hash": self.graph_node_owner_hash,
            "formal_admitted": self.formal_admitted,
        }


def nsys_export_receipt_from_dict(value: object) -> R1NsightExportReceipt:
    """Parse and validate a serialized Nsight export receipt."""

    if not isinstance(value, Mapping):
        raise ValueError("R1 Nsight export receipt mapping differs")
    expected = {
        "schema_version",
        "anchor_errors_ns",
        "graph_node_count",
        "cloned_graph_node_count",
        "profile_group_count",
        "replay_count",
        "kernel_count",
        "memcpy_count",
        "runtime_api_count",
        "graph_launch_count",
        "stream_ids",
        "unowned_event_count",
        "temporal_fallback_count",
        "graph_node_owner_hash",
        "formal_admitted",
    }
    if set(value) != expected:
        raise ValueError("R1 Nsight export receipt fields differ")
    if not isinstance(value["formal_admitted"], bool):
        raise ValueError("R1 Nsight export receipt admission differs")
    try:
        receipt = R1NsightExportReceipt(
            schema_version=str(value["schema_version"]),
            anchor_errors_ns=tuple(int(item) for item in value["anchor_errors_ns"]),
            graph_node_count=int(value["graph_node_count"]),
            cloned_graph_node_count=int(value["cloned_graph_node_count"]),
            profile_group_count=int(value["profile_group_count"]),
            replay_count=int(value["replay_count"]),
            kernel_count=int(value["kernel_count"]),
            memcpy_count=int(value["memcpy_count"]),
            runtime_api_count=int(value["runtime_api_count"]),
            graph_launch_count=int(value["graph_launch_count"]),
            stream_ids=tuple(int(item) for item in value["stream_ids"]),
            unowned_event_count=int(value["unowned_event_count"]),
            temporal_fallback_count=int(value["temporal_fallback_count"]),
            graph_node_owner_hash=str(value["graph_node_owner_hash"]),
            formal_admitted=bool(value["formal_admitted"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("R1 Nsight export receipt values differ") from error
    receipt.validate()
    if receipt.to_dict() != dict(value):
        raise ValueError("R1 Nsight export receipt derivation differs")
    return receipt


def _fit_anchor_errors(xs: Sequence[float], ys: Sequence[int]) -> tuple[int, ...]:
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("R1 Nsight anchor inventory differs")
    x_center = statistics.fmean(xs)
    y_center = statistics.fmean(ys)
    denominator = sum((value - x_center) ** 2 for value in xs)
    if denominator <= 0.0:
        raise ValueError("R1 Nsight anchor timestamps do not advance")
    slope = sum((x - x_center) * (y - y_center) for x, y in zip(xs, ys)) / denominator
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("R1 Nsight anchor slope differs")
    return tuple(
        round(abs(y - (y_center + slope * (x - x_center)))) for x, y in zip(xs, ys)
    )


def _string_expression() -> str:
    return "coalesce(n.text,s.value)"


def _ranges(connection: sqlite3.Connection, pattern: str) -> list[tuple[int, int, str]]:
    rows = connection.execute(
        f"""
        select n.start,n.end,{_string_expression()}
        from NVTX_EVENTS n left join StringIds s on n.textId=s.id
        where {_string_expression()} like ? order by n.start
        """,
        (pattern,),
    ).fetchall()
    if any(end is None or int(end) <= int(start) for start, end, _label in rows):
        raise ValueError("R1 Nsight range interval differs")
    return [(int(start), int(end), str(label)) for start, end, label in rows]


def _contained(start: int, end: int, ranges: Sequence[tuple[int, int, str]]) -> bool:
    return any(left <= start and end <= right for left, right, _label in ranges)


def derive_nsys_attribution(
    sqlite_path: Path, worker: Mapping[str, Any]
) -> tuple[R1NsightExportReceipt, R1OwnerLedger, R1TimingLedger, dict[str, object]]:
    """Rebuild calibration, graph-node owners, and timing from SQLite raw."""

    inventory_value = worker.get("profile_inventory")
    if (
        worker.get("mode") != "profile"
        or not isinstance(inventory_value, Mapping)
        or inventory_value.get("backend") not in {"nsys_pending_export", "nsys_sqlite"}
    ):
        raise ValueError("R1 Nsight worker boundary differs")
    topology = topology_ledger_from_dict(worker.get("topology"))
    calibration = clock_calibration_from_dict(worker.get("calibration_receipt"))
    inventory = inventory_value
    anchors = inventory.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 3:
        raise ValueError("R1 Nsight raw anchor inventory differs")
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        anchor_rows = connection.execute(f"""
            select n.start,{_string_expression()}
            from NVTX_EVENTS n left join StringIds s on n.textId=s.id
            where {_string_expression()} like 'boundflow.r1/calibration-anchor/%'
            order by n.start
            """).fetchall()
        expected_anchor_labels = [str(item["marker"]) for item in anchors]
        if [str(label) for _start, label in anchor_rows] != expected_anchor_labels:
            raise ValueError("R1 Nsight anchor label differs")
        anchor_errors = _fit_anchor_errors(
            [float(item["gpu_timestamp_ns"]) for item in anchors],
            [int(start) for start, _label in anchor_rows],
        )
        raw_triplets = calibration.to_dict()["triplets"]
        if not isinstance(raw_triplets, list):
            raise ValueError("R1 Nsight calibration raw differs")
        rebuilt_calibration = derive_clock_calibration(
            tuple(calibration_triplet_from_dict(item) for item in raw_triplets),
            nsys_anchor_errors_ns=anchor_errors,
            thresholds=calibration.thresholds,
        )
        marker_ranges = _ranges(connection, "boundflow.r1/graph/%")
        expected_marker_counts = Counter(
            topology.marker_for(node.ordinal) for node in topology.nodes
        )
        observed_marker_counts = Counter(label for _start, _end, label in marker_ranges)
        if observed_marker_counts != Counter(
            {label: count * 4 for label, count in expected_marker_counts.items()}
        ):
            raise ValueError("R1 Nsight capture marker inventory differs")

        graph_owner: dict[int, int] = {}
        for left, right, label in marker_ranges:
            ordinal = int(label.split("/")[2])
            node_ids = connection.execute(
                "select distinct graphNodeId from CUDA_GRAPH_NODE_EVENTS where start>=? and start<=?",
                (left, right),
            ).fetchall()
            for (node_id,) in node_ids:
                identifier = int(node_id)
                previous = graph_owner.setdefault(identifier, ordinal)
                if previous != ordinal:
                    raise ValueError("R1 Nsight graph node owner collision")
        counts = Counter(
            topology.nodes[ordinal].op_type for ordinal in graph_owner.values()
        )
        expected_counts = Counter(
            {
                op_type: sum(
                    EXPECTED_GRAPH_NODES_BY_OP[node.op_type]
                    for node in topology.nodes
                    if node.op_type == op_type
                )
                for op_type in EXPECTED_GRAPH_NODES_BY_OP
            }
        )
        if counts != expected_counts:
            raise ValueError("R1 Nsight graph-node topology differs")

        clone_rows = connection.execute(
            "select graphNodeId,originalGraphNodeId from CUDA_GRAPH_NODE_EVENTS where originalGraphNodeId is not null"
        ).fetchall()
        clone_to_original = {int(node): int(original) for node, original in clone_rows}
        group_ranges = _ranges(connection, "boundflow.r1/profile-group/%")
        if [int(label.split("/")[2]) for _left, _right, label in group_ranges] != list(
            range(20)
        ):
            raise ValueError("R1 Nsight profile group order differs")

        kernel_rows = connection.execute("""
            select k.start,k.end,k.streamId,k.correlationId,k.graphNodeId,s.value
            from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id
            order by k.start
            """).fetchall()
        memcpy_rows = connection.execute(
            "select start,end,streamId,correlationId,bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start"
        ).fetchall()
        kernels = [
            row
            for row in kernel_rows
            if _contained(int(row[0]), int(row[1]), group_ranges)
        ]
        memcpys = [
            row
            for row in memcpy_rows
            if _contained(int(row[0]), int(row[1]), group_ranges)
        ]
        replay_count = 20 * 5
        if len(kernels) != 42 * replay_count or len(memcpys) != 2 * replay_count:
            raise ValueError("R1 Nsight steady event inventory differs")

        events: list[R1OwnedEvent] = []
        bucket_ns = {item: 0 for item in R1OpType}
        stream_ids: set[int] = set()
        unowned = 0
        for start, end, stream, correlation, graph_node, name in kernels:
            original = clone_to_original.get(int(graph_node), int(graph_node))
            owner_ordinal = graph_owner.get(original)
            if owner_ordinal is None:
                unowned += 1
                continue
            node = topology.nodes[owner_ordinal]
            duration = int(end) - int(start)
            bucket_ns[node.op_type] += duration
            stream_ids.add(int(stream))
            events.append(
                R1OwnedEvent(
                    event_ordinal=len(events),
                    name=str(name),
                    owner_marker=topology.marker_for(owner_ordinal),
                    op_type=node.op_type,
                    op_ordinal=owner_ordinal,
                    start_host_ns=int(start),
                    end_host_ns=int(end),
                    stream_id=int(stream),
                    correlation_id=int(correlation or 0),
                    attribution_method=R1AttributionMethod.GRAPH_NODE,
                    shape_source="correlation_parent",
                    parent_digest=canonical_hash(node.to_dict()),
                )
            )
        runtime_marker = (
            f"boundflow.r1/runtime/input-copy/{topology.topology_hash[:12]}"
        )
        for start, end, stream, correlation, _bytes in memcpys:
            duration = int(end) - int(start)
            bucket_ns[R1OpType.INPUT_COPY] += duration
            stream_ids.add(int(stream))
            events.append(
                R1OwnedEvent(
                    event_ordinal=len(events),
                    name="cuda_memcpy_input",
                    owner_marker=runtime_marker,
                    op_type=R1OpType.INPUT_COPY,
                    op_ordinal=None,
                    start_host_ns=int(start),
                    end_host_ns=int(end),
                    stream_id=int(stream),
                    correlation_id=int(correlation or 0),
                    attribution_method=R1AttributionMethod.RUNTIME_SCOPE,
                    shape_source="runtime_scope",
                    parent_digest=None,
                )
            )
        events.sort(
            key=lambda event: (event.start_host_ns, event.end_host_ns, event.name)
        )
        events = [
            R1OwnedEvent(**{**event.__dict__, "event_ordinal": ordinal})
            for ordinal, event in enumerate(events)
        ]
        scope_start = min(left for left, _right, _label in group_ranges)
        scope_end = max(right for _left, right, _label in group_ranges)
        owner_ledger = R1OwnerLedger(
            topology=topology,
            scope_start_host_ns=scope_start,
            scope_end_host_ns=scope_end,
            events=tuple(events),
            unowned_event_count=unowned,
            temporal_fallback_count=0,
        )
        owner_ledger.validate()

        graph_wall = sum(right - left for left, right, _label in group_ranges)
        device_owned = sum(bucket_ns.values())
        if device_owned > graph_wall:
            raise ValueError("R1 Nsight device wall exceeds profile scope")
        bucket_ns[R1OpType.GRAPH_RUNTIME_SYNC] = graph_wall - device_owned
        timing = R1TimingLedger(
            graph_wall_ns=graph_wall,
            kernel_sum_ns=sum(int(end) - int(start) for start, end, *_rest in kernels),
            exclusive_by_bucket_ns=bucket_ns,
            critical_path_ns=graph_wall,
            overlap_adjusted_wall_ns=graph_wall,
            overlap_interval_count=0,
            single_stream=len(stream_ids) == 1,
            unowned_event_count=unowned,
        )
        timing.validate()

        runtime_rows = connection.execute("""
            select r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r
            join StringIds s on r.nameId=s.id order by r.start
            """).fetchall()
        runtimes = [
            row
            for row in runtime_rows
            if _contained(int(row[0]), int(row[1]), group_ranges)
        ]
        graph_launch_count = sum(
            str(row[2]).startswith("cudaGraphLaunch") for row in runtimes
        )
        owner_payload = {
            str(identifier): ordinal
            for identifier, ordinal in sorted(graph_owner.items())
        }
        receipt = R1NsightExportReceipt(
            anchor_errors_ns=anchor_errors,
            graph_node_count=len(graph_owner),
            cloned_graph_node_count=len(clone_to_original),
            profile_group_count=len(group_ranges),
            replay_count=replay_count,
            kernel_count=len(kernels),
            memcpy_count=len(memcpys),
            runtime_api_count=len(runtimes),
            graph_launch_count=graph_launch_count,
            stream_ids=tuple(sorted(stream_ids)),
            unowned_event_count=unowned,
            temporal_fallback_count=0,
            graph_node_owner_hash=canonical_hash(owner_payload),
            formal_admitted=rebuilt_calibration.formal_admitted,
        )
        receipt.validate()
        return receipt, owner_ledger, timing, rebuilt_calibration.to_dict()
    finally:
        connection.close()


__all__ = [
    "R1NsightExportReceipt",
    "derive_nsys_attribution",
    "nsys_export_receipt_from_dict",
]
