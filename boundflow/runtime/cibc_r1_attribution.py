"""Fail-closed contracts for CIBC R1 clock, topology, and route attribution."""

# pylint: disable=missing-function-docstring,too-few-public-methods,too-many-lines
# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-boolean-expressions

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import statistics
import time
from typing import Any, Mapping, Sequence

from boundflow.ir.task import BoundTask

R1_SCHEMA_VERSION = "boundflow.cibc-r1-attribution/v1"
R1_MARKER_PREFIX = "boundflow.r1"
QUERY_QUALIFICATION_TARGET = 1.0
QUERY_RESEARCH_TARGET = 1.15
QUEUE_RESEARCH_TARGET = 1.20
CALIBRATION_SAMPLE_INTERVAL_NS = 1_000_000


class R1Scope(str, Enum):
    """Timing scopes that must never share an Amdahl denominator."""

    WHOLE_GRAPH = "whole_graph"
    COMPLETE_QUERY = "complete_query"
    QUEUE_BAB = "queue_bab"


class R1OpType(str, Enum):
    """Frozen mutually exclusive R1 attribution buckets."""

    INPUT_COPY = "input_copy_lower_upper"
    CIBC_CONV = "cibc_conv"
    LINEAR = "linear"
    RELU = "relu"
    RESIDUAL_ADD = "residual_add"
    FLATTEN_VIEW = "flatten_view"
    GRAPH_RUNTIME_SYNC = "graph_launch_runtime_sync"
    UNOWNED = "unowned"


class R1AttributionMethod(str, Enum):
    """Only explicit ownership methods admitted by R1 v1."""

    CORRELATION_PARENT = "correlation_parent"
    GRAPH_NODE = "graph_node"
    RUNTIME_SCOPE = "runtime_scope"
    UNOWNED = "unowned"


class R1SpeedupSource(str, Enum):
    """Physical source of one region speedup."""

    QUERY_LOCAL = "query_local"
    HISTORICAL_INDEPENDENT = "historical_independent"
    UNAVAILABLE = "unavailable"


def canonical_json(value: object) -> str:
    """Encode finite JSON with stable separators and key order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    """Return the stable SHA256 used by all R1 receipts."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"R1 {label} fields differ")


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"R1 {label} mapping differs")
    return value


def _as_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"R1 {label} list differs")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"R1 {label} string differs")
    return value


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"R1 {label} bool differs")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"R1 {label} integer differs")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"R1 {label} float differs")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"R1 {label} finite value differs")
    return result


def _as_optional_int(value: object, label: str) -> int | None:
    return None if value is None else _as_int(value, label)


def _as_optional_str(value: object, label: str) -> str | None:
    return None if value is None else _as_str(value, label)


def _enum_value(enum_type: type[Enum], value: object, label: str) -> Any:
    try:
        return enum_type(_as_str(value, label))
    except ValueError as error:
        raise ValueError(f"R1 {label} enum differs") from error


@dataclass(frozen=True)
class R1TargetContract:
    """Frozen B0-relative targets for query and queue scopes."""

    query_qualification: float = QUERY_QUALIFICATION_TARGET
    query_research: float = QUERY_RESEARCH_TARGET
    queue_research: float = QUEUE_RESEARCH_TARGET
    performance_claimed: bool = False
    schema_version: str = R1_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != R1_SCHEMA_VERSION
            or self.query_qualification != QUERY_QUALIFICATION_TARGET
            or self.query_research != QUERY_RESEARCH_TARGET
            or self.queue_research != QUEUE_RESEARCH_TARGET
            or self.performance_claimed
        ):
            raise ValueError("R1 target contract differs")

    def target_for(self, scope: R1Scope, *, research: bool) -> float:
        self.validate()
        if scope == R1Scope.COMPLETE_QUERY:
            return self.query_research if research else self.query_qualification
        if scope == R1Scope.QUEUE_BAB and research:
            return self.queue_research
        raise ValueError("R1 target scope differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "query_qualification": self.query_qualification,
            "query_research": self.query_research,
            "queue_research": self.queue_research,
            "performance_claimed": False,
        }


def target_contract_from_dict(value: object) -> R1TargetContract:
    """Parse and revalidate a target contract without trusting serialized values."""

    row = _as_mapping(value, "target contract")
    _exact_keys(
        row,
        {
            "schema_version",
            "query_qualification",
            "query_research",
            "queue_research",
            "performance_claimed",
        },
        "target contract",
    )
    contract = R1TargetContract(
        schema_version=_as_str(row["schema_version"], "target schema"),
        query_qualification=_as_float(
            row["query_qualification"], "qualification target"
        ),
        query_research=_as_float(row["query_research"], "query research target"),
        queue_research=_as_float(row["queue_research"], "queue research target"),
        performance_claimed=_as_bool(row["performance_claimed"], "target claim"),
    )
    contract.validate()
    return contract


_TASK_OP_BUCKETS = {
    "conv2d": R1OpType.CIBC_CONV,
    "linear": R1OpType.LINEAR,
    "relu": R1OpType.RELU,
    "add": R1OpType.RESIDUAL_ADD,
    "flatten": R1OpType.FLATTEN_VIEW,
    "reshape": R1OpType.FLATTEN_VIEW,
    "view": R1OpType.FLATTEN_VIEW,
}


@dataclass(frozen=True)
class R1TopologyNode:
    """One source-plan node with shapes recovered before profiling."""

    ordinal: int
    name: str
    op_type: R1OpType
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    dtype: str
    device: str
    shape_source: str = "source_plan"

    def validate(self) -> None:
        if (
            self.ordinal < 0
            or not self.name
            or self.op_type
            in {R1OpType.INPUT_COPY, R1OpType.GRAPH_RUNTIME_SYNC, R1OpType.UNOWNED}
            or not self.inputs
            or not self.outputs
            or len(self.input_shapes) != len(self.inputs)
            or len(self.output_shapes) != len(self.outputs)
            or not self.dtype
            or not self.device
            or self.shape_source != "source_plan"
            or any(
                any(dimension < 0 for dimension in shape)
                for shape in (*self.input_shapes, *self.output_shapes)
            )
        ):
            raise ValueError("R1 topology node differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "op_type": self.op_type.value,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "dtype": self.dtype,
            "device": self.device,
            "shape_source": self.shape_source,
        }


@dataclass(frozen=True)
class R1TopologyLedger:
    """Canonical source topology and marker identity."""

    task_id: str
    external_values: tuple[str, ...]
    nodes: tuple[R1TopologyNode, ...]
    single_stream: bool
    schema_version: str = R1_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != R1_SCHEMA_VERSION
            or not self.task_id
            or not self.nodes
        ):
            raise ValueError("R1 topology ledger identity differs")
        if len(set(self.external_values)) != len(self.external_values):
            raise ValueError("R1 topology external values duplicate")
        produced: set[str] = set(self.external_values)
        for expected_ordinal, node in enumerate(self.nodes):
            node.validate()
            if node.ordinal != expected_ordinal:
                raise ValueError("R1 topology ordinal differs")
            if any(value not in produced for value in node.inputs):
                raise ValueError("R1 topology input ownership differs")
            if any(value in produced for value in node.outputs):
                raise ValueError("R1 topology output ownership differs")
            produced.update(node.outputs)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "external_values": list(self.external_values),
            "nodes": [node.to_dict() for node in self.nodes],
            "single_stream": self.single_stream,
        }

    @property
    def topology_hash(self) -> str:
        self.validate()
        return canonical_hash(self._identity_payload())

    def marker_for(self, ordinal: int) -> str:
        self.validate()
        if ordinal not in range(len(self.nodes)):
            raise ValueError("R1 marker ordinal differs")
        node = self.nodes[ordinal]
        return f"{R1_MARKER_PREFIX}/graph/{ordinal}/{node.op_type.value}/{self.topology_hash[:12]}"

    def to_dict(self) -> dict[str, object]:
        payload = self._identity_payload()
        payload["topology_hash"] = self.topology_hash
        payload["markers"] = [self.marker_for(node.ordinal) for node in self.nodes]
        return payload


def topology_from_task(
    task: BoundTask,
    *,
    external_values: Sequence[str],
    value_shapes: Mapping[str, Sequence[int]],
    value_dtypes: Mapping[str, str],
    value_devices: Mapping[str, str],
    single_stream: bool,
) -> R1TopologyLedger:
    """Build the frozen topology only from source-plan values and observed shapes."""

    nodes: list[R1TopologyNode] = []
    for ordinal, op in enumerate(task.ops):
        bucket = _TASK_OP_BUCKETS.get(op.op_type)
        if bucket is None:
            raise ValueError(f"R1 topology op unsupported: {op.op_type}")
        values = (*op.inputs, *op.outputs)
        if any(
            value not in value_shapes
            or value not in value_dtypes
            or value not in value_devices
            for value in values
        ):
            raise ValueError("R1 topology value metadata missing")
        dtypes = {value_dtypes[value] for value in values}
        devices = {value_devices[value] for value in values}
        if len(dtypes) != 1 or len(devices) != 1:
            raise ValueError("R1 topology dtype/device ownership differs")
        nodes.append(
            R1TopologyNode(
                ordinal=ordinal,
                name=op.name,
                op_type=bucket,
                inputs=tuple(op.inputs),
                outputs=tuple(op.outputs),
                input_shapes=tuple(
                    tuple(int(item) for item in value_shapes[value])
                    for value in op.inputs
                ),
                output_shapes=tuple(
                    tuple(int(item) for item in value_shapes[value])
                    for value in op.outputs
                ),
                dtype=next(iter(dtypes)),
                device=next(iter(devices)),
            )
        )
    ledger = R1TopologyLedger(
        task_id=task.task_id,
        external_values=tuple(external_values),
        nodes=tuple(nodes),
        single_stream=single_stream,
    )
    ledger.validate()
    return ledger


def _shape_tuple(value: object, label: str) -> tuple[tuple[int, ...], ...]:
    shapes = _as_list(value, label)
    return tuple(
        tuple(_as_int(item, f"{label} dimension") for item in _as_list(shape, label))
        for shape in shapes
    )


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_as_str(item, label) for item in _as_list(value, label))


def topology_node_from_dict(value: object) -> R1TopologyNode:
    """Parse one topology node with exact-field enforcement."""

    row = _as_mapping(value, "topology node")
    _exact_keys(
        row,
        {
            "ordinal",
            "name",
            "op_type",
            "inputs",
            "outputs",
            "input_shapes",
            "output_shapes",
            "dtype",
            "device",
            "shape_source",
        },
        "topology node",
    )
    node = R1TopologyNode(
        ordinal=_as_int(row["ordinal"], "topology ordinal"),
        name=_as_str(row["name"], "topology name"),
        op_type=_enum_value(R1OpType, row["op_type"], "topology op type"),
        inputs=_string_tuple(row["inputs"], "topology inputs"),
        outputs=_string_tuple(row["outputs"], "topology outputs"),
        input_shapes=_shape_tuple(row["input_shapes"], "topology input shapes"),
        output_shapes=_shape_tuple(row["output_shapes"], "topology output shapes"),
        dtype=_as_str(row["dtype"], "topology dtype"),
        device=_as_str(row["device"], "topology device"),
        shape_source=_as_str(row["shape_source"], "topology shape source"),
    )
    node.validate()
    return node


def topology_ledger_from_dict(value: object) -> R1TopologyLedger:
    """Rebuild topology identity and reject re-signed derived-field tampering."""

    row = _as_mapping(value, "topology ledger")
    _exact_keys(
        row,
        {
            "schema_version",
            "task_id",
            "external_values",
            "nodes",
            "single_stream",
            "topology_hash",
            "markers",
        },
        "topology ledger",
    )
    ledger = R1TopologyLedger(
        schema_version=_as_str(row["schema_version"], "topology schema"),
        task_id=_as_str(row["task_id"], "topology task"),
        external_values=_string_tuple(
            row["external_values"], "topology external values"
        ),
        nodes=tuple(
            topology_node_from_dict(item)
            for item in _as_list(row["nodes"], "topology nodes")
        ),
        single_stream=_as_bool(row["single_stream"], "topology stream mode"),
    )
    rebuilt = ledger.to_dict()
    if canonical_json(row) != canonical_json(rebuilt):
        raise ValueError("R1 topology derivation differs")
    return ledger


@dataclass(frozen=True)
class CalibrationThresholds:
    """Pre-registered clock-domain admission thresholds."""

    minimum_samples_per_phase: int = 64
    p95_bracket_width_ns: int = 2_000
    maximum_bracket_width_ns: int = 10_000
    maximum_fit_residual_ns: int = 2_000
    maximum_slope_drift_ppm: float = 100.0
    maximum_anchor_drift_ns: int = 2_000
    minimum_nsys_anchors: int = 3
    maximum_nsys_anchor_error_ns: int = 2_000

    def validate(self) -> None:
        if self != CalibrationThresholds() or any(
            value <= 0
            for value in (
                self.minimum_samples_per_phase,
                self.p95_bracket_width_ns,
                self.maximum_bracket_width_ns,
                self.maximum_fit_residual_ns,
                self.maximum_slope_drift_ppm,
                self.maximum_anchor_drift_ns,
                self.minimum_nsys_anchors,
                self.maximum_nsys_anchor_error_ns,
            )
        ):
            raise ValueError("R1 calibration thresholds differ")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "minimum_samples_per_phase": self.minimum_samples_per_phase,
            "p95_bracket_width_ns": self.p95_bracket_width_ns,
            "maximum_bracket_width_ns": self.maximum_bracket_width_ns,
            "maximum_fit_residual_ns": self.maximum_fit_residual_ns,
            "maximum_slope_drift_ppm": self.maximum_slope_drift_ppm,
            "maximum_anchor_drift_ns": self.maximum_anchor_drift_ns,
            "minimum_nsys_anchors": self.minimum_nsys_anchors,
            "maximum_nsys_anchor_error_ns": self.maximum_nsys_anchor_error_ns,
        }


@dataclass(frozen=True)
class CalibrationTriplet:
    """One NTP-style host/CUPTI timestamp bracket."""

    phase: str
    ordinal: int
    host_before_ns: int
    gpu_timestamp_ns: int
    host_after_ns: int

    def validate(self) -> None:
        if (
            self.phase not in {"before", "after"}
            or self.ordinal < 0
            or self.host_before_ns <= 0
            or self.gpu_timestamp_ns <= 0
            or self.host_after_ns < self.host_before_ns
        ):
            raise ValueError("R1 calibration triplet differs")

    @property
    def host_midpoint_ns(self) -> float:
        self.validate()
        return (self.host_before_ns + self.host_after_ns) / 2.0

    @property
    def bracket_width_ns(self) -> int:
        self.validate()
        return self.host_after_ns - self.host_before_ns

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "phase": self.phase,
            "ordinal": self.ordinal,
            "host_before_ns": self.host_before_ns,
            "gpu_timestamp_ns": self.gpu_timestamp_ns,
            "host_after_ns": self.host_after_ns,
        }


@dataclass(frozen=True)
class _AffineFit:
    slope: float
    offset_ns: float
    x_center: float
    y_center: float
    maximum_residual_ns: float

    def predict(self, gpu_timestamp_ns: float) -> float:
        return self.y_center + self.slope * (gpu_timestamp_ns - self.x_center)


def _fit_triplets(rows: Sequence[CalibrationTriplet]) -> _AffineFit:
    if len(rows) < 2:
        raise ValueError("R1 calibration fit requires two triplets")
    for row in rows:
        row.validate()
    xs = [float(row.gpu_timestamp_ns) for row in rows]
    ys = [row.host_midpoint_ns for row in rows]
    x_center = statistics.fmean(xs)
    y_center = statistics.fmean(ys)
    denominator = sum((value - x_center) ** 2 for value in xs)
    if denominator <= 0.0:
        raise ValueError("R1 calibration GPU timestamps do not advance")
    slope = sum((x - x_center) * (y - y_center) for x, y in zip(xs, ys)) / denominator
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("R1 calibration slope differs")
    offset = y_center - slope * x_center
    maximum_residual = max(
        abs(y - (y_center + slope * (x - x_center))) for x, y in zip(xs, ys)
    )
    return _AffineFit(slope, offset, x_center, y_center, maximum_residual)


def _percentile95(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("R1 percentile input empty")
    ordered = sorted(values)
    return ordered[max(math.ceil(0.95 * len(ordered)) - 1, 0)]


@dataclass(frozen=True)
class ClockCalibrationReceipt:
    """Raw-bound clock mapping and Nsight export admission."""

    triplets: tuple[CalibrationTriplet, ...]
    nsys_anchor_errors_ns: tuple[int, ...]
    slope: float
    offset_ns: float
    p95_bracket_width_ns: int
    maximum_bracket_width_ns: int
    maximum_fit_residual_ns: float
    slope_drift_ppm: float
    anchor_drift_ns: float
    cupti_admitted: bool
    formal_admitted: bool
    thresholds: CalibrationThresholds = CalibrationThresholds()
    schema_version: str = R1_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != R1_SCHEMA_VERSION:
            raise ValueError("R1 calibration schema differs")
        rebuilt = derive_clock_calibration(
            self.triplets,
            nsys_anchor_errors_ns=self.nsys_anchor_errors_ns,
            thresholds=self.thresholds,
        )
        if self.to_dict(revalidate=False) != rebuilt.to_dict(revalidate=False):
            raise ValueError("R1 calibration derivation differs")

    def to_dict(self, *, revalidate: bool = True) -> dict[str, object]:
        if revalidate:
            self.validate()
        return {
            "schema_version": self.schema_version,
            "triplets": [row.to_dict() for row in self.triplets],
            "nsys_anchor_errors_ns": list(self.nsys_anchor_errors_ns),
            "slope": self.slope,
            "offset_ns": self.offset_ns,
            "p95_bracket_width_ns": self.p95_bracket_width_ns,
            "maximum_bracket_width_ns": self.maximum_bracket_width_ns,
            "maximum_fit_residual_ns": self.maximum_fit_residual_ns,
            "slope_drift_ppm": self.slope_drift_ppm,
            "anchor_drift_ns": self.anchor_drift_ns,
            "cupti_admitted": self.cupti_admitted,
            "formal_admitted": self.formal_admitted,
            "thresholds": self.thresholds.to_dict(),
        }


def derive_clock_calibration(
    triplets: Sequence[CalibrationTriplet],
    *,
    nsys_anchor_errors_ns: Sequence[int] = (),
    thresholds: CalibrationThresholds = CalibrationThresholds(),
) -> ClockCalibrationReceipt:
    """Fit and admit a raw host/CUPTI calibration without trusting summaries."""

    thresholds.validate()
    rows = tuple(triplets)
    before = tuple(row for row in rows if row.phase == "before")
    after = tuple(row for row in rows if row.phase == "after")
    if any(
        row.ordinal != ordinal
        for phase in (before, after)
        for ordinal, row in enumerate(phase)
    ):
        raise ValueError("R1 calibration ordinal differs")
    if len(before) < 2 or len(after) < 2:
        raise ValueError("R1 calibration phase inventory differs")
    for phase in (before, after):
        if any(
            current.gpu_timestamp_ns <= previous.gpu_timestamp_ns
            or current.host_midpoint_ns <= previous.host_midpoint_ns
            for previous, current in zip(phase, phase[1:])
        ):
            raise ValueError("R1 calibration timestamps do not advance")
    combined_fit = _fit_triplets(rows)
    before_fit = _fit_triplets(before)
    after_fit = _fit_triplets(after)
    widths = [row.bracket_width_ns for row in rows]
    mean_slope = (before_fit.slope + after_fit.slope) / 2.0
    slope_drift_ppm = abs(before_fit.slope - after_fit.slope) / mean_slope * 1.0e6
    reference_gpu = statistics.median(row.gpu_timestamp_ns for row in rows)
    anchor_drift = abs(
        before_fit.predict(reference_gpu) - after_fit.predict(reference_gpu)
    )
    anchors = tuple(int(value) for value in nsys_anchor_errors_ns)
    if any(value < 0 for value in anchors):
        raise ValueError("R1 Nsight anchor error differs")
    cupti_admitted = (
        len(before) >= thresholds.minimum_samples_per_phase
        and len(after) >= thresholds.minimum_samples_per_phase
        and _percentile95(widths) <= thresholds.p95_bracket_width_ns
        and max(widths) <= thresholds.maximum_bracket_width_ns
        and combined_fit.maximum_residual_ns <= thresholds.maximum_fit_residual_ns
        and slope_drift_ppm <= thresholds.maximum_slope_drift_ppm
        and anchor_drift <= thresholds.maximum_anchor_drift_ns
    )
    formal_admitted = (
        cupti_admitted
        and len(anchors) >= thresholds.minimum_nsys_anchors
        and max(anchors, default=thresholds.maximum_nsys_anchor_error_ns + 1)
        <= thresholds.maximum_nsys_anchor_error_ns
    )
    return ClockCalibrationReceipt(
        triplets=rows,
        nsys_anchor_errors_ns=anchors,
        slope=combined_fit.slope,
        offset_ns=combined_fit.offset_ns,
        p95_bracket_width_ns=_percentile95(widths),
        maximum_bracket_width_ns=max(widths),
        maximum_fit_residual_ns=combined_fit.maximum_residual_ns,
        slope_drift_ppm=slope_drift_ppm,
        anchor_drift_ns=anchor_drift,
        cupti_admitted=cupti_admitted,
        formal_admitted=formal_admitted,
        thresholds=thresholds,
    )


def calibration_thresholds_from_dict(value: object) -> CalibrationThresholds:
    """Parse the immutable R1 v1 calibration thresholds."""

    row = _as_mapping(value, "calibration thresholds")
    _exact_keys(row, set(CalibrationThresholds().to_dict()), "calibration thresholds")
    thresholds = CalibrationThresholds(
        minimum_samples_per_phase=_as_int(
            row["minimum_samples_per_phase"], "minimum calibration samples"
        ),
        p95_bracket_width_ns=_as_int(row["p95_bracket_width_ns"], "p95 bracket"),
        maximum_bracket_width_ns=_as_int(
            row["maximum_bracket_width_ns"], "maximum bracket"
        ),
        maximum_fit_residual_ns=_as_int(
            row["maximum_fit_residual_ns"], "maximum fit residual"
        ),
        maximum_slope_drift_ppm=_as_float(
            row["maximum_slope_drift_ppm"], "maximum slope drift"
        ),
        maximum_anchor_drift_ns=_as_int(
            row["maximum_anchor_drift_ns"], "maximum anchor drift"
        ),
        minimum_nsys_anchors=_as_int(
            row["minimum_nsys_anchors"], "minimum Nsight anchors"
        ),
        maximum_nsys_anchor_error_ns=_as_int(
            row["maximum_nsys_anchor_error_ns"], "maximum Nsight anchor error"
        ),
    )
    thresholds.validate()
    return thresholds


def calibration_triplet_from_dict(value: object) -> CalibrationTriplet:
    """Parse one raw host/CUPTI timestamp bracket."""

    row = _as_mapping(value, "calibration triplet")
    _exact_keys(
        row,
        {
            "phase",
            "ordinal",
            "host_before_ns",
            "gpu_timestamp_ns",
            "host_after_ns",
        },
        "calibration triplet",
    )
    triplet = CalibrationTriplet(
        phase=_as_str(row["phase"], "calibration phase"),
        ordinal=_as_int(row["ordinal"], "calibration ordinal"),
        host_before_ns=_as_int(row["host_before_ns"], "calibration host before"),
        gpu_timestamp_ns=_as_int(row["gpu_timestamp_ns"], "calibration GPU timestamp"),
        host_after_ns=_as_int(row["host_after_ns"], "calibration host after"),
    )
    triplet.validate()
    return triplet


def clock_calibration_from_dict(value: object) -> ClockCalibrationReceipt:
    """Recompute every calibration summary and admission bit from raw triplets."""

    row = _as_mapping(value, "clock calibration")
    _exact_keys(
        row,
        {
            "schema_version",
            "triplets",
            "nsys_anchor_errors_ns",
            "slope",
            "offset_ns",
            "p95_bracket_width_ns",
            "maximum_bracket_width_ns",
            "maximum_fit_residual_ns",
            "slope_drift_ppm",
            "anchor_drift_ns",
            "cupti_admitted",
            "formal_admitted",
            "thresholds",
        },
        "clock calibration",
    )
    thresholds = calibration_thresholds_from_dict(row["thresholds"])
    rebuilt = derive_clock_calibration(
        tuple(
            calibration_triplet_from_dict(item)
            for item in _as_list(row["triplets"], "calibration triplets")
        ),
        nsys_anchor_errors_ns=tuple(
            _as_int(item, "Nsight anchor error")
            for item in _as_list(row["nsys_anchor_errors_ns"], "Nsight anchor errors")
        ),
        thresholds=thresholds,
    )
    if _as_str(row["schema_version"], "calibration schema") != R1_SCHEMA_VERSION:
        raise ValueError("R1 calibration schema differs")
    if canonical_json(row) != canonical_json(rebuilt.to_dict(revalidate=False)):
        raise ValueError("R1 calibration derivation differs")
    return rebuilt


class CuptiTimestampSource:
    """Minimal ctypes binding for ``cuptiGetTimestamp``."""

    def __init__(self, library_path: str | None = None) -> None:
        resolved = library_path or ctypes.util.find_library("cupti")
        if not resolved:
            raise RuntimeError("R1 CUPTI timestamp library unavailable")
        self.library_path = resolved
        self._library = ctypes.CDLL(resolved)
        self._function = self._library.cuptiGetTimestamp
        self._function.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self._function.restype = ctypes.c_int

    def timestamp_ns(self) -> int:
        value = ctypes.c_uint64()
        result = int(self._function(ctypes.byref(value)))
        if result != 0 or value.value <= 0:
            raise RuntimeError(f"R1 cuptiGetTimestamp failed: {result}")
        return int(value.value)


def collect_cupti_triplets(
    source: CuptiTimestampSource,
    *,
    phase: str,
    count: int = 64,
    sample_interval_ns: int = CALIBRATION_SAMPLE_INTERVAL_NS,
) -> tuple[CalibrationTriplet, ...]:
    """Collect raw native triplets; derivation remains a separate step."""

    if phase not in {"before", "after"} or count < 2 or sample_interval_ns < 0:
        raise ValueError("R1 CUPTI collection request differs")
    source.timestamp_ns()
    rows = []
    for ordinal in range(count):
        host_before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        gpu_timestamp = source.timestamp_ns()
        host_after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        rows.append(
            CalibrationTriplet(
                phase=phase,
                ordinal=ordinal,
                host_before_ns=host_before,
                gpu_timestamp_ns=gpu_timestamp,
                host_after_ns=host_after,
            )
        )
        if ordinal + 1 < count and sample_interval_ns:
            target_ns = host_after + sample_interval_ns
            while time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) < target_ns:
                pass
    return tuple(rows)


@dataclass(frozen=True)
class R1OwnedEvent:
    """One event assigned by explicit correlation, graph-node, or runtime scope."""

    event_ordinal: int
    name: str
    owner_marker: str
    op_type: R1OpType
    op_ordinal: int | None
    start_host_ns: int
    end_host_ns: int
    stream_id: int
    correlation_id: int
    attribution_method: R1AttributionMethod
    shape_source: str
    parent_digest: str | None

    def validate(self, topology: R1TopologyLedger) -> None:
        topology.validate()
        if (
            self.event_ordinal < 0
            or not self.name
            or self.start_host_ns < 0
            or self.end_host_ns <= self.start_host_ns
            or self.stream_id < -1
            or self.correlation_id < 0
        ):
            raise ValueError("R1 owned event interval differs")
        if (
            self.op_type == R1OpType.UNOWNED
            or self.attribution_method == R1AttributionMethod.UNOWNED
        ):
            raise ValueError("R1 unowned event forbidden")
        if self.op_ordinal is None:
            if (
                self.op_type not in {R1OpType.INPUT_COPY, R1OpType.GRAPH_RUNTIME_SYNC}
                or self.attribution_method != R1AttributionMethod.RUNTIME_SCOPE
                or self.shape_source != "runtime_scope"
                or self.parent_digest is not None
            ):
                raise ValueError("R1 runtime event ownership differs")
            return
        if self.op_ordinal not in range(len(topology.nodes)):
            raise ValueError("R1 event op ordinal differs")
        node = topology.nodes[self.op_ordinal]
        if (
            self.op_type != node.op_type
            or self.owner_marker != topology.marker_for(self.op_ordinal)
            or self.attribution_method
            not in {
                R1AttributionMethod.CORRELATION_PARENT,
                R1AttributionMethod.GRAPH_NODE,
            }
            or self.shape_source != "correlation_parent"
            or self.parent_digest is None
            or not _valid_digest(self.parent_digest)
        ):
            raise ValueError("R1 event source ownership differs")

    def to_dict(self, topology: R1TopologyLedger) -> dict[str, object]:
        self.validate(topology)
        return {
            "event_ordinal": self.event_ordinal,
            "name": self.name,
            "owner_marker": self.owner_marker,
            "op_type": self.op_type.value,
            "op_ordinal": self.op_ordinal,
            "start_host_ns": self.start_host_ns,
            "end_host_ns": self.end_host_ns,
            "stream_id": self.stream_id,
            "correlation_id": self.correlation_id,
            "attribution_method": self.attribution_method.value,
            "shape_source": self.shape_source,
            "parent_digest": self.parent_digest,
        }


@dataclass(frozen=True)
class R1OwnerLedger:
    """Complete event ownership for one calibrated graph scope."""

    topology: R1TopologyLedger
    scope_start_host_ns: int
    scope_end_host_ns: int
    events: tuple[R1OwnedEvent, ...]
    unowned_event_count: int = 0
    temporal_fallback_count: int = 0
    schema_version: str = R1_SCHEMA_VERSION

    def validate(self) -> None:
        self.topology.validate()
        if (
            self.schema_version != R1_SCHEMA_VERSION
            or self.scope_start_host_ns < 0
            or self.scope_end_host_ns <= self.scope_start_host_ns
            or not self.events
            or self.unowned_event_count != 0
            or self.temporal_fallback_count != 0
        ):
            raise ValueError("R1 owner ledger admission differs")
        streams: set[int] = set()
        for ordinal, event in enumerate(self.events):
            event.validate(self.topology)
            if event.event_ordinal != ordinal:
                raise ValueError("R1 owner event ordinal differs")
            if not (
                self.scope_start_host_ns <= event.start_host_ns
                and event.end_host_ns <= self.scope_end_host_ns
            ):
                raise ValueError("R1 owner event containment differs")
            if event.stream_id >= 0:
                streams.add(event.stream_id)
        if self.topology.single_stream and len(streams) > 1:
            raise ValueError("R1 single-stream ownership differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "topology": self.topology.to_dict(),
            "scope_start_host_ns": self.scope_start_host_ns,
            "scope_end_host_ns": self.scope_end_host_ns,
            "events": [event.to_dict(self.topology) for event in self.events],
            "unowned_event_count": self.unowned_event_count,
            "temporal_fallback_count": self.temporal_fallback_count,
        }
        payload["owner_ledger_hash"] = canonical_hash(payload)
        return payload


def owned_event_from_dict(value: object) -> R1OwnedEvent:
    """Parse one raw event; owner validation is performed against its topology."""

    row = _as_mapping(value, "owned event")
    _exact_keys(
        row,
        {
            "event_ordinal",
            "name",
            "owner_marker",
            "op_type",
            "op_ordinal",
            "start_host_ns",
            "end_host_ns",
            "stream_id",
            "correlation_id",
            "attribution_method",
            "shape_source",
            "parent_digest",
        },
        "owned event",
    )
    return R1OwnedEvent(
        event_ordinal=_as_int(row["event_ordinal"], "event ordinal"),
        name=_as_str(row["name"], "event name"),
        owner_marker=_as_str(row["owner_marker"], "event owner marker"),
        op_type=_enum_value(R1OpType, row["op_type"], "event op type"),
        op_ordinal=_as_optional_int(row["op_ordinal"], "event op ordinal"),
        start_host_ns=_as_int(row["start_host_ns"], "event start"),
        end_host_ns=_as_int(row["end_host_ns"], "event end"),
        stream_id=_as_int(row["stream_id"], "event stream"),
        correlation_id=_as_int(row["correlation_id"], "event correlation"),
        attribution_method=_enum_value(
            R1AttributionMethod, row["attribution_method"], "event attribution method"
        ),
        shape_source=_as_str(row["shape_source"], "event shape source"),
        parent_digest=_as_optional_str(row["parent_digest"], "event parent digest"),
    )


def owner_ledger_from_dict(value: object) -> R1OwnerLedger:
    """Rebuild event ownership and reject outer-digest re-signing attacks."""

    row = _as_mapping(value, "owner ledger")
    _exact_keys(
        row,
        {
            "schema_version",
            "topology",
            "scope_start_host_ns",
            "scope_end_host_ns",
            "events",
            "unowned_event_count",
            "temporal_fallback_count",
            "owner_ledger_hash",
        },
        "owner ledger",
    )
    ledger = R1OwnerLedger(
        schema_version=_as_str(row["schema_version"], "owner schema"),
        topology=topology_ledger_from_dict(row["topology"]),
        scope_start_host_ns=_as_int(row["scope_start_host_ns"], "owner scope start"),
        scope_end_host_ns=_as_int(row["scope_end_host_ns"], "owner scope end"),
        events=tuple(
            owned_event_from_dict(item)
            for item in _as_list(row["events"], "owner events")
        ),
        unowned_event_count=_as_int(row["unowned_event_count"], "unowned event count"),
        temporal_fallback_count=_as_int(
            row["temporal_fallback_count"], "temporal fallback count"
        ),
    )
    rebuilt = ledger.to_dict()
    if canonical_json(row) != canonical_json(rebuilt):
        raise ValueError("R1 owner ledger derivation differs")
    return ledger


@dataclass(frozen=True)
class R1TimingLedger:
    """Four separated timing views with single-stream degeneration checks."""

    graph_wall_ns: int
    kernel_sum_ns: int
    exclusive_by_bucket_ns: Mapping[R1OpType, int]
    critical_path_ns: int
    overlap_adjusted_wall_ns: int
    overlap_interval_count: int
    single_stream: bool
    unowned_event_count: int = 0

    def validate(self) -> None:
        expected = set(R1OpType)
        if set(self.exclusive_by_bucket_ns) != expected:
            raise ValueError("R1 timing bucket inventory differs")
        values = tuple(int(self.exclusive_by_bucket_ns[item]) for item in R1OpType)
        if (
            self.graph_wall_ns <= 0
            or self.kernel_sum_ns < 0
            or self.critical_path_ns <= 0
            or self.overlap_adjusted_wall_ns <= 0
            or self.overlap_interval_count < 0
            or self.unowned_event_count != 0
            or any(value < 0 for value in values)
            or self.exclusive_by_bucket_ns[R1OpType.UNOWNED] != 0
        ):
            raise ValueError("R1 timing ledger values differ")
        exclusive_wall = sum(values)
        tolerance = max(1_000, round(0.02 * self.graph_wall_ns))
        if exclusive_wall > self.graph_wall_ns + tolerance:
            raise ValueError("R1 exclusive wall exceeds graph scope")
        if self.single_stream or self.overlap_interval_count == 0:
            if (
                abs(exclusive_wall - self.critical_path_ns) > tolerance
                or abs(self.overlap_adjusted_wall_ns - self.critical_path_ns)
                > tolerance
            ):
                raise ValueError("R1 single-stream degeneration differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "graph_wall_ns": self.graph_wall_ns,
            "kernel_sum_ns": self.kernel_sum_ns,
            "exclusive_by_bucket_ns": {
                item.value: int(self.exclusive_by_bucket_ns[item]) for item in R1OpType
            },
            "exclusive_wall_ns": sum(
                int(self.exclusive_by_bucket_ns[item]) for item in R1OpType
            ),
            "critical_path_ns": self.critical_path_ns,
            "overlap_adjusted_wall_ns": self.overlap_adjusted_wall_ns,
            "overlap_interval_count": self.overlap_interval_count,
            "single_stream": self.single_stream,
            "unowned_event_count": self.unowned_event_count,
        }


def timing_ledger_from_dict(value: object) -> R1TimingLedger:
    """Parse timing views and recompute the exclusive wall summary."""

    row = _as_mapping(value, "timing ledger")
    _exact_keys(
        row,
        {
            "graph_wall_ns",
            "kernel_sum_ns",
            "exclusive_by_bucket_ns",
            "exclusive_wall_ns",
            "critical_path_ns",
            "overlap_adjusted_wall_ns",
            "overlap_interval_count",
            "single_stream",
            "unowned_event_count",
        },
        "timing ledger",
    )
    raw_buckets = _as_mapping(row["exclusive_by_bucket_ns"], "timing buckets")
    if set(raw_buckets) != {item.value for item in R1OpType}:
        raise ValueError("R1 timing bucket inventory differs")
    ledger = R1TimingLedger(
        graph_wall_ns=_as_int(row["graph_wall_ns"], "graph wall"),
        kernel_sum_ns=_as_int(row["kernel_sum_ns"], "kernel sum"),
        exclusive_by_bucket_ns={
            item: _as_int(raw_buckets[item.value], f"{item.value} wall")
            for item in R1OpType
        },
        critical_path_ns=_as_int(row["critical_path_ns"], "critical path"),
        overlap_adjusted_wall_ns=_as_int(
            row["overlap_adjusted_wall_ns"], "overlap-adjusted wall"
        ),
        overlap_interval_count=_as_int(
            row["overlap_interval_count"], "overlap interval count"
        ),
        single_stream=_as_bool(row["single_stream"], "timing stream mode"),
        unowned_event_count=_as_int(row["unowned_event_count"], "timing unowned count"),
    )
    rebuilt = ledger.to_dict()
    if canonical_json(row) != canonical_json(rebuilt):
        raise ValueError("R1 timing ledger derivation differs")
    return ledger


@dataclass(frozen=True)
class R1ScopedOpportunity:
    """One disjoint baseline-side share and its physical local speedup."""

    scope: R1Scope
    op_type: R1OpType
    share: float
    region_speedup: float
    speedup_source: R1SpeedupSource
    admitted: bool

    def validate(self) -> None:
        if (
            not math.isfinite(self.share)
            or not 0.0 <= self.share < 1.0
            or not math.isfinite(self.region_speedup)
            or self.region_speedup <= 0.0
            or self.op_type == R1OpType.UNOWNED
        ):
            raise ValueError("R1 scoped opportunity values differ")
        if (
            self.scope == R1Scope.COMPLETE_QUERY
            and self.speedup_source == R1SpeedupSource.HISTORICAL_INDEPENDENT
        ):
            raise ValueError("R1 independent graph speedup cannot enter query scope")
        if self.speedup_source == R1SpeedupSource.UNAVAILABLE and (
            self.region_speedup != 1.0 or self.admitted
        ):
            raise ValueError("R1 unavailable speedup must be conservative")
        if self.admitted != (self.speedup_source == R1SpeedupSource.QUERY_LOCAL):
            raise ValueError("R1 query-local admission differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "scope": self.scope.value,
            "op_type": self.op_type.value,
            "share": self.share,
            "region_speedup": self.region_speedup,
            "speedup_source": self.speedup_source.value,
            "admitted": self.admitted,
        }


def required_region_speedup(*, share: float, target: float) -> float | None:
    """Invert Amdahl for one timing scope; ``None`` means impossible."""

    if (
        not math.isfinite(share)
        or not 0.0 <= share < 1.0
        or not math.isfinite(target)
        or target <= 0.0
    ):
        raise ValueError("R1 required speedup input differs")
    denominator = 1.0 / target - (1.0 - share)
    if denominator <= 0.0:
        return None
    result = share / denominator
    return result if math.isfinite(result) and result > 0.0 else None


def projected_b0_query_ratio(
    *, current_b3_to_b0_ratio: float, opportunities: Sequence[R1ScopedOpportunity]
) -> float:
    """Project only disjoint query-local B3-side opportunities to B0 ratio."""

    if not math.isfinite(current_b3_to_b0_ratio) or current_b3_to_b0_ratio <= 0.0:
        raise ValueError("R1 current B3/B0 ratio differs")
    rows = tuple(opportunities)
    if not rows:
        raise ValueError("R1 query projection opportunities empty")
    if len({row.op_type for row in rows}) != len(rows):
        raise ValueError("R1 query projection bucket duplicates")
    for row in rows:
        row.validate()
        if row.scope != R1Scope.COMPLETE_QUERY:
            raise ValueError("R1 query projection scope differs")
    share_sum = sum(row.share for row in rows)
    if share_sum >= 1.0:
        raise ValueError("R1 query projection shares overlap")
    denominator = (1.0 - share_sum) + sum(
        row.share / row.region_speedup for row in rows
    )
    return current_b3_to_b0_ratio / denominator


@dataclass(frozen=True)
class R1RouteReceipt:
    """Mechanical R1-D result; still not an observed performance claim."""

    current_b3_to_b0_ratio: float
    opportunities: tuple[R1ScopedOpportunity, ...]
    target_contract: R1TargetContract = R1TargetContract()
    performance_claimed: bool = False
    schema_version: str = R1_SCHEMA_VERSION

    @property
    def projected_b0_query_ratio(self) -> float:
        return projected_b0_query_ratio(
            current_b3_to_b0_ratio=self.current_b3_to_b0_ratio,
            opportunities=self.opportunities,
        )

    def validate(self) -> None:
        self.target_contract.validate()
        if self.schema_version != R1_SCHEMA_VERSION or self.performance_claimed:
            raise ValueError("R1 route receipt identity differs")
        _ = self.projected_b0_query_ratio

    def to_dict(self) -> dict[str, object]:
        self.validate()
        projected = self.projected_b0_query_ratio
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "current_b3_to_b0_ratio": self.current_b3_to_b0_ratio,
            "opportunities": [row.to_dict() for row in self.opportunities],
            "target_contract": self.target_contract.to_dict(),
            "projected_b0_query_ratio": projected,
            "qualification_go": projected >= self.target_contract.query_qualification,
            "research_priority_go": projected >= self.target_contract.query_research,
            "performance_claimed": False,
        }
        payload["route_receipt_hash"] = canonical_hash(payload)
        return payload


def scoped_opportunity_from_dict(value: object) -> R1ScopedOpportunity:
    """Parse one disjoint opportunity and enforce scope/source ownership."""

    row = _as_mapping(value, "scoped opportunity")
    _exact_keys(
        row,
        {
            "scope",
            "op_type",
            "share",
            "region_speedup",
            "speedup_source",
            "admitted",
        },
        "scoped opportunity",
    )
    opportunity = R1ScopedOpportunity(
        scope=_enum_value(R1Scope, row["scope"], "opportunity scope"),
        op_type=_enum_value(R1OpType, row["op_type"], "opportunity op type"),
        share=_as_float(row["share"], "opportunity share"),
        region_speedup=_as_float(row["region_speedup"], "opportunity speedup"),
        speedup_source=_enum_value(
            R1SpeedupSource, row["speedup_source"], "opportunity speedup source"
        ),
        admitted=_as_bool(row["admitted"], "opportunity admission"),
    )
    opportunity.validate()
    return opportunity


def route_receipt_from_dict(value: object) -> R1RouteReceipt:
    """Recompute query projection and route verdict from primitive inputs."""

    row = _as_mapping(value, "route receipt")
    _exact_keys(
        row,
        {
            "schema_version",
            "current_b3_to_b0_ratio",
            "opportunities",
            "target_contract",
            "projected_b0_query_ratio",
            "qualification_go",
            "research_priority_go",
            "performance_claimed",
            "route_receipt_hash",
        },
        "route receipt",
    )
    receipt = R1RouteReceipt(
        schema_version=_as_str(row["schema_version"], "route schema"),
        current_b3_to_b0_ratio=_as_float(
            row["current_b3_to_b0_ratio"], "current B3/B0 ratio"
        ),
        opportunities=tuple(
            scoped_opportunity_from_dict(item)
            for item in _as_list(row["opportunities"], "route opportunities")
        ),
        target_contract=target_contract_from_dict(row["target_contract"]),
        performance_claimed=_as_bool(row["performance_claimed"], "route claim"),
    )
    rebuilt = receipt.to_dict()
    if canonical_json(row) != canonical_json(rebuilt):
        raise ValueError("R1 route derivation differs")
    return receipt


__all__ = [
    "CalibrationThresholds",
    "CalibrationTriplet",
    "ClockCalibrationReceipt",
    "CuptiTimestampSource",
    "R1AttributionMethod",
    "R1OpType",
    "R1OwnedEvent",
    "R1OwnerLedger",
    "R1RouteReceipt",
    "R1Scope",
    "R1ScopedOpportunity",
    "R1SpeedupSource",
    "R1TargetContract",
    "R1TimingLedger",
    "R1TopologyLedger",
    "R1TopologyNode",
    "canonical_hash",
    "calibration_thresholds_from_dict",
    "calibration_triplet_from_dict",
    "clock_calibration_from_dict",
    "collect_cupti_triplets",
    "derive_clock_calibration",
    "owned_event_from_dict",
    "owner_ledger_from_dict",
    "projected_b0_query_ratio",
    "required_region_speedup",
    "route_receipt_from_dict",
    "scoped_opportunity_from_dict",
    "target_contract_from_dict",
    "timing_ledger_from_dict",
    "topology_from_task",
    "topology_ledger_from_dict",
    "topology_node_from_dict",
]
