"""Candidate-independent topology and liveness summaries for placement planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence, Tuple

from ..ir.task import BoundTask

STATIC_BARRIER_SCHEMA_VERSION = "boundflow.materialization_static_barrier/v2"


@dataclass(frozen=True)
class StaticBarrierSummary:  # pylint: disable=too-many-instance-attributes
    """Static cost features for one ReLU pre-activation barrier."""

    barrier_id: str
    relu_output_id: str
    producer_op_type: str
    topo_index: int
    value_shape_per_domain: Tuple[int, ...]
    value_numel_per_domain: int
    spec_batch_size: int
    domain_batch_size: int
    element_size_bytes: int
    coefficient_elements: int
    coefficient_bytes: int
    estimated_dense_flops: int
    reuse_count: int
    direct_consumer_count: int
    direct_live_span: int
    downstream_depth: int
    downstream_merge_count: int
    downstream_branch_count: int
    downstream_path_count: int
    is_merge_output: bool
    is_branch_source: bool

    def validate(self) -> None:
        """Validate identities and non-negative static graph quantities."""

        for name in ("barrier_id", "relu_output_id", "producer_op_type"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "topo_index",
            "value_numel_per_domain",
            "spec_batch_size",
            "domain_batch_size",
            "element_size_bytes",
            "coefficient_elements",
            "coefficient_bytes",
            "estimated_dense_flops",
            "reuse_count",
            "direct_consumer_count",
            "direct_live_span",
            "downstream_depth",
            "downstream_merge_count",
            "downstream_branch_count",
            "downstream_path_count",
        ):
            value = int(getattr(self, name))
            minimum = (
                1
                if name
                in {
                    "value_numel_per_domain",
                    "spec_batch_size",
                    "domain_batch_size",
                    "element_size_bytes",
                    "coefficient_elements",
                    "coefficient_bytes",
                    "estimated_dense_flops",
                    "reuse_count",
                    "downstream_path_count",
                }
                else 0
            )
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {value}")
        if not self.value_shape_per_domain or any(
            int(dimension) <= 0 for dimension in self.value_shape_per_domain
        ):
            raise ValueError("value_shape_per_domain must contain positive dimensions")
        if _shape_numel(self.value_shape_per_domain) != int(
            self.value_numel_per_domain
        ):
            raise ValueError("value shape and numel do not match")
        expected_elements = (
            int(self.spec_batch_size)
            * int(self.domain_batch_size)
            * int(self.value_numel_per_domain)
        )
        if int(self.coefficient_elements) != expected_elements:
            raise ValueError("coefficient_elements does not match batch axes and shape")
        if int(self.coefficient_bytes) != expected_elements * int(
            self.element_size_bytes
        ):
            raise ValueError("coefficient_bytes does not match elements and dtype size")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible record."""

        self.validate()
        return {
            "schema_version": STATIC_BARRIER_SCHEMA_VERSION,
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StaticBarrierSummary":
        """Load and validate a static barrier record."""

        if payload.get("schema_version") != STATIC_BARRIER_SCHEMA_VERSION:
            raise ValueError("unsupported static barrier schema")
        fields = {
            key: value for key, value in payload.items() if key != "schema_version"
        }
        shape = fields.get("value_shape_per_domain")
        if not isinstance(shape, (list, tuple)):
            raise ValueError("value_shape_per_domain must be a sequence")
        fields["value_shape_per_domain"] = tuple(int(value) for value in shape)
        summary = cls(**fields)  # type: ignore[arg-type]
        summary.validate()
        return summary


def _shape_numel(shape: Sequence[int]) -> int:
    if not shape or any(int(dimension) <= 0 for dimension in shape):
        raise ValueError(f"barrier shape must contain positive dimensions: {shape}")
    return math.prod(int(dimension) for dimension in shape)


def summarize_static_barriers(  # pylint: disable=too-many-locals
    task: BoundTask,
    barrier_shapes: Mapping[str, Sequence[int]],
    *,
    spec_size: int,
    domain_batch_size: int,
    element_size_bytes: int,
) -> Tuple[StaticBarrierSummary, ...]:
    """Derive candidate-independent barrier features from a topological task graph."""

    if min(int(spec_size), int(domain_batch_size), int(element_size_bytes)) <= 0:
        raise ValueError("spec/domain/element-size values must be positive")
    producer: dict[str, tuple[int, str]] = {}
    consumers: dict[str, list[int]] = {}
    for index, op in enumerate(task.ops):
        for output in op.outputs:
            producer[output] = (index, op.op_type)
        for input_name in op.inputs:
            consumers.setdefault(input_name, []).append(index)

    successors: dict[int, set[int]] = {index: set() for index in range(len(task.ops))}
    for index, op in enumerate(task.ops):
        for output in op.outputs:
            successors[index].update(consumers.get(output, ()))

    depth_cache: dict[int, int] = {}
    path_cache: dict[int, int] = {}

    def depth(index: int) -> int:
        if index not in depth_cache:
            depth_cache[index] = (
                0
                if not successors[index]
                else 1 + max(depth(next_index) for next_index in successors[index])
            )
        return depth_cache[index]

    def path_count(index: int) -> int:
        if index not in path_cache:
            path_cache[index] = (
                1
                if not successors[index]
                else sum(path_count(next_index) for next_index in successors[index])
            )
        return path_cache[index]

    def reachable(start: int) -> set[int]:
        pending = list(successors[start])
        output: set[int] = set()
        while pending:
            index = pending.pop()
            if index in output:
                continue
            output.add(index)
            pending.extend(successors[index] - output)
        return output

    summaries: list[StaticBarrierSummary] = []
    for relu_index, op in enumerate(task.ops):
        if op.op_type != "relu":
            continue
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise ValueError("static barrier summary requires unary ReLU ops")
        barrier_id = op.inputs[0]
        relu_output = op.outputs[0]
        if barrier_id not in barrier_shapes:
            raise KeyError(f"missing static shape for barrier: {barrier_id}")
        shape = tuple(int(value) for value in barrier_shapes[barrier_id])
        if shape[0] != int(domain_batch_size):
            raise ValueError(
                f"barrier {barrier_id} domain dimension {shape[0]} "
                f"!= {domain_batch_size}"
            )
        value_numel = _shape_numel(shape[1:])
        coefficient_elements = int(domain_batch_size) * int(spec_size) * value_numel
        direct_consumers = consumers.get(relu_output, [])
        downstream = reachable(relu_index)
        producer_index, producer_op_type = producer.get(
            barrier_id, (relu_index, "input")
        )
        del producer_index
        summary = StaticBarrierSummary(
            barrier_id=barrier_id,
            relu_output_id=relu_output,
            producer_op_type=producer_op_type,
            topo_index=relu_index,
            value_shape_per_domain=tuple(shape[1:]),
            value_numel_per_domain=value_numel,
            spec_batch_size=int(spec_size),
            domain_batch_size=int(domain_batch_size),
            element_size_bytes=int(element_size_bytes),
            coefficient_elements=coefficient_elements,
            coefficient_bytes=coefficient_elements * int(element_size_bytes),
            estimated_dense_flops=coefficient_elements,
            reuse_count=max(1, path_count(relu_index)),
            direct_consumer_count=len(direct_consumers),
            direct_live_span=(
                max(direct_consumers) - relu_index if direct_consumers else 0
            ),
            downstream_depth=depth(relu_index),
            downstream_merge_count=sum(
                task.ops[index].op_type in {"add", "concat"} for index in downstream
            ),
            downstream_branch_count=sum(
                len(successors[index]) > 1 for index in downstream
            ),
            downstream_path_count=path_count(relu_index),
            is_merge_output=producer_op_type in {"add", "concat"},
            is_branch_source=len(direct_consumers) > 1,
        )
        summary.validate()
        summaries.append(summary)
    if set(barrier_shapes) != {summary.barrier_id for summary in summaries}:
        raise ValueError("barrier shape identities do not match task ReLU inputs")
    return tuple(summaries)


__all__ = [
    "STATIC_BARRIER_SCHEMA_VERSION",
    "StaticBarrierSummary",
    "summarize_static_barriers",
]
