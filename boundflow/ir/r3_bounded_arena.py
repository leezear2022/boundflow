"""Typed R3-1b0 reverse recurrence and two-scratch liveness contracts."""

# pylint: disable=missing-function-docstring,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json

R31B_TRACE_SCHEMA = "boundflow.r3-1b0-bounded-arena-trace/v1"
R31B_MAX_SCRATCH_SLOTS = 2


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


class R31BStepKind(str, Enum):
    """Closed operation vocabulary before any TIR lowering is admitted."""

    SPEC_SEED = "spec_seed"
    LINEAR_RIGHT = "linear_right"
    CONV2D_RIGHT = "conv2d_right"
    RELU_LOWER = "relu_lower"
    RESHAPE_VIEW = "reshape_view"
    RESIDUAL_REGION = "residual_region"
    INPUT_CONCRETIZE = "input_concretize"


@dataclass(frozen=True)
class R31BBranchV1:
    """One fused reverse path from a primal Add input to its common join value."""

    source_value: str
    join_value: str
    primal_ops: tuple[str, ...]
    identity: bool

    def validate(self) -> None:
        if (
            not self.source_value
            or not self.join_value
            or self.identity != (len(self.primal_ops) == 0)
            or len(set(self.primal_ops)) != len(self.primal_ops)
        ):
            raise ValueError("R3-1b residual branch differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "source_value": self.source_value,
            "join_value": self.join_value,
            "primal_ops": list(self.primal_ops),
            "identity": self.identity,
        }


@dataclass(frozen=True)
class R31BTraceStepV1:
    """One scheduled reverse recurrence step with explicit scratch ownership."""

    ordinal: int
    kind: R31BStepKind
    input_value: str
    output_value: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    primal_ops: tuple[str, ...]
    input_slot: int
    output_slot: int
    in_place: bool
    accumulate_into_output: bool = False
    branches: tuple[R31BBranchV1, ...] = ()

    def validate(self) -> None:
        if (
            self.ordinal < 0
            or not self.input_value
            or not self.output_value
            or not self.input_shape
            or not self.output_shape
            or any(dimension <= 0 for dimension in self.input_shape + self.output_shape)
            or len(set(self.primal_ops)) != len(self.primal_ops)
            or self.input_slot not in {-1, 0, 1}
            or self.output_slot not in {-1, 0, 1}
        ):
            raise ValueError("R3-1b trace step differs")
        if self.kind == R31BStepKind.SPEC_SEED:
            if (
                self.input_slot != -1
                or self.output_slot not in {0, 1}
                or self.primal_ops
            ):
                raise ValueError("R3-1b seed step differs")
        elif self.kind == R31BStepKind.INPUT_CONCRETIZE:
            if (
                self.input_slot not in {0, 1}
                or self.output_slot != -1
                or self.primal_ops
            ):
                raise ValueError("R3-1b concretize step differs")
        elif (
            not self.primal_ops
            or self.input_slot not in {0, 1}
            or self.output_slot not in {0, 1}
        ):
            raise ValueError("R3-1b executable step differs")
        if self.in_place != (
            self.input_slot == self.output_slot
            and self.kind not in {R31BStepKind.SPEC_SEED, R31BStepKind.INPUT_CONCRETIZE}
        ):
            raise ValueError("R3-1b in-place declaration differs")
        if self.kind == R31BStepKind.RESIDUAL_REGION:
            if (
                len(self.branches) != 2
                or self.input_slot == self.output_slot
                or self.in_place
                or self.accumulate_into_output is not True
                or len({branch.source_value for branch in self.branches}) != 2
                or len({branch.join_value for branch in self.branches}) != 1
                or any(
                    branch.join_value != self.output_value for branch in self.branches
                )
            ):
                raise ValueError("R3-1b residual step differs")
            for branch in self.branches:
                branch.validate()
        elif self.branches or self.accumulate_into_output:
            raise ValueError("R3-1b non-residual branch fields differ")

    @property
    def input_numel(self) -> int:
        return _numel(self.input_shape)

    @property
    def output_numel(self) -> int:
        return _numel(self.output_shape)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "ordinal": self.ordinal,
            "kind": self.kind.value,
            "input_value": self.input_value,
            "output_value": self.output_value,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "primal_ops": list(self.primal_ops),
            "input_slot": self.input_slot,
            "output_slot": self.output_slot,
            "in_place": self.in_place,
            "accumulate_into_output": self.accumulate_into_output,
            "branches": [branch.to_dict() for branch in self.branches],
        }


@dataclass(frozen=True)
class R31BBoundedArenaTraceV1:
    """Frozen full-lower schedule before compiled module construction."""

    source_hash: str
    topology_hash: str
    production_plan_hash: str
    steps: tuple[R31BTraceStepV1, ...]
    scratch_slot_count: int
    scratch_capacity_elements: int
    start_node_id: str = "25/Conv_8"
    domain_count: int = 6
    spec_count: int = 1
    dtype: str = "torch.float32"
    compiled_region: bool = False
    timing_recorded: bool = False
    performance_claimed: bool = False
    schema_version: str = R31B_TRACE_SCHEMA

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_hash": self.source_hash,
            "topology_hash": self.topology_hash,
            "production_plan_hash": self.production_plan_hash,
            "steps": [step.to_dict() for step in self.steps],
            "scratch_slot_count": self.scratch_slot_count,
            "scratch_capacity_elements": self.scratch_capacity_elements,
            "start_node_id": self.start_node_id,
            "domain_count": self.domain_count,
            "spec_count": self.spec_count,
            "dtype": self.dtype,
            "compiled_region": self.compiled_region,
            "timing_recorded": self.timing_recorded,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _hash(self.identity_payload())

    def validate(self) -> None:
        if (
            self.schema_version != R31B_TRACE_SCHEMA
            or not _is_hash(self.source_hash)
            or not _is_hash(self.topology_hash)
            or not _is_hash(self.production_plan_hash)
            or self.topology_hash != _hash([step.to_dict() for step in self.steps])
            or len(self.steps) != 12
            or tuple(step.ordinal for step in self.steps) != tuple(range(12))
            or self.scratch_slot_count != 2
            or self.scratch_slot_count > R31B_MAX_SCRATCH_SLOTS
            or self.scratch_capacity_elements <= 0
            or self.start_node_id != "25/Conv_8"
            or self.domain_count != 6
            or self.spec_count != 1
            or self.dtype != "torch.float32"
            or self.compiled_region
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("R3-1b bounded-arena trace differs")
        for step in self.steps:
            step.validate()
        if (
            self.steps[0].kind != R31BStepKind.SPEC_SEED
            or self.steps[-1].kind != R31BStepKind.INPUT_CONCRETIZE
            or sum(step.kind == R31BStepKind.RESIDUAL_REGION for step in self.steps)
            != 2
            or max(max(step.input_numel, step.output_numel) for step in self.steps)
            != self.scratch_capacity_elements
        ):
            raise ValueError("R3-1b trace closure differs")
        live_slot = self.steps[0].output_slot
        for step in self.steps[1:]:
            if step.input_slot != live_slot:
                raise ValueError("R3-1b scratch continuity differs")
            if step.kind != R31BStepKind.INPUT_CONCRETIZE:
                live_slot = step.output_slot


__all__ = [
    "R31BBranchV1",
    "R31BBoundedArenaTraceV1",
    "R31BStepKind",
    "R31BTraceStepV1",
]
