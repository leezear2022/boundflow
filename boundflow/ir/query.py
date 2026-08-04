"""First-class verification Query IR for box inputs and linear unsafe clauses."""

# pylint: disable=too-many-boolean-expressions,too-many-instance-attributes
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Tuple

VNNLIB_BOX_QUERY_IR_SCHEMA_VERSION = "boundflow.vnnlib_box_query_ir/v1"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LinearUnsafeClauseIR:
    """One unsafe DNF disjunct represented by a scalar safety margin."""

    ordinal: int
    coefficients: Tuple[float, ...]
    threshold: float
    source_relation: str
    source_expression_hash: str

    def validate(self) -> None:
        if (
            self.ordinal < 0
            or not self.coefficients
            or not all(math.isfinite(value) for value in self.coefficients)
            or not any(value != 0.0 for value in self.coefficients)
            or not math.isfinite(self.threshold)
            or self.source_relation not in {">=", ">", "<=", "<"}
            or not _is_sha256(self.source_expression_hash)
        ):
            raise ValueError("linear unsafe clause IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "ordinal": self.ordinal,
            "coefficients": list(self.coefficients),
            "threshold": self.threshold,
            "source_relation": self.source_relation,
            "source_expression_hash": self.source_expression_hash,
            "margin_semantics": "unsafe_when_margin_le_threshold",
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class VNNLibBoxQueryIR:
    """A fully bounded input box and OR-of-single-linear-unsafe-clauses query."""

    query_id: str
    source_sha256: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...]
    input_lower: Tuple[float, ...]
    input_upper: Tuple[float, ...]
    unsafe_clauses: Tuple[LinearUnsafeClauseIR, ...]
    schema_version: str = VNNLIB_BOX_QUERY_IR_SCHEMA_VERSION

    def validate(self) -> None:
        expected_inputs = tuple(f"X_{index}" for index in range(len(self.input_names)))
        expected_outputs = tuple(
            f"Y_{index}" for index in range(len(self.output_names))
        )
        clause_widths = {len(clause.coefficients) for clause in self.unsafe_clauses}
        if (
            self.schema_version != VNNLIB_BOX_QUERY_IR_SCHEMA_VERSION
            or not self.query_id
            or not _is_sha256(self.source_sha256)
            or self.input_names != expected_inputs
            or self.output_names != expected_outputs
            or not self.input_names
            or not self.output_names
            or len(self.input_lower) != len(self.input_names)
            or len(self.input_upper) != len(self.input_names)
            or not self.unsafe_clauses
            or clause_widths != {len(self.output_names)}
            or not all(
                math.isfinite(value) for value in (*self.input_lower, *self.input_upper)
            )
            or any(
                lower > upper
                for lower, upper in zip(self.input_lower, self.input_upper)
            )
        ):
            raise ValueError("VNNLIB box Query IR is invalid")
        for ordinal, clause in enumerate(self.unsafe_clauses):
            clause.validate()
            if clause.ordinal != ordinal:
                raise ValueError("VNNLIB unsafe clause ordinal differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "source_sha256": self.source_sha256,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "input_lower": list(self.input_lower),
            "input_upper": list(self.input_upper),
            "unsafe_clauses": [clause.to_dict() for clause in self.unsafe_clauses],
            "property_aggregation": "all_unsafe_disjuncts_must_be_refuted",
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


__all__ = [
    "LinearUnsafeClauseIR",
    "VNNLIB_BOX_QUERY_IR_SCHEMA_VERSION",
    "VNNLibBoxQueryIR",
]
