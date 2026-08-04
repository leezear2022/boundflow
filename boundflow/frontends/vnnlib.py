"""Fail-closed VNNLIB frontend for bounded boxes and linear unsafe clauses."""

# pylint: disable=too-many-branches,too-many-locals,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Sequence, TypeAlias

import torch

from ..ir.query import LinearUnsafeClauseIR, VNNLibBoxQueryIR

SExpr: TypeAlias = str | list["SExpr"]
_TOKEN_PATTERN = re.compile(r"\(|\)|[^\s()]+")
_VARIABLE_PATTERN = re.compile(r"([XY])_(0|[1-9][0-9]*)\Z")


def _source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _expression_hash(expression: SExpr) -> str:
    def encode(value: SExpr) -> str:
        if isinstance(value, str):
            return value
        return "(" + " ".join(encode(item) for item in value) + ")"

    return hashlib.sha256(encode(expression).encode("utf-8")).hexdigest()


def _strip_comments(text: str) -> str:
    return "\n".join(line.split(";", maxsplit=1)[0] for line in text.splitlines())


def _parse_expressions(text: str) -> list[SExpr]:
    tokens = _TOKEN_PATTERN.findall(_strip_comments(text))
    cursor = 0

    def parse_one() -> SExpr:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("VNNLIB expression ended unexpectedly")
        token = tokens[cursor]
        cursor += 1
        if token == ")":
            raise ValueError("VNNLIB contains an unmatched closing parenthesis")
        if token != "(":
            return token
        items: list[SExpr] = []
        while True:
            if cursor >= len(tokens):
                raise ValueError("VNNLIB contains an unmatched opening parenthesis")
            if tokens[cursor] == ")":
                cursor += 1
                return items
            items.append(parse_one())

    expressions: list[SExpr] = []
    while cursor < len(tokens):
        expressions.append(parse_one())
    if not expressions:
        raise ValueError("VNNLIB source is empty")
    return expressions


def _list(expression: SExpr, label: str) -> list[SExpr]:
    if not isinstance(expression, list) or not expression:
        raise ValueError(f"{label} must be a non-empty expression")
    return expression


def _atom(expression: SExpr, label: str) -> str:
    if not isinstance(expression, str):
        raise ValueError(f"{label} must be an atom")
    return expression


def _number(expression: SExpr, label: str) -> float:
    atom = _atom(expression, label)
    try:
        value = float(atom)
    except ValueError as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _variables(expression: SExpr) -> set[str]:
    if isinstance(expression, str):
        return {expression} if _VARIABLE_PATTERN.fullmatch(expression) else set()
    found: set[str] = set()
    for item in expression:
        found.update(_variables(item))
    return found


def _variable_index(name: str, prefix: str) -> int:
    match = _VARIABLE_PATTERN.fullmatch(name)
    if match is None or match.group(1) != prefix:
        raise ValueError(f"expected a {prefix} variable, got {name!r}")
    return int(match.group(2))


def _parse_input_bound(expression: SExpr) -> tuple[str, str, float]:
    relation = _list(expression, "VNNLIB input assertion")
    if len(relation) != 3 or _atom(relation[0], "input relation") not in {"<=", ">="}:
        raise ValueError("VNNLIB input assertion must be one scalar <=/>= bound")
    operator = _atom(relation[0], "input relation")
    left = _atom(relation[1], "input bound left operand")
    right = _atom(relation[2], "input bound right operand")
    left_is_input = _VARIABLE_PATTERN.fullmatch(left) is not None and left.startswith(
        "X_"
    )
    right_is_input = _VARIABLE_PATTERN.fullmatch(
        right
    ) is not None and right.startswith("X_")
    if left_is_input == right_is_input:
        raise ValueError(
            "VNNLIB input bound must compare one X variable and one scalar"
        )
    if left_is_input:
        name = left
        value = _number(right, "input bound scalar")
        side = "upper" if operator == "<=" else "lower"
    else:
        name = right
        value = _number(left, "input bound scalar")
        side = "lower" if operator == "<=" else "upper"
    return name, side, value


@dataclass(frozen=True)
class _LinearExpression:
    coefficients: tuple[float, ...]
    constant: float

    def scaled(self, factor: float) -> "_LinearExpression":
        return _LinearExpression(
            tuple(factor * value for value in self.coefficients),
            factor * self.constant,
        )

    def plus(self, other: "_LinearExpression") -> "_LinearExpression":
        return _LinearExpression(
            tuple(
                left + right
                for left, right in zip(self.coefficients, other.coefficients)
            ),
            self.constant + other.constant,
        )


def _linear_expression(expression: SExpr, output_count: int) -> _LinearExpression:
    zeros = (0.0,) * output_count
    if isinstance(expression, str):
        match = _VARIABLE_PATTERN.fullmatch(expression)
        if match is not None:
            if match.group(1) != "Y":
                raise ValueError(
                    "VNNLIB output constraint cannot reference X variables"
                )
            index = int(match.group(2))
            if index >= output_count:
                raise ValueError("VNNLIB output variable index is out of range")
            coefficients = [0.0] * output_count
            coefficients[index] = 1.0
            return _LinearExpression(tuple(coefficients), 0.0)
        return _LinearExpression(zeros, _number(expression, "linear constant"))
    items = _list(expression, "linear output expression")
    operator = _atom(items[0], "linear output operator")
    arguments = items[1:]
    if operator == "+" and arguments:
        result = _LinearExpression(zeros, 0.0)
        for argument in arguments:
            result = result.plus(_linear_expression(argument, output_count))
        return result
    if operator == "-" and arguments:
        result = _linear_expression(arguments[0], output_count)
        if len(arguments) == 1:
            return result.scaled(-1.0)
        for argument in arguments[1:]:
            result = result.plus(
                _linear_expression(argument, output_count).scaled(-1.0)
            )
        return result
    if operator == "*" and len(arguments) == 2:
        left = _linear_expression(arguments[0], output_count)
        right = _linear_expression(arguments[1], output_count)
        left_scalar = not any(left.coefficients)
        right_scalar = not any(right.coefficients)
        if left_scalar == right_scalar:
            raise ValueError("VNNLIB multiplication must contain one scalar factor")
        return (
            right.scaled(left.constant) if left_scalar else left.scaled(right.constant)
        )
    raise ValueError(f"unsupported VNNLIB linear operator: {operator!r}")


def _parse_output_clause(
    expression: SExpr, *, ordinal: int, output_count: int
) -> LinearUnsafeClauseIR:
    branch = _list(expression, "VNNLIB unsafe disjunct")
    if _atom(branch[0], "unsafe disjunct operator") == "and":
        if len(branch) != 2:
            raise ValueError("VNNLIB v1 requires one inequality per unsafe disjunct")
        branch = _list(branch[1], "VNNLIB unsafe relation")
    if len(branch) != 3:
        raise ValueError("VNNLIB unsafe clause must be one binary relation")
    relation = _atom(branch[0], "VNNLIB unsafe relation")
    if relation not in {">=", ">", "<=", "<"}:
        raise ValueError("VNNLIB unsafe relation is unsupported")
    left = _linear_expression(branch[1], output_count)
    right = _linear_expression(branch[2], output_count)
    if relation in {">=", ">"}:
        margin = right.plus(left.scaled(-1.0))
    else:
        margin = left.plus(right.scaled(-1.0))
    clause = LinearUnsafeClauseIR(
        ordinal=ordinal,
        coefficients=margin.coefficients,
        threshold=-margin.constant,
        source_relation=relation,
        source_expression_hash=_expression_hash(branch),
    )
    clause.validate()
    return clause


def parse_vnnlib_box_query(
    text: str, *, query_id: str, expected_source_sha256: str | None = None
) -> VNNLibBoxQueryIR:
    """Parse the supported VNNLIB subset into immutable Query IR."""

    source_sha256 = _source_sha256(text)
    if expected_source_sha256 is not None and expected_source_sha256 != source_sha256:
        raise ValueError("VNNLIB source digest differs")
    declarations: dict[str, str] = {}
    input_assertions: list[SExpr] = []
    output_assertions: list[SExpr] = []
    for expression in _parse_expressions(text):
        form = _list(expression, "VNNLIB top-level form")
        operator = _atom(form[0], "VNNLIB top-level operator")
        if operator == "declare-const":
            if len(form) != 3 or _atom(form[2], "declaration type") != "Real":
                raise ValueError("VNNLIB declaration must be a Real constant")
            name = _atom(form[1], "declaration name")
            if _VARIABLE_PATTERN.fullmatch(name) is None or name in declarations:
                raise ValueError("VNNLIB variable declaration is invalid or duplicated")
            declarations[name] = "Real"
            continue
        if operator != "assert" or len(form) != 2:
            raise ValueError(f"unsupported VNNLIB top-level form: {operator!r}")
        names = _variables(form[1])
        if not names or any(name not in declarations for name in names):
            raise ValueError("VNNLIB assertion references missing/no variables")
        if all(name.startswith("X_") for name in names):
            input_assertions.append(form[1])
        elif all(name.startswith("Y_") for name in names):
            output_assertions.append(form[1])
        else:
            raise ValueError("VNNLIB assertion cannot mix input and output variables")

    input_names = tuple(
        sorted(
            (name for name in declarations if name.startswith("X_")),
            key=lambda name: _variable_index(name, "X"),
        )
    )
    output_names = tuple(
        sorted(
            (name for name in declarations if name.startswith("Y_")),
            key=lambda name: _variable_index(name, "Y"),
        )
    )
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for assertion in input_assertions:
        name, side, value = _parse_input_bound(assertion)
        target = lower if side == "lower" else upper
        if name in target:
            raise ValueError(f"VNNLIB duplicates the {side} bound for {name}")
        target[name] = value
    if set(lower) != set(input_names) or set(upper) != set(input_names):
        raise ValueError("VNNLIB input box is incomplete")
    if len(output_assertions) != 1:
        raise ValueError("VNNLIB v1 requires exactly one output assertion")
    output_formula = _list(output_assertions[0], "VNNLIB output assertion")
    if (
        _atom(output_formula[0], "VNNLIB output formula") != "or"
        or len(output_formula) < 2
    ):
        raise ValueError("VNNLIB v1 output assertion must be a non-empty OR")
    clauses = tuple(
        _parse_output_clause(branch, ordinal=ordinal, output_count=len(output_names))
        for ordinal, branch in enumerate(output_formula[1:])
    )
    query = VNNLibBoxQueryIR(
        query_id=query_id,
        source_sha256=source_sha256,
        input_names=input_names,
        output_names=output_names,
        input_lower=tuple(lower[name] for name in input_names),
        input_upper=tuple(upper[name] for name in input_names),
        unsafe_clauses=clauses,
    )
    query.validate()
    return query


def import_vnnlib_box_query(path: str | Path, *, query_id: str) -> VNNLibBoxQueryIR:
    source = Path(path).read_text(encoding="utf-8")
    return parse_vnnlib_box_query(source, query_id=query_id)


@dataclass(frozen=True)
class VNNLibQueryTensors:
    """Materialized tensor view consumed by BoundFlow verification runtimes."""

    input_lower: torch.Tensor
    input_upper: torch.Tensor
    linear_spec_c: torch.Tensor
    thresholds: torch.Tensor


def materialize_vnnlib_box_query(
    query: VNNLibBoxQueryIR,
    *,
    input_shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> VNNLibQueryTensors:
    query.validate()
    shape = tuple(int(dimension) for dimension in input_shape)
    if (
        not shape
        or any(dimension < 1 for dimension in shape)
        or math.prod(shape) != len(query.input_names)
        or not dtype.is_floating_point
    ):
        raise ValueError("VNNLIB query tensor materialization shape/dtype differs")
    lower = torch.tensor(query.input_lower, dtype=dtype, device=device).reshape(
        (1, *shape)
    )
    upper = torch.tensor(query.input_upper, dtype=dtype, device=device).reshape(
        (1, *shape)
    )
    linear_spec_c = torch.tensor(
        [clause.coefficients for clause in query.unsafe_clauses],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)
    thresholds = torch.tensor(
        [clause.threshold for clause in query.unsafe_clauses],
        dtype=dtype,
        device=device,
    )
    return VNNLibQueryTensors(
        input_lower=lower,
        input_upper=upper,
        linear_spec_c=linear_spec_c,
        thresholds=thresholds,
    )


__all__ = [
    "VNNLibQueryTensors",
    "import_vnnlib_box_query",
    "materialize_vnnlib_box_query",
    "parse_vnnlib_box_query",
]
