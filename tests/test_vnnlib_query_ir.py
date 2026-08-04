"""Tests for fail-closed VNNLIB box/property Query IR import."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import hashlib

import pytest
import torch

from boundflow.frontends.vnnlib import (
    materialize_vnnlib_box_query,
    parse_vnnlib_box_query,
)

SOURCE = """
; compact classification property
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 2.0))
(assert (<= 0.25 X_1))
(assert (>= 3.5 X_1))
(assert (or
  (and (>= Y_0 Y_2))
  (and (<= Y_1 Y_2))
))
"""


def test_vnnlib_query_ir_preserves_box_and_safety_margin_orientation() -> None:
    query = parse_vnnlib_box_query(SOURCE, query_id="classification:0")

    assert query.source_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert query.input_names == ("X_0", "X_1")
    assert query.output_names == ("Y_0", "Y_1", "Y_2")
    assert query.input_lower == (-1.0, 0.25)
    assert query.input_upper == (2.0, 3.5)
    assert query.unsafe_clauses[0].coefficients == (-1.0, 0.0, 1.0)
    assert query.unsafe_clauses[1].coefficients == (0.0, 1.0, -1.0)
    assert tuple(clause.threshold for clause in query.unsafe_clauses) == (0.0, 0.0)
    assert query.stable_hash() == query.stable_hash()


def test_vnnlib_query_ir_materializes_runtime_tensors() -> None:
    query = parse_vnnlib_box_query(SOURCE, query_id="classification:0")
    tensors = materialize_vnnlib_box_query(query, input_shape=(2,), dtype=torch.float64)

    assert tensors.input_lower.shape == (1, 2)
    assert tensors.input_upper.shape == (1, 2)
    assert tensors.linear_spec_c.shape == (1, 2, 3)
    assert tensors.thresholds.shape == (2,)
    assert tensors.input_lower.dtype == torch.float64
    assert torch.equal(
        tensors.linear_spec_c,
        torch.tensor([[[-1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]], dtype=torch.float64),
    )


def test_vnnlib_query_ir_supports_linear_constants_and_scalar_products() -> None:
    source = SOURCE.replace(
        "(and (>= Y_0 Y_2))",
        "(and (>= (+ Y_0 1.5) (* 2.0 Y_2)))",
    )
    query = parse_vnnlib_box_query(source, query_id="linear:0")

    assert query.unsafe_clauses[0].coefficients == (-1.0, 0.0, 2.0)
    assert query.unsafe_clauses[0].threshold == 1.5


@pytest.mark.parametrize(
    "source, message",
    [
        (SOURCE.replace("(assert (<= X_0 2.0))", ""), "incomplete"),
        (
            SOURCE.replace(
                "(assert (<= X_0 2.0))",
                "(assert (<= X_0 2.0))\n(assert (<= X_0 1.0))",
            ),
            "duplicates",
        ),
        (SOURCE.replace("X_1", "X_2"), "Query IR is invalid"),
        (
            SOURCE.replace(
                "(and (>= Y_0 Y_2))",
                "(and (>= Y_0 Y_2) (>= Y_1 Y_2))",
            ),
            "one inequality",
        ),
        (SOURCE.replace("(>= Y_0 Y_2)", "(= Y_0 Y_2)"), "unsupported"),
    ],
)
def test_vnnlib_query_ir_rejects_unsupported_or_ambiguous_sources(
    source: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_vnnlib_box_query(source, query_id="invalid")


def test_vnnlib_query_ir_rejects_source_digest_or_shape_tamper() -> None:
    with pytest.raises(ValueError, match="digest differs"):
        parse_vnnlib_box_query(
            SOURCE, query_id="classification:0", expected_source_sha256="0" * 64
        )

    query = parse_vnnlib_box_query(SOURCE, query_id="classification:0")
    with pytest.raises(ValueError, match="shape/dtype differs"):
        materialize_vnnlib_box_query(query, input_shape=(3,))
