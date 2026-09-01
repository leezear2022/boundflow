"""Contract tests for the formal CIBC IBP horizontal-fusion artifact."""

# pylint: disable=missing-function-docstring,duplicate-code

import copy

import pytest

from scripts import run_cibc_ibp_horizontal_artifact as artifact


def _groups(speedup: float) -> list[dict[str, float | int]]:
    return [
        {
            "group": ordinal,
            "baseline_ms": speedup,
            "candidate_ms": 1.0,
            "speedup": speedup,
        }
        for ordinal in range(30)
    ]


def _seal(value: dict[str, object]) -> dict[str, object]:
    value["worker_hash"] = artifact.canonical_hash(value)
    return value


def _operator_worker(ordinal: int, speedup: float) -> dict[str, object]:
    rows = [
        {
            "op_ordinal": op_ordinal,
            "groups": _groups(speedup),
            "baseline_median_ms": speedup,
            "candidate_median_ms": 1.0,
            "speedup": speedup,
            "maximum_absolute_difference": 0.0,
            "sign_exact": True,
        }
        for op_ordinal in artifact.CONV_ORDINALS
    ]
    return _seal(
        {
            "schema_version": artifact.WORKER_SCHEMA,
            "mode": "operator",
            "run_ordinal": ordinal,
            "order": artifact.OPERATOR_ORDERS[ordinal],
            "threads_per_block": artifact.SCHEDULES[ordinal],
            "environment": {
                "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
                "compute_capability": [8, 9],
            },
            "performance_claimed": False,
            "operators": rows,
            "operator_count": 6,
            "group_count": 30,
            "operator_repeats": 500,
            "plan_owned": True,
        }
    )


def _model_worker(ordinal: int, speedup: float) -> dict[str, object]:
    return _seal(
        {
            "schema_version": artifact.WORKER_SCHEMA,
            "mode": "model",
            "run_ordinal": ordinal,
            "order": artifact.MODEL_ORDERS[ordinal],
            "threads_per_block": 128,
            "environment": {
                "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
                "compute_capability": [8, 9],
            },
            "performance_claimed": False,
            "groups": _groups(speedup),
            "baseline_median_ms": speedup,
            "candidate_median_ms": 1.0,
            "speedup": speedup,
            "maximum_absolute_difference": 0.0,
            "final_maximum_absolute_difference": 0.0,
            "sign_exact": True,
            "conv_coverage": 6,
            "group_count": 30,
            "model_repeats": 100,
            "input_copy_included": True,
            "baseline_cuda_graph": True,
            "candidate_cuda_graph": True,
        }
    )


def test_cibc_ibp_formal_summary_selects_best_frozen_schedule() -> None:
    operators = [
        _operator_worker(0, 4.0),
        _operator_worker(1, 8.0),
        _operator_worker(2, 6.0),
    ]
    models = [_model_worker(ordinal, 2.0) for ordinal in range(6)]
    summary = artifact.derive_summary(operators, models)
    assert summary["selected_threads_per_block"] == 128
    assert summary["operator_speedup_geomean"] == pytest.approx(8.0)
    assert summary["model_speedup_geomean"] == pytest.approx(2.0)
    assert summary["performance_admitted"] is True


def test_cibc_ibp_formal_worker_rejects_resigned_timing_derivation() -> None:
    worker = _model_worker(0, 2.0)
    changed = copy.deepcopy(worker)
    changed["groups"][0]["speedup"] = 3.0  # type: ignore[index]
    changed.pop("worker_hash")
    _seal(changed)
    with pytest.raises(ValueError, match="timing group derivation"):
        artifact.validate_worker(
            changed, mode="model", ordinal=0, order="BC", threads=128
        )


def test_cibc_ibp_formal_worker_rejects_resigned_conv_coverage() -> None:
    worker = _model_worker(0, 2.0)
    worker["conv_coverage"] = 5
    worker.pop("worker_hash")
    _seal(worker)
    with pytest.raises(ValueError, match="model coverage"):
        artifact.validate_worker(
            worker, mode="model", ordinal=0, order="BC", threads=128
        )
