"""Synthetic semantic tests for the MR3 production bridge formal gates."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy

import pytest

from boundflow.runtime.mr3_production_bridge_formal import (
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    EXPECTED_RUNS,
    FORMAL_SCHEMA,
    MODEL_SHA256,
    PROPERTY_SHA256,
    SOURCE_COMMIT,
    STATUS,
    VNNCOMP_COMMIT,
    WORKER_SCHEMA,
    canonical_hash,
    derive_summary,
)
from scripts.probe_mr3_production_bridge_tamper import run as run_tamper


def _tensor(shape: tuple[int, ...], value: float):
    count = 1
    for dimension in shape:
        count *= dimension
    return {
        "shape": list(shape),
        "stride": (
            [count // max(1, shape[0]), 1] if len(shape) == 2 else [1] * len(shape)
        ),
        "dtype": "torch.float32",
        "device": "cuda:0",
        "requires_grad": True,
        "numel": count,
        "content_sha256": "a" * 64,
        "values": [value] * count,
    }


def _bridge_receipt():
    return {
        "evaluation_count": 10,
        "forward_launch_count": 10,
        "backward_launch_count": 9,
        "empty_beta_tensor_count": 10,
        "empty_beta_numel": 0,
        "relu_conv_content_match_count": 10,
        "relu_conv_pointer_match_count": 0,
        "persistent_dense_a_count": 0,
        "fallback_count": 0,
        "eager_count": 0,
        "native_shadow_count": 0,
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _worker(mode: str, offset: float):
    evaluation = [
        {
            "evaluation_ordinal": ordinal,
            "lower": _tensor((1, 1), -0.5 + ordinal * 0.01 + offset),
            "aggregate_loss": 0.5 - ordinal * 0.01 - offset,
        }
        for ordinal in range(10)
    ]
    mutation = []
    for ordinal in range(9):
        row = {
            "mutation_ordinal": ordinal,
            "gradient": _tensor((1, 1), 0.01 + ordinal * 0.001 + offset),
            "alpha_pre_clamp": _tensor((1, 1), 0.5 + offset),
            "exp_avg": _tensor((1, 1), 0.001 + offset),
            "exp_avg_sq": _tensor((1, 1), 0.0001 + offset),
            "alpha_post_clamp": _tensor((1, 1), 0.5 + offset),
            "lr_used": 0.01 * (0.98**ordinal),
            "optimizer_step": float(ordinal + 1),
            "clamp_mask": {
                "zero_count": 0,
                "one_count": 0,
                "interior_count": 1,
            },
        }
        mutation.append(row)
    value = {
        "schema_version": WORKER_SCHEMA,
        "mode": mode,
        "source": {
            "abcrown_commit": ABCROWN_COMMIT,
            "auto_lirpa_commit": AUTO_LIRPA_COMMIT,
            "vnncomp_commit": VNNCOMP_COMMIT,
            "model_sha256": MODEL_SHA256,
            "property_sha256": PROPERTY_SHA256,
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
        },
        "solver_result": {
            "status": "verified",
            "success": True,
            "visited_domains": [6],
        },
        "outer_result_state": [_tensor((1, 1), -0.25 + offset)],
        "inner_result_states": [[row["lower"]] for row in evaluation],
        "final_target_alpha_state": _tensor((1, 1), 0.5 + offset),
        "final_module_state": [_tensor((1, 1), 0.5 + offset)],
        "region_states": [
            {
                "evaluation_ordinal": ordinal,
                "lower_a": _tensor((1, 1), 0.1 + offset),
                "lower_bias": _tensor((1, 1), 0.2 + offset),
            }
            for ordinal in range(10)
        ],
        "evaluation_trajectory": evaluation,
        "mutation_trajectory": mutation,
        "final_clip_state": {
            "alpha": _tensor((1, 1), 0.5 + offset),
            "clamp_mask": {
                "zero_count": 0,
                "one_count": 0,
                "interior_count": 1,
            },
        },
        "bridge_receipt": _bridge_receipt() if mode == "bridge" else None,
        "atomic_receipt": {
            "exact_call_launch_count": 1,
            "staged_emit_count": 1,
            "atomic_commit_count": 1,
            "rollback_count": 0,
        },
        "timing_recorded": False,
        "performance_claimed": False,
    }
    value["worker_hash"] = canonical_hash(value)
    return value


def _rollback():
    value = {
        "schema_version": WORKER_SCHEMA,
        "mode": "bridge",
        "injected_failure_evaluation": 5,
        "caught_failure": "MR3 injected candidate failure",
        "atomic_receipt": {
            "exact_call_launch_count": 1,
            "staged_emit_count": 0,
            "atomic_commit_count": 0,
            "rollback_count": 1,
            "owner_tensor_count": 12,
            "owner_content_hash_before": "a" * 64,
            "owner_content_hash_after": "a" * 64,
            "owner_pointer_hash_before": "b" * 64,
            "owner_pointer_hash_after": "b" * 64,
            "version_delta_min": 1,
            "version_delta_max": 6,
        },
        "timing_recorded": False,
        "performance_claimed": False,
    }
    value["worker_hash"] = canonical_hash(value)
    return value


def _raw():
    runs = []
    for pair, position, mode in EXPECTED_RUNS:
        runs.append(
            {
                "pair_index": pair,
                "position": position,
                "mode": mode,
                "worker": _worker(mode, 1.0e-7 if mode == "bridge" else 0.0),
            }
        )
    value = {
        "schema_version": FORMAL_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "run_order": [list(run) for run in EXPECTED_RUNS],
        "runs": runs,
        "rollback_probe": _rollback(),
        "timing_recorded": False,
        "performance_claimed": False,
    }
    value["raw_hash"] = canonical_hash(value)
    return value


def _resign(raw, worker_index: int):
    worker = raw["runs"][worker_index]["worker"]
    unsigned_worker = dict(worker)
    unsigned_worker.pop("worker_hash")
    worker["worker_hash"] = canonical_hash(unsigned_worker)
    unsigned_raw = dict(raw)
    unsigned_raw.pop("raw_hash")
    raw["raw_hash"] = canonical_hash(unsigned_raw)


def test_formal_summary_closes_only_single_site_correctness() -> None:
    summary = derive_summary(_raw())
    assert summary["status"] == STATUS
    assert summary["pair_count"] == 5
    assert summary["candidate_forward_count"] == 50
    assert summary["timing_open"] is False


def test_fully_resigned_optimizer_drift_fails_closed() -> None:
    raw = _raw()
    raw["runs"][1]["worker"]["mutation_trajectory"][0]["exp_avg"]["values"][0] += 0.01
    _resign(raw, 1)
    with pytest.raises(ValueError, match="numeric drift"):
        derive_summary(raw)


def test_all_fully_resigned_tamper_cases_are_rejected() -> None:
    report = run_tamper(deepcopy(_raw()))
    assert report["case_count"] >= 14
    assert report["rejected_count"] == report["case_count"]
