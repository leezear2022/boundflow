#!/usr/bin/env python3
"""Verify RVIR-v4 formal step-trace parity with the frozen V4-0 capture."""

# pylint: disable=protected-access,wrong-import-position,too-many-locals
# pylint: disable=too-many-statements,too-many-branches
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_production_state import (
    ProductionStateSnapshotV4,
    production_snapshot_from_payload_v4,
)
from scripts import run_rvir_v4_optimizer_step_artifact as optimizer_artifact
from scripts import run_rvir_v4_production_state_capture as production_artifact

REPORT_SCHEMA_VERSION = "boundflow.rvir-v4-optimizer-step-source-parity/v1"
ATOL = 2e-4
RTOL = 2e-4


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_capture(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("RVIR-v4 source-parity capture root differs")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _metadata_schema(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"content_sha256", "value"}
    }


def _call_structure(capture: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls = capture.get("calls")
    if not isinstance(calls, list):
        raise TypeError("RVIR-v4 source-parity call inventory differs")
    structure: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise TypeError("RVIR-v4 source-parity call row differs")
        row = {
            key: call.get(key)
            for key in (
                "call_id",
                "parent_call_id",
                "core_id",
                "depth",
                "phase",
                "external_phase",
                "method",
                "bound_lower",
                "bound_upper",
                "pre_alpha_count",
                "pre_beta_value_count",
                "post_alpha_count",
                "post_beta_value_count",
            )
        }
        for name in ("pre_state", "post_state", "result_tensors"):
            values = call.get(name)
            if not isinstance(values, list):
                raise TypeError(f"RVIR-v4 source-parity {name} inventory differs")
            row[name] = [_metadata_schema(value) for value in values]
        structure.append(row)
    return structure


def _numeric_diff(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        raise ValueError("RVIR-v4 source-parity tensor type/shape differs")
    if not left.is_floating_point():
        if not torch.equal(left, right):
            raise ValueError("RVIR-v4 source-parity nonfloat tensor differs")
        return 0.0, 0.0
    if not torch.equal(torch.isfinite(left), torch.isfinite(right)):
        raise ValueError("RVIR-v4 source-parity finite mask differs")
    if not torch.equal(torch.sign(left), torch.sign(right)):
        raise ValueError("RVIR-v4 source-parity tensor sign differs")
    finite = torch.isfinite(left)
    if not bool(finite.any().item()):
        return 0.0, 0.0
    difference = (left[finite] - right[finite]).abs()
    allowed = ATOL + RTOL * left[finite].abs()
    if not bool((difference <= allowed).all().item()):
        raise ValueError("RVIR-v4 source-parity numeric tolerance differs")
    denominator = torch.maximum(
        left[finite].abs(), torch.full_like(left[finite], 1e-12)
    )
    return float(difference.max().item()), float(
        (difference / denominator).max().item()
    )


def compare_snapshots(
    baseline: ProductionStateSnapshotV4,
    candidate: ProductionStateSnapshotV4,
) -> dict[str, object]:
    """Compare exact snapshot structure and tolerance-bound finite values."""

    baseline.validate()
    candidate.validate()
    if (
        baseline.schema_version != candidate.schema_version
        or baseline.snapshot_id != candidate.snapshot_id
        or baseline.optimizer_policy.to_dict() != candidate.optimizer_policy.to_dict()
        or [entry.to_dict() for entry in baseline.history]
        != [entry.to_dict() for entry in candidate.history]
    ):
        raise ValueError("RVIR-v4 source-parity snapshot identity differs")
    left = baseline.tensor_map()
    right = candidate.tensor_map()
    if set(left) != set(right):
        raise ValueError("RVIR-v4 source-parity tensor paths differ")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    changed_paths = 0
    for path in sorted(left):
        left_tensor = left[path]
        right_tensor = right[path]
        left_schema = _metadata_schema(left_tensor.metadata())
        right_schema = _metadata_schema(right_tensor.metadata())
        if left_schema != right_schema:
            raise ValueError(f"RVIR-v4 source-parity tensor schema differs: {path}")
        absolute, relative = _numeric_diff(left_tensor.value, right_tensor.value)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        changed_paths += left_tensor.content_sha256 != right_tensor.content_sha256
    return {
        "tensor_path_count": len(left),
        "history_entry_count": len(baseline.history),
        "changed_content_digest_path_count": changed_paths,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "schema_history_policy_exact": True,
        "finite_mask_exact": True,
        "sign_exact": True,
    }


def _mutation_structure(core: Mapping[str, Any]) -> list[dict[str, object]]:
    mutations = core.get("mutations")
    if not isinstance(mutations, list):
        raise TypeError("RVIR-v4 source-parity mutation inventory differs")
    return [
        {
            "semantic_path": row.get("semantic_path"),
            "changed": row.get("changed"),
        }
        for row in mutations
    ]


def _one_core(capture: Mapping[str, Any]) -> Mapping[str, Any]:
    cores = capture.get("cores")
    if (
        not isinstance(cores, list)
        or len(cores) != 1
        or not isinstance(cores[0], Mapping)
    ):
        raise ValueError("RVIR-v4 source-parity core inventory differs")
    return cast(Mapping[str, Any], cores[0])


def compare_captures(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, object]:
    """Compare fixed source identity, call topology, snapshots, and core outputs."""

    for name in ("source", "protocol", "solver_result"):
        if baseline.get(name) != candidate.get(name):
            raise ValueError(f"RVIR-v4 source-parity {name} differs")
    baseline_calls = _call_structure(baseline)
    candidate_calls = _call_structure(candidate)
    if baseline_calls != candidate_calls:
        raise ValueError("RVIR-v4 source-parity call topology/schema differs")
    baseline_core = _one_core(baseline)
    candidate_core = _one_core(candidate)
    exact_core_fields = (
        "core_id",
        "branching_decision",
        "branching_points",
        "split_depth",
        "batch_size",
        "n_verified",
        "n_splits",
    )
    if any(
        baseline_core.get(name) != candidate_core.get(name)
        for name in exact_core_fields
    ):
        raise ValueError("RVIR-v4 source-parity core structure differs")
    if _mutation_structure(baseline_core) != _mutation_structure(candidate_core):
        raise ValueError("RVIR-v4 source-parity mutation structure differs")
    baseline_pre = production_snapshot_from_payload_v4(baseline_core["pre_snapshot"])
    candidate_pre = production_snapshot_from_payload_v4(candidate_core["pre_snapshot"])
    baseline_post = production_snapshot_from_payload_v4(baseline_core["post_snapshot"])
    candidate_post = production_snapshot_from_payload_v4(
        candidate_core["post_snapshot"]
    )
    pre = compare_snapshots(baseline_pre, candidate_pre)
    post = compare_snapshots(baseline_post, candidate_post)
    lower_absolute, lower_relative = _numeric_diff(
        cast(torch.Tensor, baseline_core["lower"]),
        cast(torch.Tensor, candidate_core["lower"]),
    )
    upper_absolute, upper_relative = _numeric_diff(
        cast(torch.Tensor, baseline_core["upper"]),
        cast(torch.Tensor, candidate_core["upper"]),
    )
    return {
        "source_protocol_solver_exact": True,
        "call_topology_schema_exact": True,
        "call_count": len(baseline_calls),
        "core_structure_exact": True,
        "mutation_path_changed_flags_exact": True,
        "pre_snapshot": pre,
        "post_snapshot": post,
        "result_lower_maximum_absolute_difference": lower_absolute,
        "result_lower_maximum_relative_difference": lower_relative,
        "result_upper_maximum_absolute_difference": upper_absolute,
        "result_upper_maximum_relative_difference": upper_relative,
        "result_finite_mask_exact": True,
        "result_sign_exact": True,
    }


def build_report(
    baseline_artifact: Path, candidate_artifact: Path
) -> dict[str, object]:
    """Replay both artifacts and build the independent cross-artifact parity report."""

    baseline_artifact = baseline_artifact.resolve()
    candidate_artifact = candidate_artifact.resolve()
    baseline_replay = production_artifact._replay(baseline_artifact)
    candidate_replay = optimizer_artifact._replay(candidate_artifact)
    baseline_path = baseline_artifact / production_artifact.CAPTURE_FILE
    candidate_path = candidate_artifact / optimizer_artifact.WORKER_CAPTURE_FILE
    parity = compare_captures(
        _load_capture(baseline_path), _load_capture(candidate_path)
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "verifier_code_sha256": _file_sha256(Path(__file__).resolve()),
        "baseline_manifest_sha256": _file_sha256(baseline_artifact / "manifest.json"),
        "candidate_manifest_sha256": _file_sha256(candidate_artifact / "manifest.json"),
        "baseline_capture_sha256": _file_sha256(baseline_path),
        "candidate_capture_sha256": _file_sha256(candidate_path),
        "atol": ATOL,
        "rtol": RTOL,
        "baseline_replay": baseline_replay,
        "candidate_replay": candidate_replay,
        "parity": parity,
        "status": "source-parity-passed",
        "performance_claimed": False,
    }
    report["report_hash"] = _canonical_hash(report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Build and write the formal source-parity report."""

    args = _parse_args()
    report = build_report(args.baseline_artifact, args.candidate_artifact)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.report, report)
    print(_canonical_json(report))


if __name__ == "__main__":
    main()
