"""Generate/replay architecture-held-out IR-5 evidence with fair batching."""

# pylint: disable=duplicate-code,too-many-arguments,too-many-locals
# pylint: disable=too-many-positional-arguments,too-many-statements

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Any, Iterable, Sequence

import torch

from boundflow.ir.plan import BackendKind
from boundflow.planner.adaptive_plan_evaluator import AdaptiveEvaluationContext
from boundflow.planner.fair_adaptive_plan_evaluator import (
    FairAdaptivePlanPolicy,
    FairAdaptivePolicyOutcome,
    FairBaselinePlanIds,
    evaluate_fair_adaptive_plan_policies,
    summarize_fair_adaptive_outcomes,
)
from boundflow.planner.fair_batching_measurement import (
    batched_original_observation,
    compiler_candidate_observation,
    fixed_single_observation,
    measure_batched_original,
    measure_batched_original_from_forward_trace,
    ordinary_batching_observation,
    verify_single_query_matches_batch,
)
from boundflow.planner.measured_adaptive_benchmark import (
    CandidateMeasurement,
    MeasuredWorkloadSpec,
    TypedCNNWorkloadSpec,
    TypedResidualCNNWorkloadSpec,
    TypedWorkloadSpec,
    fit_backend_calibrations,
    measure_workload,
)
from boundflow.planner.typed_benchmark_workloads import (
    build_cnn_candidate,
    build_residual_cnn_candidate,
)

ARTIFACT_SCHEMA = "boundflow.ir5-family-fair-artifact/v1"
RESIDUAL_FINAL_ARTIFACT_SCHEMA = "boundflow.ir5-residual-final-artifact/v2"
RESIDUAL_FINAL_V3_ARTIFACT_SCHEMA = "boundflow.ir5-residual-final-artifact/v3"
CALIBRATION_WORKLOADS = (
    TypedWorkloadSpec("calibration-mlp-small", "calibration", 4, 32, 32, 8, 6101),
    TypedWorkloadSpec("calibration-mlp-wide", "calibration", 4, 96, 96, 10, 6102),
)
HELDOUT_WORKLOADS = (
    TypedCNNWorkloadSpec("heldout-cnn-gray", "heldout", 4, 1, 16, 4, 8, 10, 6201),
    TypedCNNWorkloadSpec("heldout-cnn-color", "heldout", 4, 3, 16, 8, 12, 10, 6202),
)
RESIDUAL_CALIBRATION_WORKLOADS = (
    TypedCNNWorkloadSpec(
        "calibration-chain-gray", "calibration", 4, 1, 16, 4, 8, 10, 7201
    ),
    TypedCNNWorkloadSpec(
        "calibration-chain-color", "calibration", 4, 3, 16, 8, 12, 10, 7202
    ),
)
RESIDUAL_HELDOUT_WORKLOADS = (
    TypedResidualCNNWorkloadSpec(
        "final-residual-gray-v2", "heldout", 4, 1, 14, 5, 12, 7401
    ),
    TypedResidualCNNWorkloadSpec(
        "final-residual-color-v2", "heldout", 4, 3, 18, 7, 12, 7402
    ),
)
RESIDUAL_V3_HELDOUT_WORKLOADS = (
    TypedResidualCNNWorkloadSpec(
        "final-residual-gray-v3", "heldout", 4, 1, 14, 5, 12, 7501
    ),
    TypedResidualCNNWorkloadSpec(
        "final-residual-color-v3", "heldout", 4, 3, 18, 7, 12, 7502
    ),
)
BACKENDS = (
    BackendKind.REFERENCE,
    BackendKind.PYTORCH_DENSE,
    BackendKind.PYTORCH_CHUNKED,
    BackendKind.TVM_FUSED_TIR,
)
COMPILER_BACKENDS = BACKENDS[1:]
HIGH_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024
LOW_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024
FILES = (
    "split.json",
    "calibration.jsonl",
    "calibration_models.json",
    "heldout_compiler.jsonl",
    "fixed_single.jsonl",
    "batched_original.jsonl",
    "semantic_batch_checks.jsonl",
    "outcomes.jsonl",
    "summary.json",
)


@dataclass(frozen=True)
class ArtifactSuite:  # pylint: disable=too-many-instance-attributes
    """One immutable calibration→held-out artifact protocol."""

    name: str
    schema_version: str
    calibration_family: str
    heldout_family: str
    calibration: tuple[MeasuredWorkloadSpec, ...]
    heldout: tuple[TypedCNNWorkloadSpec | TypedResidualCNNWorkloadSpec, ...]
    fit_scope: str
    evidence_scope: str
    from_forward_trace: bool
    explicit_single_input_slice: bool = False

    @property
    def baseline_plan_id(self) -> str:
        """Return the exact baseline identity required by this suite."""

        return (
            "batched-original-from-forward-trace"
            if self.from_forward_trace
            else "batched-original"
        )


FAMILY_FAIR_V1 = ArtifactSuite(
    name="family-fair-v1",
    schema_version=ARTIFACT_SCHEMA,
    calibration_family="mlp",
    heldout_family="chain_cnn",
    calibration=CALIBRATION_WORKLOADS,
    heldout=HELDOUT_WORKLOADS,
    fit_scope="mlp_calibration_only_before_cnn_heldout",
    evidence_scope="mlp_calibration_to_chain_cnn_heldout_with_fair_batching",
    from_forward_trace=False,
)
RESIDUAL_FINAL_V2 = ArtifactSuite(
    name="residual-final-v2",
    schema_version=RESIDUAL_FINAL_ARTIFACT_SCHEMA,
    calibration_family="chain_cnn",
    heldout_family="residual_cnn",
    calibration=RESIDUAL_CALIBRATION_WORKLOADS,
    heldout=RESIDUAL_HELDOUT_WORKLOADS,
    fit_scope="chain_cnn_calibration_only_before_residual_cnn_final",
    evidence_scope=(
        "chain_cnn_calibration_to_residual_cnn_final_with_from_trace_fair_batching"
    ),
    from_forward_trace=True,
)
RESIDUAL_FINAL_V3 = ArtifactSuite(
    name="residual-final-v3",
    schema_version=RESIDUAL_FINAL_V3_ARTIFACT_SCHEMA,
    calibration_family="chain_cnn",
    heldout_family="residual_cnn",
    calibration=RESIDUAL_CALIBRATION_WORKLOADS,
    heldout=RESIDUAL_V3_HELDOUT_WORKLOADS,
    fit_scope="chain_cnn_calibration_only_before_residual_cnn_final_v3",
    evidence_scope=(
        "chain_cnn_calibration_to_residual_cnn_final_v3_with_exact_input_slice"
    ),
    from_forward_trace=True,
    explicit_single_input_slice=True,
)
SUITES = {item.name: item for item in (FAMILY_FAIR_V1, RESIDUAL_FINAL_V3)}


def generate_artifact(
    out_dir: Path,
    *,
    device: str,
    warm_samples: int,
    suite: ArtifactSuite = FAMILY_FAIR_V1,
) -> None:
    """Run calibration, architecture-held-out candidates, and fair baselines."""

    if suite.heldout_family == "residual_cnn" and device != "cuda":
        raise ValueError("IR-5 residual final suite is CUDA-only")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"artifact output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "tvm_cache"
    _write_json(
        out_dir / "split.json",
        {
            "schema_version": suite.schema_version,
            "suite": suite.name,
            "device": device,
            "warm_samples": warm_samples,
            "calibration_family": suite.calibration_family,
            "heldout_family": suite.heldout_family,
            "calibration": [item.to_dict() for item in suite.calibration],
            "heldout": [item.to_dict() for item in suite.heldout],
            "backends": [item.value for item in BACKENDS],
            "physical_batch_contract": (
                "one box is one query; batch latency divided by exact batch size; "
                "compile/setup is charged once and is never divided"
            ),
            "timed_solver_contract": (
                "CROWN backward from a precomputed forward trace"
                if suite.from_forward_trace
                else "full legacy CROWN including forward trace"
            ),
            **(
                {
                    "single_query_binding_contract": (
                        "exact_clone_of_batched_input_center_first_query"
                    )
                }
                if suite.explicit_single_input_slice
                else {}
            ),
            "context_contract": _context_contract(),
            "freeze_scope": "all constants frozen before artifact generation",
        },
    )
    calibration = _measure_many(
        suite.calibration,
        BACKENDS,
        device=device,
        warm_samples=warm_samples,
        cache_root=cache_root / "calibration",
    )
    _write_jsonl(out_dir / "calibration.jsonl", _records(calibration))
    models = fit_backend_calibrations(calibration)
    _write_json(
        out_dir / "calibration_models.json",
        {
            "schema_version": suite.schema_version,
            "fit_scope": suite.fit_scope,
            "models": [
                models[backend].to_dict() for backend in sorted(models, key=str)
            ],
        },
    )

    heldout = _measure_many(
        suite.heldout,
        BACKENDS,
        device=device,
        warm_samples=warm_samples,
        cache_root=cache_root / "heldout",
    )
    _write_jsonl(out_dir / "heldout_compiler.jsonl", _records(heldout))
    fixed_rows: list[CandidateMeasurement] = []
    original_rows = []
    semantic_checks = []
    all_outcomes: list[FairAdaptivePolicyOutcome] = []
    context_rows: list[dict[str, object]] = []
    for workload in suite.heldout:
        selected = tuple(
            item
            for item in heldout
            if item.workload.workload_id == workload.workload_id
        )
        reference = next(
            item for item in selected if item.backend == BackendKind.REFERENCE
        )
        batched_prepared = _prepared_convolutional(
            workload, batch=workload.batch, device=device
        )
        single_spec = _single_spec(workload)
        single_measurement = measure_workload(
            single_spec,
            (BackendKind.REFERENCE,),
            device=device,
            warm_samples=warm_samples,
            cache_root=cache_root / "fixed-single",
        )[0]
        fixed_rows.append(single_measurement)
        single_prepared = _prepared_convolutional(
            single_spec,
            batch=1,
            device=device,
            input_center=(
                batched_prepared.input_spec.center[:1].detach().clone()
                if suite.explicit_single_input_slice
                else None
            ),
        )
        semantic_checks.append(
            {
                "workload_id": workload.workload_id,
                **(
                    _verify_input_slice_identity(batched_prepared, single_prepared)
                    if suite.explicit_single_input_slice
                    else {}
                ),
                **verify_single_query_matches_batch(batched_prepared, single_prepared),
            }
        )
        original = (
            measure_batched_original_from_forward_trace(
                batched_prepared,
                workload,
                device=device,
                warm_samples=warm_samples,
            )
            if suite.from_forward_trace
            else measure_batched_original(
                batched_prepared,
                workload,
                device=device,
                warm_samples=warm_samples,
            )
        )
        original_rows.append(original)
        observations = [
            fixed_single_observation(single_measurement),
            ordinary_batching_observation(reference),
            batched_original_observation(original),
            *(
                compiler_candidate_observation(item, models[item.backend])
                for item in selected
                if item.backend in COMPILER_BACKENDS
            ),
        ]
        artifact_keys = tuple(
            sorted(
                item.compiled_artifact_key
                for item in observations
                if item.compiled_artifact_key is not None
            )
        )
        contexts = _contexts(workload.workload_id, artifact_keys)
        context_rows.extend(
            {
                "workload_id": workload.workload_id,
                "context_id": item.context_id,
                "memory_budget_bytes": item.memory_budget_bytes,
                "expected_query_count": item.expected_query_count,
                "cached_artifact_keys": list(item.cached_artifact_keys),
                "budget_source": "frozen_split_contract",
            }
            for item in contexts
        )
        all_outcomes.extend(
            evaluate_fair_adaptive_plan_policies(
                contexts,
                observations,
                baselines=FairBaselinePlanIds(
                    "fixed-single",
                    "ordinary-batching",
                    suite.baseline_plan_id,
                ),
                selectable_plan_ids=tuple(
                    f"compiler:{backend.value}" for backend in COMPILER_BACKENDS
                ),
            )
        )
    _write_jsonl(out_dir / "fixed_single.jsonl", _records(fixed_rows))
    _write_jsonl(
        out_dir / "batched_original.jsonl",
        (item.to_dict() for item in original_rows),
    )
    _write_jsonl(out_dir / "semantic_batch_checks.jsonl", semantic_checks)
    _write_jsonl(
        out_dir / "outcomes.jsonl",
        (item.to_dict() for item in all_outcomes),
    )
    _write_json(
        out_dir / "summary.json",
        _summary(
            calibration,
            heldout,
            fixed_rows,
            original_rows,
            semantic_checks,
            all_outcomes,
            context_rows,
            suite,
        ),
    )
    _write_json(
        out_dir / "manifest.json",
        {
            "schema_version": suite.schema_version,
            "suite": suite.name,
            "evidence_scope": suite.evidence_scope,
            "environment": _environment(device),
            "files": {name: _sha256(out_dir / name) for name in FILES},
        },
    )


def replay_artifact(
    artifact_dir: Path,
    *,
    semantic: bool,
    suite: ArtifactSuite = FAMILY_FAIR_V1,
) -> None:
    """Verify content addresses, split freeze, and optional batch semantics."""

    manifest = _read_json(artifact_dir / "manifest.json")
    if (
        manifest.get("schema_version") != suite.schema_version
        or manifest.get("suite", FAMILY_FAIR_V1.name) != suite.name
    ):
        raise ValueError("IR-5 family artifact manifest schema mismatch")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or set(expected_files) != set(FILES):
        raise ValueError("IR-5 family artifact file set mismatch")
    for name in FILES:
        if _sha256(artifact_dir / name) != expected_files[name]:
            raise ValueError(f"IR-5 family artifact digest mismatch: {name}")
    split = _read_json(artifact_dir / "split.json")
    expected_split = {
        "calibration": [item.to_dict() for item in suite.calibration],
        "heldout": [item.to_dict() for item in suite.heldout],
        "context_contract": _context_contract(),
    }
    if suite.explicit_single_input_slice:
        expected_split["single_query_binding_contract"] = (
            "exact_clone_of_batched_input_center_first_query"
        )
    if any(split.get(key) != value for key, value in expected_split.items()):
        raise ValueError("IR-5 family split/context drift")
    if semantic:
        device = str(split["device"])
        for workload in suite.heldout:
            batched = _prepared_convolutional(
                workload, batch=workload.batch, device=device
            )
            single = _prepared_convolutional(
                _single_spec(workload),
                batch=1,
                device=device,
                input_center=(
                    batched.input_spec.center[:1].detach().clone()
                    if suite.explicit_single_input_slice
                    else None
                ),
            )
            if suite.explicit_single_input_slice:
                _verify_input_slice_identity(batched, single)
            verify_single_query_matches_batch(batched, single)
            if suite.from_forward_trace:
                measure_batched_original_from_forward_trace(
                    batched,
                    workload,
                    device=device,
                    warm_samples=1,
                )
            else:
                measure_batched_original(
                    batched,
                    workload,
                    device=device,
                    warm_samples=1,
                )


def _measure_many(
    workloads,
    backends,
    *,
    device: str,
    warm_samples: int,
    cache_root: Path,
) -> tuple[CandidateMeasurement, ...]:
    measured: list[CandidateMeasurement] = []
    for workload in workloads:
        measured.extend(
            measure_workload(
                workload,
                backends,
                device=device,
                warm_samples=warm_samples,
                cache_root=cache_root,
            )
        )
    return tuple(measured)


def _single_spec(
    workload: TypedCNNWorkloadSpec | TypedResidualCNNWorkloadSpec,
) -> TypedCNNWorkloadSpec | TypedResidualCNNWorkloadSpec:
    if isinstance(workload, TypedResidualCNNWorkloadSpec):
        return TypedResidualCNNWorkloadSpec(
            f"{workload.workload_id}:single",
            "heldout",
            1,
            workload.input_channels,
            workload.image_size,
            workload.block_channels,
            workload.output_dim,
            workload.seed,
        )
    return TypedCNNWorkloadSpec(
        f"{workload.workload_id}:single",
        "heldout",
        1,
        workload.input_channels,
        workload.image_size,
        workload.conv1_channels,
        workload.conv2_channels,
        workload.output_dim,
        workload.seed,
    )


def _prepared_convolutional(
    workload: TypedCNNWorkloadSpec | TypedResidualCNNWorkloadSpec,
    *,
    batch: int,
    device: str = "cuda",
    input_center: torch.Tensor | None = None,
):
    if isinstance(workload, TypedResidualCNNWorkloadSpec):
        return build_residual_cnn_candidate(
            workload_id=workload.workload_id,
            backend=BackendKind.REFERENCE,
            device=device,
            batch=batch,
            input_channels=workload.input_channels,
            image_size=workload.image_size,
            block_channels=workload.block_channels,
            output_dim=workload.output_dim,
            seed=workload.seed,
            input_center=input_center,
        )
    return build_cnn_candidate(
        workload_id=workload.workload_id,
        backend=BackendKind.REFERENCE,
        device=device,
        batch=batch,
        input_channels=workload.input_channels,
        image_size=workload.image_size,
        conv1_channels=workload.conv1_channels,
        conv2_channels=workload.conv2_channels,
        output_dim=workload.output_dim,
        seed=workload.seed,
        input_center=input_center,
    )


def _contexts(
    workload_id: str, cached_artifact_keys: tuple[str, ...]
) -> tuple[AdaptiveEvaluationContext, ...]:
    return (
        AdaptiveEvaluationContext(
            f"{workload_id}:cold-single", HIGH_MEMORY_BUDGET_BYTES, 1
        ),
        AdaptiveEvaluationContext(
            f"{workload_id}:cold-repeated", HIGH_MEMORY_BUDGET_BYTES, 64
        ),
        AdaptiveEvaluationContext(
            f"{workload_id}:warm-single",
            HIGH_MEMORY_BUDGET_BYTES,
            1,
            cached_artifact_keys,
        ),
        AdaptiveEvaluationContext(
            f"{workload_id}:low-memory", LOW_MEMORY_BUDGET_BYTES, 16
        ),
    )


def _verify_input_slice_identity(batched, single) -> dict[str, object]:
    """Fail unless fixed-single owns an exact clone of batch query zero."""

    expected = batched.input_spec.center[:1]
    actual = single.input_spec.center
    exact = bool(torch.equal(expected, actual))
    max_diff = float((expected - actual).abs().max().item())
    if not exact:
        raise ValueError("fixed-single input is not the exact batched first query")
    return {
        "input_center_exact": exact,
        "input_center_max_abs_diff": max_diff,
    }


def _context_contract() -> dict[str, object]:
    return {
        "high_memory_budget_bytes": HIGH_MEMORY_BUDGET_BYTES,
        "low_memory_budget_bytes": LOW_MEMORY_BUDGET_BYTES,
        "expected_query_counts": {
            "cold_single": 1,
            "cold_repeated": 64,
            "warm_single": 1,
            "low_memory": 16,
        },
    }


def _summary(
    calibration,
    heldout,
    fixed_rows,
    original_rows,
    semantic_checks,
    outcomes,
    contexts,
    suite: ArtifactSuite,
) -> dict[str, object]:
    global_choices = {
        item.context_id: item.selected_plan_id
        for item in outcomes
        if item.policy == FairAdaptivePlanPolicy.GLOBAL
    }
    switches = {}
    for workload in suite.heldout:
        high = global_choices[f"{workload.workload_id}:cold-repeated"]
        low = global_choices[f"{workload.workload_id}:low-memory"]
        switches[workload.workload_id] = {
            "high_memory_plan": high,
            "low_memory_plan": low,
            "switched": high != low,
        }
    summary = summarize_fair_adaptive_outcomes(outcomes)
    global_summary = summary.get("global")
    if not isinstance(global_summary, dict):
        raise ValueError("fair summary omits Global policy")
    global_regret_p90_value = global_summary.get("regret_p90")
    if not isinstance(global_regret_p90_value, (int, float)):
        raise ValueError("fair summary omits Global p90 regret")
    global_regret_p90 = float(global_regret_p90_value)
    pareto = {
        workload.workload_id: _compiler_latency_memory_pareto(
            tuple(
                item
                for item in heldout
                if item.workload.workload_id == workload.workload_id
                and item.backend in COMPILER_BACKENDS
            )
        )
        for workload in suite.heldout
    }
    global_feasible = sum(
        item.feasible
        for item in outcomes
        if item.policy == FairAdaptivePlanPolicy.GLOBAL
    )
    return {
        "schema_version": suite.schema_version,
        "suite": suite.name,
        "evidence_scope": suite.evidence_scope,
        "counts": {
            "calibration_measurements": len(calibration),
            "heldout_compiler_measurements": len(heldout),
            "fixed_single_measurements": len(fixed_rows),
            "batched_original_measurements": len(original_rows),
            "semantic_batch_checks": len(semantic_checks),
            "contexts": len(contexts),
            "policy_outcomes": len(outcomes),
        },
        "gates": {
            "architecture_families_disjoint": True,
            "all_compiler_candidates_semantic_allclose": all(
                item.semantic_allclose for item in heldout
            ),
            "all_batched_original_semantic_allclose": all(
                item.semantic_allclose for item in original_rows
            ),
            "all_fixed_single_match_batched_first_query": all(
                item["semantic_allclose"] for item in semantic_checks
            ),
            "all_policies_evaluated": len(outcomes)
            == len(contexts) * len(FairAdaptivePlanPolicy),
            "global_feasible_all_contexts": global_feasible == len(contexts),
            "any_multi_budget_global_switch": any(
                item["switched"] for item in switches.values()
            ),
            "global_p90_regret_lte_1_20": global_regret_p90 <= 1.20,
            "compiler_latency_memory_pareto_all_workloads": all(
                item["has_tradeoff"] for item in pareto.values()
            ),
        },
        "policy_summary": summary,
        "multi_budget_global_switches": switches,
        "compiler_latency_memory_pareto": pareto,
        "contexts": contexts,
        "limitations": [
            (
                "held-out is a reduced residual-CNN family, not VNN-COMP"
                if suite.heldout_family == "residual_cnn"
                else "held-out is a reduced chain-CNN family, not residual/VNN-COMP"
            ),
            "calibration uses a median one-feature MAC proxy",
            "ordinary batching is typed reference; batched original is legacy plain CROWN",
            "per-query latency is physical batch wall time divided by batch size",
            "wall-clock samples are not exact-replayed",
        ],
    }


def _compiler_latency_memory_pareto(
    measurements: tuple[CandidateMeasurement, ...],
) -> dict[str, object]:
    """Require at least two non-dominated compiler points with a real tradeoff."""

    points = tuple(
        (
            item.backend.value,
            statistics.median(item.warm_latency_ms) / item.workload.batch,
            item.measured_peak_bytes,
        )
        for item in measurements
    )
    frontier = tuple(
        point
        for point in points
        if not any(
            other[1] <= point[1]
            and other[2] <= point[2]
            and (other[1] < point[1] or other[2] < point[2])
            for other in points
            if other[0] != point[0]
        )
    )
    has_tradeoff = (
        len(frontier) >= 2 and len({(point[1], point[2]) for point in frontier}) >= 2
    )
    return {
        "has_tradeoff": has_tradeoff,
        "frontier": [
            {
                "backend": backend,
                "median_latency_ms_per_query": latency,
                "measured_peak_bytes": memory,
            }
            for backend, latency, memory in frontier
        ],
    }


def _records(
    measurements: Sequence[CandidateMeasurement],
) -> Iterable[dict[str, object]]:
    return (item.to_dict() for item in measurements)


def _environment(device: str) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if device == "cuda" and torch.cuda.is_available()
            else None
        ),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.write_text(
        "".join(_canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Run fresh family evaluation or deterministic artifact replay."""

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--out-dir", type=Path, required=True)
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    generate.add_argument("--warm-samples", type=int, default=9)
    generate.add_argument("--suite", choices=tuple(SUITES), default=FAMILY_FAIR_V1.name)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    replay.add_argument("--semantic", action="store_true")
    replay.add_argument("--suite", choices=tuple(SUITES), default=FAMILY_FAIR_V1.name)
    args = parser.parse_args(argv)
    suite = SUITES[args.suite]
    if args.command == "generate":
        generate_artifact(
            args.out_dir,
            device=args.device,
            warm_samples=args.warm_samples,
            suite=suite,
        )
    else:
        replay_artifact(args.artifact_dir, semantic=args.semantic, suite=suite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
