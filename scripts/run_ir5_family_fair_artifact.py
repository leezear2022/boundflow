"""Generate/replay architecture-held-out IR-5 evidence with fair batching."""

# pylint: disable=duplicate-code,too-many-arguments,too-many-locals
# pylint: disable=too-many-positional-arguments,too-many-statements

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
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
    ordinary_batching_observation,
    verify_single_query_matches_batch,
)
from boundflow.planner.measured_adaptive_benchmark import (
    CandidateMeasurement,
    TypedCNNWorkloadSpec,
    TypedWorkloadSpec,
    fit_backend_calibrations,
    measure_workload,
)
from boundflow.planner.typed_benchmark_workloads import build_cnn_candidate

ARTIFACT_SCHEMA = "boundflow.ir5-family-fair-artifact/v1"
CALIBRATION_WORKLOADS = (
    TypedWorkloadSpec("calibration-mlp-small", "calibration", 4, 32, 32, 8, 6101),
    TypedWorkloadSpec("calibration-mlp-wide", "calibration", 4, 96, 96, 10, 6102),
)
HELDOUT_WORKLOADS = (
    TypedCNNWorkloadSpec("heldout-cnn-gray", "heldout", 4, 1, 16, 4, 8, 10, 6201),
    TypedCNNWorkloadSpec("heldout-cnn-color", "heldout", 4, 3, 16, 8, 12, 10, 6202),
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


def generate_artifact(
    out_dir: Path,
    *,
    device: str,
    warm_samples: int,
) -> None:
    """Run calibration, architecture-held-out candidates, and fair baselines."""

    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"artifact output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "tvm_cache"
    _write_json(
        out_dir / "split.json",
        {
            "schema_version": ARTIFACT_SCHEMA,
            "device": device,
            "warm_samples": warm_samples,
            "calibration_family": "mlp",
            "heldout_family": "chain_cnn",
            "calibration": [item.to_dict() for item in CALIBRATION_WORKLOADS],
            "heldout": [item.to_dict() for item in HELDOUT_WORKLOADS],
            "backends": [item.value for item in BACKENDS],
            "physical_batch_contract": (
                "one box is one query; batch latency divided by exact batch size; "
                "compile/setup is charged once and is never divided"
            ),
            "context_contract": _context_contract(),
            "freeze_scope": "all constants frozen before artifact generation",
        },
    )
    calibration = _measure_many(
        CALIBRATION_WORKLOADS,
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
            "schema_version": ARTIFACT_SCHEMA,
            "fit_scope": "mlp_calibration_only_before_cnn_heldout",
            "models": [
                models[backend].to_dict() for backend in sorted(models, key=str)
            ],
        },
    )

    heldout = _measure_many(
        HELDOUT_WORKLOADS,
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
    for workload in HELDOUT_WORKLOADS:
        selected = tuple(
            item
            for item in heldout
            if item.workload.workload_id == workload.workload_id
        )
        reference = next(
            item for item in selected if item.backend == BackendKind.REFERENCE
        )
        batched_prepared = _prepared_cnn(workload, batch=workload.batch, device=device)
        single_spec = _single_spec(workload)
        single_measurement = measure_workload(
            single_spec,
            (BackendKind.REFERENCE,),
            device=device,
            warm_samples=warm_samples,
            cache_root=cache_root / "fixed-single",
        )[0]
        fixed_rows.append(single_measurement)
        single_prepared = _prepared_cnn(single_spec, batch=1, device=device)
        semantic_checks.append(
            {
                "workload_id": workload.workload_id,
                **verify_single_query_matches_batch(batched_prepared, single_prepared),
            }
        )
        original = measure_batched_original(
            batched_prepared,
            workload,
            device=device,
            warm_samples=warm_samples,
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
                    "batched-original",
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
        ),
    )
    _write_json(
        out_dir / "manifest.json",
        {
            "schema_version": ARTIFACT_SCHEMA,
            "evidence_scope": (
                "mlp_calibration_to_chain_cnn_heldout_with_fair_batching"
            ),
            "environment": _environment(device),
            "files": {name: _sha256(out_dir / name) for name in FILES},
        },
    )


def replay_artifact(artifact_dir: Path, *, semantic: bool) -> None:
    """Verify content addresses, split freeze, and optional batch semantics."""

    manifest = _read_json(artifact_dir / "manifest.json")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("IR-5 family artifact manifest schema mismatch")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or set(expected_files) != set(FILES):
        raise ValueError("IR-5 family artifact file set mismatch")
    for name in FILES:
        if _sha256(artifact_dir / name) != expected_files[name]:
            raise ValueError(f"IR-5 family artifact digest mismatch: {name}")
    split = _read_json(artifact_dir / "split.json")
    expected_split = {
        "calibration": [item.to_dict() for item in CALIBRATION_WORKLOADS],
        "heldout": [item.to_dict() for item in HELDOUT_WORKLOADS],
        "context_contract": _context_contract(),
    }
    if any(split.get(key) != value for key, value in expected_split.items()):
        raise ValueError("IR-5 family split/context drift")
    if semantic:
        device = str(split["device"])
        for workload in HELDOUT_WORKLOADS:
            batched = _prepared_cnn(workload, batch=workload.batch, device=device)
            single = _prepared_cnn(_single_spec(workload), batch=1, device=device)
            verify_single_query_matches_batch(batched, single)
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


def _single_spec(workload: TypedCNNWorkloadSpec) -> TypedCNNWorkloadSpec:
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


def _prepared_cnn(
    workload: TypedCNNWorkloadSpec,
    *,
    batch: int,
    device: str = "cuda",
):
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
) -> dict[str, object]:
    global_choices = {
        item.context_id: item.selected_plan_id
        for item in outcomes
        if item.policy == FairAdaptivePlanPolicy.GLOBAL
    }
    switches = {}
    for workload in HELDOUT_WORKLOADS:
        high = global_choices[f"{workload.workload_id}:cold-repeated"]
        low = global_choices[f"{workload.workload_id}:low-memory"]
        switches[workload.workload_id] = {
            "high_memory_plan": high,
            "low_memory_plan": low,
            "switched": high != low,
        }
    summary = summarize_fair_adaptive_outcomes(outcomes)
    global_feasible = sum(
        item.feasible
        for item in outcomes
        if item.policy == FairAdaptivePlanPolicy.GLOBAL
    )
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "evidence_scope": "mlp_calibration_to_chain_cnn_heldout_fair_batching",
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
        },
        "policy_summary": summary,
        "multi_budget_global_switches": switches,
        "contexts": contexts,
        "limitations": [
            "held-out is a reduced chain-CNN family, not residual/VNN-COMP",
            "calibration uses a median one-feature MAC proxy",
            "ordinary batching is typed reference; batched original is legacy plain CROWN",
            "per-query latency is physical batch wall time divided by batch size",
            "wall-clock samples are not exact-replayed",
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
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    replay.add_argument("--semantic", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_artifact(
            args.out_dir,
            device=args.device,
            warm_samples=args.warm_samples,
        )
    else:
        replay_artifact(args.artifact_dir, semantic=args.semantic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
