"""Generate and replay the IR-5 held-out typed adaptive-plan artifact."""

# pylint: disable=duplicate-code,too-many-locals,too-many-statements

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import torch

from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.ir.plan import BackendKind
from boundflow.planner.adaptive_plan_evaluator import (
    AdaptiveEvaluationContext,
    AdaptivePlanPolicy,
    AdaptivePolicyOutcome,
    evaluate_adaptive_plan_policies,
    summarize_adaptive_outcomes,
)
from boundflow.planner.measured_adaptive_benchmark import (
    CandidateMeasurement,
    TypedWorkloadSpec,
    build_heldout_observations,
    fit_backend_calibrations,
    measure_workload,
)
from boundflow.planner.typed_benchmark_workloads import build_mlp_candidate
from boundflow.runtime.task_backend_dispatch import TypedTaskBackendRegistry
from boundflow.runtime.task_ir_executor import execute_task_ir_semantics

ARTIFACT_SCHEMA = "boundflow.ir5-heldout-adaptive-artifact/v1"
CALIBRATION_WORKLOADS = (
    TypedWorkloadSpec("calibration-small", "calibration", 2, 48, 96, 12, 5101),
    TypedWorkloadSpec("calibration-wide", "calibration", 4, 96, 160, 24, 5102),
)
HELDOUT_WORKLOADS = (
    TypedWorkloadSpec("heldout-medium", "heldout", 3, 80, 144, 20, 5201),
    TypedWorkloadSpec("heldout-large", "heldout", 4, 160, 256, 32, 5202),
)
DEFAULT_BACKENDS = (
    BackendKind.REFERENCE,
    BackendKind.PYTORCH_DENSE,
    BackendKind.PYTORCH_CHUNKED,
    BackendKind.TVM_FUSED_TIR,
)
HIGH_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024
LOW_MEMORY_BUDGETS = (
    ("heldout-medium", 8_800_000),
    ("heldout-large", 9_400_000),
)
FILES = (
    "split.json",
    "calibration.jsonl",
    "calibration_models.json",
    "heldout.jsonl",
    "outcomes.jsonl",
    "summary.json",
)


def generate_artifact(
    out_dir: Path,
    *,
    device: str,
    warm_samples: int,
    backends: Sequence[BackendKind],
) -> None:
    """Measure frozen splits and write a content-addressed evidence directory."""

    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"artifact output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "tvm_cache"
    split_payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "freeze_rule": (
            "predictions are fitted from calibration.jsonl before heldout.jsonl "
            "is joined; held-out measurements are used only for evaluation/oracle"
        ),
        "device": device,
        "warm_samples": warm_samples,
        "backends": [backend.value for backend in backends],
        "calibration": [item.to_dict() for item in CALIBRATION_WORKLOADS],
        "heldout": [item.to_dict() for item in HELDOUT_WORKLOADS],
        "context_contract": _context_contract_payload(),
    }
    _write_json(out_dir / "split.json", split_payload)

    calibration = _measure_split(
        CALIBRATION_WORKLOADS,
        backends,
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
            "fit_scope": "calibration_only",
            "models": [
                models[backend].to_dict() for backend in sorted(models, key=str)
            ],
        },
    )

    heldout = _measure_split(
        HELDOUT_WORKLOADS,
        backends,
        device=device,
        warm_samples=warm_samples,
        cache_root=cache_root / "heldout",
    )
    _write_jsonl(out_dir / "heldout.jsonl", _records(heldout))
    all_outcomes: list[AdaptivePolicyOutcome] = []
    context_records: list[dict[str, object]] = []
    for workload in HELDOUT_WORKLOADS:
        selected = tuple(
            item
            for item in heldout
            if item.workload.workload_id == workload.workload_id
        )
        observations = build_heldout_observations(selected, models)
        artifact_keys = tuple(
            sorted(
                item.compiled_artifact_key
                for item in selected
                if item.compiled_artifact_key is not None
            )
        )
        contexts = (
            AdaptiveEvaluationContext(
                f"{workload.workload_id}:cold-single",
                HIGH_MEMORY_BUDGET_BYTES,
                1,
            ),
            AdaptiveEvaluationContext(
                f"{workload.workload_id}:cold-repeated",
                HIGH_MEMORY_BUDGET_BYTES,
                100,
            ),
            AdaptiveEvaluationContext(
                f"{workload.workload_id}:warm-single",
                HIGH_MEMORY_BUDGET_BYTES,
                1,
                artifact_keys,
            ),
            AdaptiveEvaluationContext(
                f"{workload.workload_id}:low-memory",
                _low_memory_budget(workload.workload_id),
                10,
            ),
        )
        context_records.extend(
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
            evaluate_adaptive_plan_policies(
                contexts,
                observations,
                fixed_plan_id=BackendKind.REFERENCE.value,
            )
        )
    _write_jsonl(
        out_dir / "outcomes.jsonl",
        (item.to_dict() for item in all_outcomes),
    )
    summary = _build_summary(calibration, heldout, all_outcomes, context_records)
    _write_json(out_dir / "summary.json", summary)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA,
        "evidence_scope": "heldout_measured_typed_schedule",
        "environment": _environment(device),
        "files": {name: _sha256(out_dir / name) for name in FILES},
    }
    _write_json(out_dir / "manifest.json", manifest)


def replay_artifact(artifact_dir: Path, *, semantic: bool) -> None:
    """Validate file integrity, split freeze, and optionally reference semantics."""

    manifest = _read_json(artifact_dir / "manifest.json")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("IR-5 artifact manifest schema mismatch")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or set(expected_files) != set(FILES):
        raise ValueError("IR-5 artifact file set mismatch")
    for name in FILES:
        if _sha256(artifact_dir / name) != expected_files[name]:
            raise ValueError(f"IR-5 artifact digest mismatch: {name}")
    split = _read_json(artifact_dir / "split.json")
    if split.get("calibration") != [item.to_dict() for item in CALIBRATION_WORKLOADS]:
        raise ValueError("IR-5 calibration split drift")
    if split.get("heldout") != [item.to_dict() for item in HELDOUT_WORKLOADS]:
        raise ValueError("IR-5 held-out split drift")
    if split.get("context_contract") != _context_contract_payload():
        raise ValueError("IR-5 frozen resource context drift")
    calibration = _read_jsonl(artifact_dir / "calibration.jsonl")
    heldout = _read_jsonl(artifact_dir / "heldout.jsonl")
    if any(_workload_field(item, "split") != "calibration" for item in calibration):
        raise ValueError("held-out row leaked into calibration artifact")
    if any(_workload_field(item, "split") != "heldout" for item in heldout):
        raise ValueError("calibration row leaked into held-out artifact")
    if semantic:
        device = str(split["device"])
        for workload in (*CALIBRATION_WORKLOADS, *HELDOUT_WORKLOADS):
            expected = next(
                item
                for item in (*calibration, *heldout)
                if _workload_field(item, "workload_id") == workload.workload_id
                and item["backend"] == BackendKind.REFERENCE.value
            )
            lower_hash, upper_hash = _replay_reference(workload, device)
            if (
                lower_hash != expected["reference_lower_hash"]
                or upper_hash != expected["reference_upper_hash"]
            ):
                raise ValueError(f"IR-5 semantic replay drift: {workload.workload_id}")


def _measure_split(
    workloads: Sequence[TypedWorkloadSpec],
    backends: Sequence[BackendKind],
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


def _low_memory_budget(workload_id: str) -> int:
    budgets = dict(LOW_MEMORY_BUDGETS)
    if workload_id not in budgets:
        raise ValueError(f"missing frozen memory budget: {workload_id}")
    return budgets[workload_id]


def _context_contract_payload() -> dict[str, object]:
    return {
        "high_memory_budget_bytes": HIGH_MEMORY_BUDGET_BYTES,
        "low_memory_budget_bytes": dict(LOW_MEMORY_BUDGETS),
        "expected_query_counts": {
            "cold_single": 1,
            "cold_repeated": 100,
            "warm_single": 1,
            "low_memory": 10,
        },
        "freeze_scope": "constant_before_final_heldout_measurement",
    }


def _build_summary(calibration, heldout, outcomes, contexts) -> dict[str, object]:
    global_choices = {
        item.context_id: item.selected_plan_id
        for item in outcomes
        if item.policy == AdaptivePlanPolicy.GLOBAL
    }
    multi_budget_switches = {}
    for workload in HELDOUT_WORKLOADS:
        high = global_choices[f"{workload.workload_id}:cold-repeated"]
        low = global_choices[f"{workload.workload_id}:low-memory"]
        multi_budget_switches[workload.workload_id] = {
            "high_memory_plan": high,
            "low_memory_plan": low,
            "switched": high != low,
        }
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "evidence_scope": "heldout_measured_typed_schedule",
        "counts": {
            "calibration_measurements": len(calibration),
            "heldout_measurements": len(heldout),
            "contexts": len(contexts),
            "policy_outcomes": len(outcomes),
        },
        "gates": {
            "all_candidates_semantic_allclose": all(
                item.semantic_allclose for item in (*calibration, *heldout)
            ),
            "calibration_heldout_disjoint": not (
                {item.workload.workload_id for item in calibration}
                & {item.workload.workload_id for item in heldout}
            ),
            "all_policies_evaluated": len(outcomes)
            == len(contexts) * len(AdaptivePlanPolicy),
            "any_multi_budget_global_switch": any(
                item["switched"] for item in multi_budget_switches.values()
            ),
        },
        "multi_budget_global_switches": multi_budget_switches,
        "policy_summary": summarize_adaptive_outcomes(outcomes),
        "contexts": contexts,
        "limitations": [
            "the frozen workload family is deterministic plain-CROWN MLP only",
            "calibration uses a median one-feature shape model, not an autotuner",
            "measured compile/setup is the TVM cache miss/disk-load event total",
            "wall-clock latency is validated statistically, not exact-replayed",
        ],
    }


def _replay_reference(workload: TypedWorkloadSpec, device: str) -> tuple[str, str]:
    prepared = build_mlp_candidate(
        workload_id=workload.workload_id,
        backend=BackendKind.REFERENCE,
        device=device,
        batch=workload.batch,
        input_dim=workload.input_dim,
        hidden_dim=workload.hidden_dim,
        output_dim=workload.output_dim,
        seed=workload.seed,
    )
    result, _trace = execute_task_ir_semantics(
        prepared.task_module,
        prepared.schedule,
        bound_module=prepared.bound_module,
        template=prepared.template,
        instance=prepared.instance,
        legacy_task_module=prepared.legacy_module,
        input_spec=prepared.input_spec,
        relu_pre=prepared.relu_pre,
        backend=TypedTaskBackendRegistry(),
    )
    if device == "cuda":
        torch.cuda.synchronize()
    return tensor_content_hash(result.lower), tensor_content_hash(result.upper)


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _workload_field(row: Mapping[str, object], field: str) -> object:
    workload = row.get("workload")
    if not isinstance(workload, dict) or field not in workload:
        raise ValueError("IR-5 measurement workload payload is invalid")
    return workload[field]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Run fresh measurement or deterministic replay checks."""

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--out-dir", type=Path, required=True)
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    generate.add_argument("--warm-samples", type=int, default=9)
    generate.add_argument(
        "--backends",
        nargs="+",
        choices=tuple(item.value for item in DEFAULT_BACKENDS),
        default=[item.value for item in DEFAULT_BACKENDS],
    )
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    replay.add_argument("--semantic", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_artifact(
            args.out_dir,
            device=args.device,
            warm_samples=args.warm_samples,
            backends=tuple(BackendKind(item) for item in args.backends),
        )
    else:
        replay_artifact(args.artifact_dir, semantic=args.semantic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
