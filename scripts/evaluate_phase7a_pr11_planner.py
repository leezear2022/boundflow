#!/usr/bin/env python
"""Evaluate PR-11 policies on architecture-family held-out PR-10 profiles."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Optional, Sequence, TextIO

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boundflow.planner.materialization import (  # pylint: disable=wrong-import-position
    BoundMethod,
    MaterializationAction,
    MaterializationContext,
    MaterializationObservation,
    MaterializationPlanRecord,
    MaterializationPolicy,
    OptimizationStage,
    TargetProfile,
    estimate_operator_tree_summary,
    plan_materialization,
    plan_materialization_oracle,
)
from boundflow.planner.materialization_cost_model import (  # pylint: disable=wrong-import-position
    MaterializationCalibrationSample,
    MaterializationCostModel,
    fit_materialization_cost_model,
)

EVAL_SCHEMA_VERSION = "boundflow.pr11-planner-eval/v1"
DEFAULT_CALIBRATION_WORKLOADS = ("mlp_chain", "cnn_chain", "residual_block")
DEFAULT_VALIDATION_WORKLOADS = ("add_concat_dag",)
DEFAULT_HELDOUT_WORKLOADS = ("mini_resnet",)
DEFAULT_BUDGETS_MIB = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
POLICIES = (
    MaterializationPolicy.ALWAYS_DENSE,
    MaterializationPolicy.ALWAYS_STRUCTURED,
    MaterializationPolicy.METHOD_ONLY,
    MaterializationPolicy.MEMORY_THRESHOLD,
    MaterializationPolicy.LOCAL_GREEDY,
    MaterializationPolicy.GLOBAL,
)


@dataclass(frozen=True)
class WorkloadShape:
    """Static features independently derived from the normalized primal graph."""

    output_numel: int
    relu_numels: tuple[int, ...]
    operator_nodes: int


WORKLOAD_SHAPES = {
    "mlp_chain": WorkloadShape(10, (128, 128), 5),
    "cnn_chain": WorkloadShape(10, (2048, 1024), 6),
    "residual_block": WorkloadShape(10, (2048, 2048, 2048), 9),
    "add_concat_dag": WorkloadShape(10, (1024, 1024, 1024), 9),
    "mini_resnet": WorkloadShape(10, (2048,) * 7, 19),
}


@dataclass(frozen=True)
class ProfileCase:
    """Paired dense/structured measurements for one query configuration."""

    run_id: str
    workload: str
    tier: str
    method: str
    spec_size: int
    domain_batch_size: int
    observations: tuple[MaterializationObservation, ...]

    @property
    def case_id(self) -> str:
        """Return a mode-independent query identifier."""

        return (
            f"{self.workload}:{self.method}:s{self.spec_size}:"
            f"d{self.domain_batch_size}"
        )


def _optional_int(value: str) -> Optional[int]:
    return int(value) if value else None


def _optional_float(value: str) -> Optional[float]:
    return float(value) if value else None


# pylint: disable-next=too-many-locals
def load_profile_cases(path: Path) -> list[ProfileCase]:
    """Load and pair normalized PR-10 rows without dropping OOM observations."""

    grouped: dict[tuple[str, str, int, int], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "status",
            "run_id",
            "workload",
            "tier",
            "method",
            "relu_backward_mode",
            "spec_batch",
            "domain_batch",
            "latency_ms_median_trace_off",
            "peak_cuda_allocated_bytes_trace_off",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"profile CSV missing fields: {sorted(missing)}")
        for row in reader:
            key = (
                row["workload"],
                row["method"],
                int(row["spec_batch"]),
                int(row["domain_batch"]),
            )
            entry = grouped.setdefault(
                key,
                {
                    "run_id": row["run_id"],
                    "tier": row["tier"],
                    "observations": {},
                },
            )
            mode = MaterializationAction(row["relu_backward_mode"])
            observations = entry["observations"]
            if not isinstance(observations, dict):  # pragma: no cover - internal guard
                raise TypeError("invalid profile grouping state")
            if mode in observations:
                raise ValueError(f"duplicate profile row for {key} mode={mode.value}")
            observations[mode] = MaterializationObservation(
                action=mode,
                status=row["status"],
                peak_bytes=_optional_int(row["peak_cuda_allocated_bytes_trace_off"]),
                latency_ms=_optional_float(row["latency_ms_median_trace_off"]),
            )

    cases: list[ProfileCase] = []
    for key, entry in sorted(grouped.items()):
        workload, method, spec_size, domain_batch = key
        observations = entry["observations"]
        if not isinstance(observations, dict):  # pragma: no cover - internal guard
            raise TypeError("invalid profile grouping state")
        expected = {
            MaterializationAction.DENSE,
            MaterializationAction.STRUCTURED,
        }
        if set(observations) != expected:
            raise ValueError(f"unpaired profile case {key}: modes={list(observations)}")
        cases.append(
            ProfileCase(
                run_id=str(entry["run_id"]),
                workload=workload,
                tier=str(entry["tier"]),
                method=method,
                spec_size=spec_size,
                domain_batch_size=domain_batch,
                observations=tuple(
                    observations[action]
                    for action in sorted(expected, key=lambda item: item.value)
                ),
            )
        )
    return cases


def _method_state(method: str) -> tuple[BoundMethod, bool, bool, bool]:
    if method == BoundMethod.CROWN.value:
        return BoundMethod.CROWN, False, False, False
    if method == BoundMethod.ALPHA_CROWN.value:
        return BoundMethod.ALPHA_CROWN, True, True, False
    if method == BoundMethod.ALPHA_BETA_CROWN.value:
        return BoundMethod.ALPHA_BETA_CROWN, True, True, True
    raise ValueError(f"unsupported profile method: {method}")


def context_for_case(
    case: ProfileCase, *, memory_budget_bytes: int, safety_margin: float
) -> MaterializationContext:
    """Build a profile-independent static context for one profile case."""

    shape = WORKLOAD_SHAPES.get(case.workload)
    if shape is None:
        raise ValueError(f"missing static workload shape: {case.workload}")
    method, requires_grad, alpha_enabled, beta_enabled = _method_state(case.method)
    summary = estimate_operator_tree_summary(
        domain_batch_size=case.domain_batch_size,
        spec_size=case.spec_size,
        output_numel=shape.output_numel,
        relu_numels=shape.relu_numels,
        element_size=4,
        operator_nodes=shape.operator_nodes,
    )
    return MaterializationContext(
        bound_method=method,
        requires_grad=requires_grad,
        optimization_stage=(
            OptimizationStage.ALPHA_OPTIMIZE
            if requires_grad
            else OptimizationStage.INFERENCE
        ),
        alpha_enabled=alpha_enabled,
        beta_enabled=beta_enabled,
        split_state_present=beta_enabled,
        batch_size=case.domain_batch_size,
        spec_size=case.spec_size,
        domain_batch_size=case.domain_batch_size,
        operator_summary=summary,
        memory_budget_bytes=int(memory_budget_bytes),
        available_memory_bytes=int(memory_budget_bytes),
        safety_margin=float(safety_margin),
        target=TargetProfile(),
    )


def calibration_samples(
    cases: Iterable[ProfileCase], calibration_workloads: set[str]
) -> list[MaterializationCalibrationSample]:
    """Create samples exclusively from named calibration architecture families."""

    samples: list[MaterializationCalibrationSample] = []
    unlimited_budget = 1 << 62
    for case in cases:
        if case.workload not in calibration_workloads:
            continue
        context = context_for_case(
            case, memory_budget_bytes=unlimited_budget, safety_margin=1.0
        )
        samples.extend(
            MaterializationCalibrationSample(context=context, observation=observation)
            for observation in case.observations
        )
    return samples


def _actual_outcome(
    action: MaterializationAction,
    observations: tuple[MaterializationObservation, ...],
    *,
    safe_budget: int,
) -> dict[str, object]:
    if action == MaterializationAction.REDUCE_BATCH:
        return {
            "status": "replan_required",
            "feasible": False,
            "peak_bytes": None,
            "latency_ms": None,
        }
    observation = next(item for item in observations if item.action == action)
    feasible = bool(
        observation.status == "ok"
        and observation.peak_bytes is not None
        and int(observation.peak_bytes) <= int(safe_budget)
    )
    return {
        "status": observation.status,
        "feasible": feasible,
        "peak_bytes": observation.peak_bytes,
        "latency_ms": observation.latency_ms,
    }


def evaluate_cases(  # pylint: disable=too-many-arguments,too-many-locals
    cases: Sequence[ProfileCase],
    *,
    model: MaterializationCostModel,
    calibration_workloads: tuple[str, ...],
    validation_workloads: tuple[str, ...],
    heldout_workloads: tuple[str, ...],
    budgets_bytes: Sequence[int],
    safety_margin: float,
) -> Iterable[dict[str, object]]:
    """Yield held-out policy rows with predictions and measured Oracle outcomes."""

    calibration_set = set(calibration_workloads)
    validation_set = set(validation_workloads)
    heldout_set = set(heldout_workloads)
    overlaps = {
        "calibration/validation": calibration_set.intersection(validation_set),
        "calibration/heldout": calibration_set.intersection(heldout_set),
        "validation/heldout": validation_set.intersection(heldout_set),
    }
    for label, overlap in overlaps.items():
        if overlap:
            raise ValueError(f"{label} workload overlap: {sorted(overlap)}")
    for case in cases:
        if case.workload not in heldout_set:
            continue
        for budget in budgets_bytes:
            raw_context = context_for_case(
                case,
                memory_budget_bytes=int(budget),
                safety_margin=float(safety_margin),
            )
            predicted_context = model.predict(raw_context)
            oracle = plan_materialization_oracle(raw_context, case.observations)
            oracle_outcome = _actual_outcome(
                oracle.action,
                case.observations,
                safe_budget=raw_context.safe_memory_budget_bytes,
            )
            for policy in POLICIES:
                plan = plan_materialization(predicted_context, policy=policy)
                outcome = _actual_outcome(
                    plan.action,
                    case.observations,
                    safe_budget=raw_context.safe_memory_budget_bytes,
                )
                regret: Optional[float] = None
                if outcome["feasible"] and oracle_outcome["feasible"]:
                    latency = outcome["latency_ms"]
                    oracle_latency = oracle_outcome["latency_ms"]
                    if isinstance(latency, (int, float)) and isinstance(
                        oracle_latency, (int, float)
                    ):
                        regret = float(latency) / float(oracle_latency)
                yield {
                    "schema_version": EVAL_SCHEMA_VERSION,
                    "status": "ok",
                    "source_profile_run_id": case.run_id,
                    "case_id": case.case_id,
                    "split": {
                        "kind": "architecture_family_heldout",
                        "calibration_workloads": list(calibration_workloads),
                        "validation_workloads": list(validation_workloads),
                        "model_fit_workloads": list(
                            calibration_workloads + validation_workloads
                        ),
                        "heldout_workloads": list(heldout_workloads),
                        "workload_role": "final_heldout",
                    },
                    "workload": {
                        "name": case.workload,
                        "tier": case.tier,
                        "method": case.method,
                        "spec_size": case.spec_size,
                        "domain_batch_size": case.domain_batch_size,
                    },
                    "budget": {
                        "memory_budget_bytes": int(budget),
                        "safety_margin": float(safety_margin),
                        "safe_memory_budget_bytes": raw_context.safe_memory_budget_bytes,
                    },
                    "decision": MaterializationPlanRecord(
                        context=predicted_context, plan=plan
                    ).to_dict(),
                    "oracle": {
                        "plan": oracle.to_dict(),
                        "actual": oracle_outcome,
                    },
                    "actual": outcome,
                    "metrics": {
                        "feasible": bool(outcome["feasible"]),
                        "oracle_feasible": bool(oracle_outcome["feasible"]),
                        "latency_regret_ratio": regret,
                        "unexpected_failure": bool(
                            oracle_outcome["feasible"] and not outcome["feasible"]
                        ),
                    },
                }


def _parse_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("workload list must be non-empty")
    return names


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(item.strip()) * 1024 * 1024 for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "budgets must be comma-separated MiB integers"
        ) from error
    if not budgets or any(item <= 0 for item in budgets):
        raise argparse.ArgumentTypeError("budgets must be positive")
    return budgets


def _write_jsonl(rows: Iterable[dict[str, object]], handle: TextIO) -> int:
    count = 0
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Fit on calibration families and emit held-out policy evaluation JSONL."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument(
        "--calibration-workloads",
        type=_parse_names,
        default=DEFAULT_CALIBRATION_WORKLOADS,
    )
    parser.add_argument(
        "--validation-workloads",
        type=_parse_names,
        default=DEFAULT_VALIDATION_WORKLOADS,
    )
    parser.add_argument(
        "--heldout-workloads", type=_parse_names, default=DEFAULT_HELDOUT_WORKLOADS
    )
    parser.add_argument(
        "--budgets-mib",
        type=_parse_budgets,
        default=tuple(value * 1024 * 1024 for value in DEFAULT_BUDGETS_MIB),
    )
    parser.add_argument("--safety-margin", type=float, default=0.9)
    parser.add_argument("--ridge", type=float, default=1e-6)
    args = parser.parse_args(argv)

    cases = load_profile_cases(args.input)
    calibration = tuple(args.calibration_workloads)
    validation = tuple(args.validation_workloads)
    heldout = tuple(args.heldout_workloads)
    samples = calibration_samples(cases, set(calibration + validation))
    model = fit_materialization_cost_model(samples, ridge=float(args.ridge))
    rows = evaluate_cases(
        cases,
        model=model,
        calibration_workloads=calibration,
        validation_workloads=validation,
        heldout_workloads=heldout,
        budgets_bytes=tuple(args.budgets_mib),
        safety_margin=float(args.safety_margin),
    )

    output_handle: TextIO
    close_output = False
    if args.output is None:
        output_handle = sys.stdout
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", encoding="utf-8")
        close_output = True
    try:
        count = _write_jsonl(rows, output_handle)
    finally:
        if close_output:
            output_handle.close()

    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": model.schema_version,
            "split": {
                "calibration_workloads": list(calibration),
                "validation_workloads": list(validation),
                "model_fit_workloads": list(calibration + validation),
                "heldout_workloads": list(heldout),
            },
            "model": model.to_dict(),
        }
        args.model_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.manifest_output is not None:
        if args.output is None:
            parser.error("--manifest-output requires --output")
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "boundflow.pr11-planner-eval-manifest/v1",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_commit": _git_value("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "source": {
                "path": str(args.input),
                "sha256": _sha256(args.input),
            },
            "split": {
                "calibration_workloads": list(calibration),
                "validation_workloads": list(validation),
                "model_fit_workloads": list(calibration + validation),
                "heldout_workloads": list(heldout),
            },
            "budgets_bytes": list(args.budgets_mib),
            "safety_margin": float(args.safety_margin),
            "ridge": float(args.ridge),
            "policies": [policy.value for policy in POLICIES],
            "row_count": count,
            "outputs": {
                "jsonl": {
                    "path": str(args.output),
                    "sha256": _sha256(args.output),
                },
                "model": (
                    {
                        "path": str(args.model_output),
                        "sha256": _sha256(args.model_output),
                    }
                    if args.model_output is not None
                    else None
                ),
            },
        }
        args.manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote {count} held-out planner evaluation rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
