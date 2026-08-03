#!/usr/bin/env python
"""Profile measured dense/structured combinations across ReLU barriers."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Iterable, Optional, Sequence

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boundflow.frontends.pytorch.frontend import (
    import_torch,
)  # pylint: disable=wrong-import-position
from boundflow.planner import (
    plan_interval_ibp_v0,
)  # pylint: disable=wrong-import-position
from boundflow.planner.materialization import (
    MaterializationAction,
)  # pylint: disable=wrong-import-position
from boundflow.planner.materialization_placement import (  # pylint: disable=wrong-import-position
    BarrierPlacement,
    MaterializationPlacementPlan,
    PLACEMENT_SCHEMA_VERSION,
    PlacementPolicy,
)
from boundflow.planner.materialization_static_features import (  # pylint: disable=wrong-import-position
    StaticBarrierSummary,
    summarize_static_barriers,
)
from boundflow.runtime.crown_ibp import (  # pylint: disable=wrong-import-position
    _forward_ibp_trace_mlp,
    run_crown_ibp_mlp,
)
from boundflow.runtime.materialization import (
    trace_materializations,
)  # pylint: disable=wrong-import-position
from boundflow.runtime.task_executor import (
    InputSpec,
)  # pylint: disable=wrong-import-position
from scripts.profile_phase7a_pr10_materialization import (  # pylint: disable=wrong-import-position
    BasicBlock,
    WORKLOADS,
    Workload,
    _make_workload,
    _measure_trace_off,
)

PROFILE_SCHEMA_VERSION = "boundflow.pr11-barrier-placement-profile/v3"
PLACEMENT_WORKLOADS = (*WORKLOADS, "mini_resnet2", "branched_resnet")


class MiniResNet2(nn.Module):
    """Two-block calibration network; three-block MiniResNet remains held out."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.block1 = BasicBlock(8)
        self.block2 = BasicBlock(8)
        self.head = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate two residual blocks."""

        x = torch.relu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x.flatten(1))


class BranchedResNet(nn.Module):
    """Held-out parallel residual topology with add, concat, and seven barriers."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.left = BasicBlock(8)
        self.right = BasicBlock(8)
        self.fuse = nn.Conv2d(16, 8, 1)
        self.head = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate parallel residual branches and their fused representation."""

        stem = torch.relu(self.stem(x))
        left = self.left(stem)
        right = self.right(stem)
        merged = torch.relu(left + right)
        fused = torch.relu(self.fuse(torch.cat((merged, stem), dim=1)))
        return self.head(fused.flatten(1))


def _make_placement_workload(name: str, device: torch.device) -> Workload:
    if name not in {"mini_resnet2", "branched_resnet"}:
        return _make_workload(name, device)
    torch.manual_seed(0)
    builder = MiniResNet2 if name == "mini_resnet2" else BranchedResNet
    model = builder().eval().to(device)
    model.requires_grad_(False)
    return Workload(
        name=name,
        model=model,
        input_shape=(3, 32, 32),
        tier=(
            "non_toy_calibration_structure"
            if name == "mini_resnet2"
            else "non_toy_heldout_topology"
        ),
    )


def _parse_csv_list(value: str, *, allowed: Iterable[str]) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid or not values:
        raise argparse.ArgumentTypeError(f"unsupported or empty values: {invalid}")
    return values


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _placement_plan(
    barrier_ids: tuple[str, ...], actions: tuple[MaterializationAction, ...]
) -> MaterializationPlacementPlan:
    placements = tuple(
        BarrierPlacement(
            barrier_id=barrier_id,
            action=action,
            persistent_bytes=0,
            ephemeral_bytes=0,
            latency_ms=0.0,
            reason="measured_exhaustive_candidate",
        )
        for barrier_id, action in zip(barrier_ids, actions)
    )
    return MaterializationPlacementPlan(
        schema_version=PLACEMENT_SCHEMA_VERSION,
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
        placements=placements,
        predicted_peak_bytes=0,
        predicted_latency_ms=0.0,
        safe_memory_budget_bytes=1 << 62,
        requires_replan=False,
        recommended_domain_batch_size=1,
        reason="measured_exhaustive_candidate",
    )


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs - rhs).abs().max().detach().cpu().item())


def _profile_combo(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    workload,
    module,
    spec_size: int,
    domain_batch: int,
    device: torch.device,
    barrier_ids: tuple[str, ...],
    static_barriers: tuple[StaticBarrierSummary, ...],
    actions: tuple[MaterializationAction, ...],
    dense_reference,
    spec: InputSpec,
    linear_spec: torch.Tensor,
    run_id: str,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    combo = "".join(
        "D" if action == MaterializationAction.DENSE else "S" for action in actions
    )
    query_id = f"{workload.name}:CROWN:s{spec_size}:d{domain_batch}:p{combo}"
    plan = _placement_plan(barrier_ids, actions)
    base: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "status": "ok",
        "error": None,
        "run_id": run_id,
        "query_id": query_id,
        "workload": {"name": workload.name, "tier": workload.tier},
        "method": "CROWN",
        "spec_size": int(spec_size),
        "domain_batch_size": int(domain_batch),
        "domain_source": "synthetic_fixed_domain_batch",
        "device": str(device),
        "barrier_ids": list(barrier_ids),
        "static_barriers": [summary.to_dict() for summary in static_barriers],
        "placement": plan.to_dict(),
    }

    def invoke():
        return run_crown_ibp_mlp(
            module,
            spec,
            linear_spec_C=linear_spec,
            materialization_placement_plan=plan,
        )

    try:
        timing = _measure_trace_off(
            lambda: (invoke(), {}, {}),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        with trace_materializations(
            run_id=run_id,
            query_id=query_id,
            bound_method="CROWN",
            solver_phase="backward",
            spec_batch=spec_size,
            domain_batch=domain_batch,
            capture_cuda_memory=device.type == "cuda",
        ) as trace:
            result = invoke()
        lower_diff = _max_abs_diff(result.lower, dense_reference.lower)
        upper_diff = _max_abs_diff(result.upper, dense_reference.upper)
        correctness = {
            "finite": bool(
                torch.isfinite(result.lower).all()
                and torch.isfinite(result.upper).all()
            ),
            "lower_le_upper": bool((result.lower <= result.upper + 1e-6).all()),
            "dense_max_abs_diff_lower": lower_diff,
            "dense_max_abs_diff_upper": upper_diff,
            "allclose_dense": bool(
                torch.allclose(
                    result.lower, dense_reference.lower, atol=1e-5, rtol=1e-5
                )
                and torch.allclose(
                    result.upper, dense_reference.upper, atol=1e-5, rtol=1e-5
                )
            ),
        }
        base.update(
            {
                "timing_trace_off": timing,
                "trace_on": trace.to_record(),
                "correctness": correctness,
            }
        )
        if not correctness["allclose_dense"]:
            base["status"] = "fail"
            base["error"] = {
                "type": "CorrectnessGate",
                "message": str(base["correctness"]),
            }
    except torch.cuda.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        base["status"] = "oom"
        base["error"] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:  # pylint: disable=broad-exception-caught
        rendered = traceback.format_exc()
        base["status"] = "fail"
        base["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
    return base


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _write_outputs(
    out_dir: Path, rows: list[dict[str, object]], argv: list[str]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "boundflow.pr11-barrier-placement-manifest/v3",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "argv": argv,
        "row_count": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({str(row["status"]) for row in rows})
        },
        "outputs": {"raw.jsonl": hashlib.sha256(raw_path.read_bytes()).hexdigest()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# pylint: disable-next=too-many-locals,too-many-statements
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Enumerate selected placement combinations and write raw evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-root", default="artifacts/phase7a-pr11")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workloads", default="mini_resnet")
    parser.add_argument("--spec-sizes", default="9")
    parser.add_argument("--domain-batches", default="1")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-combinations", type=int, default=0)
    parser.add_argument(
        "--combination-order", choices=("lexicographic", "shuffled"), default="shuffled"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    workloads = _parse_csv_list(args.workloads, allowed=PLACEMENT_WORKLOADS)
    spec_sizes = _parse_int_list(args.spec_sizes)
    domain_batches = _parse_int_list(args.domain_batches)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.warmup < 0 or args.repeats <= 0 or args.max_combinations < 0:
        parser.error(
            "warmup/max-combinations must be non-negative; repeats must be positive"
        )

    device = torch.device(device_name)
    torch.set_num_threads(1)
    run_id = args.run_id or f"pr11-barriers-{int(time.time())}-{os.getpid()}"
    rows: list[dict[str, object]] = []
    for workload_name in workloads:
        workload = _make_placement_workload(workload_name, device)
        dummy = torch.zeros((1, *workload.input_shape), device=device)
        program = import_torch(
            workload.model, (dummy,), export_mode="export", normalize=True
        )
        module = plan_interval_ibp_v0(program)
        for spec_size in spec_sizes:
            for domain_batch in domain_batches:
                torch.manual_seed(1)
                center = torch.randn(
                    (domain_batch, *workload.input_shape), device=device
                )
                spec = InputSpec.linf(
                    value_name=module.get_entry_task().input_values[0],
                    center=center,
                    eps=0.03,
                )
                linear_spec = torch.randn((domain_batch, spec_size, 10), device=device)
                _interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
                barrier_ids = tuple(relu_pre)
                element_sizes = {
                    int(state.lower.element_size()) for state in relu_pre.values()
                }
                if len(element_sizes) != 1:
                    raise ValueError("all ReLU barriers must use one element size")
                static_barriers = summarize_static_barriers(
                    module.get_entry_task(),
                    {
                        name: tuple(int(dimension) for dimension in state.lower.shape)
                        for name, state in relu_pre.items()
                    },
                    spec_size=spec_size,
                    domain_batch_size=domain_batch,
                    element_size_bytes=next(iter(element_sizes)),
                )
                dense_actions = tuple(MaterializationAction.DENSE for _ in barrier_ids)
                dense_reference = run_crown_ibp_mlp(
                    module,
                    spec,
                    linear_spec_C=linear_spec,
                    materialization_placement_plan=_placement_plan(
                        barrier_ids, dense_actions
                    ),
                )
                combinations = list(
                    itertools.product(
                        (MaterializationAction.DENSE, MaterializationAction.STRUCTURED),
                        repeat=len(barrier_ids),
                    )
                )
                if args.combination_order == "shuffled":
                    random.Random(int(args.seed)).shuffle(combinations)
                for index, actions in enumerate(combinations):
                    if args.max_combinations and index >= args.max_combinations:
                        break
                    rows.append(
                        _profile_combo(
                            workload=workload,
                            module=module,
                            spec_size=spec_size,
                            domain_batch=domain_batch,
                            device=device,
                            barrier_ids=barrier_ids,
                            static_barriers=static_barriers,
                            actions=tuple(actions),
                            dense_reference=dense_reference,
                            spec=spec,
                            linear_spec=linear_spec,
                            run_id=run_id,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
                    )
    out_dir = Path(args.out_root) / run_id
    effective_argv = list(sys.argv if argv is None else [Path(sys.argv[0]).name, *argv])
    _write_outputs(out_dir, rows, effective_argv)
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
