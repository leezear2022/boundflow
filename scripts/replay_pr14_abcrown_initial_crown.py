#!/usr/bin/env python3
"""Replay one real αβ-CROWN initial plain-CROWN query in BoundFlow."""

# mypy: disable-error-code=import-untyped
# pylint: disable=broad-exception-caught,import-outside-toplevel
# pylint: disable=too-many-arguments,too-many-locals,too-many-statements

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import torch

from boundflow.backends.tvm.fused_crown_cache import FusedCrownModuleCache
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.abcrown_adapter import (
    ABCrownInitialCrownCapture,
    CapturedABCrownQuery,
    file_sha256,
)
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.fused_crown import (
    TVMFusedCrownExecutor,
    build_fused_crown_runtime_selection,
)
from boundflow.runtime.task_executor import InputSpec, PythonTaskExecutor

SCHEMA_VERSION = "boundflow.pr14-initial-crown-replay/v1"
SUPPORTED_BACKENDS = (
    "pytorch_eager",
    "pytorch_chunked",
    "tvm_fused_tir",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vnnlib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-name", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument(
        "--backends",
        default=",".join(SUPPORTED_BACKENDS),
        help="Comma-separated BoundFlow backends",
    )
    return parser.parse_args()


def _require_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    return resolved


def _git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _plain_tensor(value: torch.Tensor) -> torch.Tensor:
    """Strip external Tensor subclasses while preserving storage semantics."""

    if type(value) is torch.Tensor:  # pylint: disable=unidiomatic-typecheck
        return value
    return value.as_subclass(torch.Tensor)


def _result_bounds(result: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(result, (tuple, list)) or not result:
        raise TypeError("compute_bounds replay did not return a tuple/list")
    lower = result[0]
    upper = result[1] if len(result) > 1 else None
    if not torch.is_tensor(lower):
        raise TypeError("compute_bounds replay lower bound is not a tensor")
    if upper is not None and not torch.is_tensor(upper):
        upper = None
    return _plain_tensor(lower), None if upper is None else _plain_tensor(upper)


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("benchmark summary requires samples")
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
    }


def _benchmark_cuda_call(
    call: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    host_samples: list[float] = []
    event_samples: list[float] = []
    result = None
    for _ in range(repeats):
        start_event = (
            torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        )
        end_event = (
            torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        )
        started = time.perf_counter_ns()
        if start_event is not None:
            start_event.record()
        result = call()
        if end_event is not None:
            assert start_event is not None
            end_event.record()
            end_event.synchronize()
            event_samples.append(float(start_event.elapsed_time(end_event)))
        host_samples.append((time.perf_counter_ns() - started) / 1e6)
    return (
        {
            "warmup": warmup,
            "repeats": repeats,
            "host": _summary(host_samples),
            "cuda_event": None if not event_samples else _summary(event_samples),
            "host_samples_ms": host_samples,
            "cuda_event_samples_ms": event_samples,
        },
        result,
    )


def _peak_memory_bytes(call: Callable[[], Any], device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    call()
    torch.cuda.synchronize(device)
    return max(0, int(torch.cuda.max_memory_allocated(device) - baseline))


def _comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual = _plain_tensor(actual)
    expected = _plain_tensor(expected).to(device=actual.device, dtype=actual.dtype)
    if actual.shape != expected.shape:
        return {
            "allclose": False,
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "max_abs_diff": None,
            "atol": atol,
            "rtol": rtol,
        }
    difference = (actual - expected).abs()
    sign_match = (actual >= 0) == (expected >= 0)
    return {
        "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "shape_match": True,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs_diff": float(difference.max().item()),
        "nonnegative_actual": int((actual >= 0).sum().item()),
        "nonnegative_expected": int((expected >= 0).sum().item()),
        "sign_agreement": int(sign_match.sum().item()),
        "sign_total": int(sign_match.numel()),
        "atol": atol,
        "rtol": rtol,
    }


def _move_module(module: Any, device: torch.device, dtype: torch.dtype) -> None:
    params = module.bindings.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("BFTaskModule params binding must be a dictionary")
    module.bindings["params"] = {
        name: (
            value.to(device=device, dtype=dtype).contiguous()
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else (
                value.to(device=device).contiguous()
                if torch.is_tensor(value)
                else value
            )
        )
        for name, value in params.items()
    }


def _build_boundflow_query(
    model: Path,
    captured: CapturedABCrownQuery,
    device: torch.device,
) -> tuple[Any, InputSpec, torch.Tensor]:
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    if len(program.graph.inputs) != 1:
        raise ValueError("PR-14B replay currently requires exactly one model input")
    module = plan_interval_ibp_v0(program)
    lower = _plain_tensor(captured.input_lower).to(device=device)
    upper = _plain_tensor(captured.input_upper).to(device=device)
    linear_spec = _plain_tensor(captured.linear_spec_c).to(device=device)
    _move_module(module, device, lower.dtype)
    spec = InputSpec.box(value_name=program.graph.inputs[0], lower=lower, upper=upper)
    return module, spec, linear_spec


def _nominal_forward_check(
    model: Path,
    module: Any,
    spec: InputSpec,
    *,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Separate ONNX import semantics from bound-relaxation tightness."""

    import onnxruntime as ort

    center = spec.center
    exact_spec = InputSpec.box(
        value_name=spec.value_name,
        lower=center,
        upper=center,
    )
    boundflow_output = PythonTaskExecutor().run_ibp(module, exact_spec).lower
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("nominal ONNX check requires one input and one output")
    onnx_output = torch.from_numpy(
        session.run(
            None,
            {session.get_inputs()[0].name: center.detach().cpu().numpy()},
        )[0]
    ).to(device=center.device, dtype=center.dtype)
    return (
        _comparison(boundflow_output, onnx_output, atol=atol, rtol=rtol),
        boundflow_output,
        onnx_output,
    )


def _boundflow_backend_call(
    module: Any,
    spec: InputSpec,
    linear_spec: torch.Tensor,
    *,
    backend: str,
    chunk_rows: int,
    cache_dir: Path,
) -> tuple[Callable[[], Any], int]:
    selection = build_fused_crown_runtime_selection(
        module.get_entry_task().ops, backend=backend, chunk_rows=chunk_rows
    )
    executor = selection.executor
    if backend == "tvm_fused_tir":
        executor = TVMFusedCrownExecutor(compile_cache=FusedCrownModuleCache(cache_dir))

    def call() -> Any:
        with torch.no_grad():
            return run_crown_ibp_mlp(
                module,
                spec,
                linear_spec_C=linear_spec,
                fused_crown_executor=executor,
                fused_crown_steps=selection.steps if executor is not None else (),
            )

    return call, len(selection.steps)


def _parse_backends(raw: str) -> tuple[str, ...]:
    backends = tuple(item.strip() for item in raw.split(",") if item.strip())
    invalid = sorted(set(backends) - set(SUPPORTED_BACKENDS))
    if not backends or invalid:
        raise ValueError(f"invalid backends: {invalid or raw!r}")
    return tuple(dict.fromkeys(backends))


def main() -> None:  # pylint: disable=too-many-branches
    """Capture, replay, benchmark, and persist one exact initial query."""

    args = _parse_args()
    abcrown_root = args.abcrown_root.expanduser().resolve()
    complete_verifier = abcrown_root / "complete_verifier"
    auto_lirpa = abcrown_root / "auto_LiRPA"
    if not (complete_verifier / "abcrown.py").is_file():
        raise FileNotFoundError(f"invalid αβ-CROWN checkout: {abcrown_root}")
    if not (auto_lirpa / "auto_LiRPA" / "__init__.py").is_file():
        raise FileNotFoundError("αβ-CROWN auto_LiRPA submodule is absent")
    model = _require_file(args.model, "ONNX model")
    vnnlib = _require_file(args.vnnlib, "VNNLIB property")
    config_path = None if args.config is None else _require_file(args.config, "config")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    backends = _parse_backends(args.backends)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    sys.path.insert(0, str(auto_lirpa))
    sys.path.insert(0, str(abcrown_root))
    abcrown = importlib.import_module("abcrown")
    auto_lirpa_module = importlib.import_module("auto_LiRPA")
    config_builder = abcrown.ConfigBuilder
    config = (
        config_builder.from_defaults()
        if config_path is None
        else config_builder.from_yaml(str(config_path))
    )
    overrides = {
        "general/device": args.device,
        "general/complete_verifier": "skip",
        "solver/bound_prop_method": "crown",
        "solver/init_bound_prop_method": "same",
        "attack/pgd_order": "skip",
    }
    for key, value in overrides.items():
        config.set(key, value)

    capture = ABCrownInitialCrownCapture()
    solver = abcrown.ABCrownSolver(str(model), config=config.copy())
    constraints = abcrown.IOConstraints(vnnlib_path=str(vnnlib))
    with capture.instrument(auto_lirpa_module.BoundedModule):
        solver_result = solver.verify(constraints=constraints)
    if capture.captured is None:
        raise RuntimeError("no initial plain-CROWN query was captured")
    captured = capture.captured
    if not captured.bound_lower_requested:
        raise NotImplementedError("PR-14B replay currently requires lower bounds")
    requested_outputs_match = bool(captured.bound_upper_requested)

    external_expected = _plain_tensor(captured.external_lower)

    def external_call() -> Any:
        with torch.no_grad():
            return captured.replay_external()

    external_timing, external_result = _benchmark_cuda_call(
        external_call,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    external_replay_lower, external_replay_upper = _result_bounds(external_result)
    external_row: dict[str, Any] = {
        "timing": external_timing,
        "peak_memory_bytes": _peak_memory_bytes(external_call, device),
        "captured_replay": _comparison(
            external_replay_lower,
            external_expected,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "upper_present": external_replay_upper is not None,
    }

    module, spec, linear_spec = _build_boundflow_query(model, captured, device)
    nominal_check, boundflow_nominal, onnx_nominal = _nominal_forward_check(
        model,
        module,
        spec,
        atol=args.atol,
        rtol=args.rtol,
    )
    backend_rows: dict[str, Any] = {}
    backend_results: dict[str, Any] = {}
    tensor_payload: dict[str, Any] = {
        "input_lower": _plain_tensor(captured.input_lower).detach().cpu(),
        "input_upper": _plain_tensor(captured.input_upper).detach().cpu(),
        "linear_spec_c": _plain_tensor(captured.linear_spec_c).detach().cpu(),
        "external_lower": external_expected.detach().cpu(),
        "external_upper": (
            None
            if captured.external_upper is None
            else _plain_tensor(captured.external_upper).detach().cpu()
        ),
        "boundflow_nominal_output": boundflow_nominal.detach().cpu(),
        "onnx_nominal_output": onnx_nominal.detach().cpu(),
    }
    for backend in backends:
        call, planned_regions = _boundflow_backend_call(
            module,
            spec,
            linear_spec,
            backend=backend,
            chunk_rows=args.chunk_rows,
            cache_dir=output_dir / "tvm-cache",
        )
        result = call()
        lower_comparison = _comparison(
            result.lower,
            external_expected,
            atol=args.atol,
            rtol=args.rtol,
        )
        finite = bool(
            torch.isfinite(result.lower).all() and torch.isfinite(result.upper).all()
        )
        ordered = bool((result.lower <= result.upper).all())
        upper_comparison = (
            None
            if not captured.bound_upper_requested
            else (
                _comparison(
                    result.upper,
                    _plain_tensor(captured.external_upper),
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if captured.external_upper is not None
                else {"allclose": False, "reason": "external_upper_missing"}
            )
        )
        correct = bool(
            lower_comparison["allclose"]
            and finite
            and ordered
            and (upper_comparison is None or bool(upper_comparison["allclose"]))
        )
        timing = None
        peak_memory = None
        if correct and requested_outputs_match:
            timing, result = _benchmark_cuda_call(
                call,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            peak_memory = _peak_memory_bytes(call, device)
        row: dict[str, Any] = {
            "status": "ok" if correct else "bound_equivalence_failure",
            "planned_fused_regions": planned_regions,
            "timing": timing,
            "peak_memory_bytes": peak_memory,
            "benchmark_skipped_reason": (
                None
                if correct and requested_outputs_match
                else (
                    "requested_outputs_mismatch"
                    if correct
                    else "bound_equivalence_gate_failed"
                )
            ),
            "lower_vs_external": lower_comparison,
            "upper_vs_external": upper_comparison,
            "finite": finite,
            "ordered": ordered,
        }
        backend_rows[backend] = row
        backend_results[backend] = result
        tensor_payload[f"boundflow_{backend}_lower"] = result.lower.detach().cpu()
        tensor_payload[f"boundflow_{backend}_upper"] = result.upper.detach().cpu()

    eager_result = backend_results.get("pytorch_eager")
    if eager_result is not None:
        for backend, result in backend_results.items():
            backend_rows[backend]["lower_vs_boundflow_eager"] = _comparison(
                result.lower,
                eager_result.lower,
                atol=args.atol,
                rtol=args.rtol,
            )

    widths = (
        _plain_tensor(captured.input_upper) - _plain_tensor(captured.input_lower)
    ).detach()
    all_correct = (
        bool(external_row["captured_replay"]["allclose"])
        and bool(nominal_check["allclose"])
        and all(row["status"] == "ok" for row in backend_rows.values())
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if all_correct else "bound_equivalence_failure",
        "workload_name": args.workload_name,
        "abcrown_commit": _git_revision(abcrown_root),
        "model": str(model),
        "model_sha256": file_sha256(model),
        "vnnlib": str(vnnlib),
        "vnnlib_sha256": file_sha256(vnnlib),
        "config": None if config_path is None else str(config_path),
        "config_sha256": None if config_path is None else file_sha256(config_path),
        "config_overrides": overrides,
        "capture": {
            "method": captured.method,
            "solver_phase": captured.solver_phase,
            "bound_lower_requested": captured.bound_lower_requested,
            "bound_upper_requested": captured.bound_upper_requested,
            "input_shape": list(captured.input_lower.shape),
            "spec_shape": list(captured.linear_spec_c.shape),
            "box_width_min": float(widths.min().item()),
            "box_width_max": float(widths.max().item()),
            "box_width_unique": int(torch.unique(widths.cpu()).numel()),
        },
        "solver_result": {
            "status": getattr(solver_result, "status", None),
            "success": getattr(solver_result, "success", None),
        },
        "external": external_row,
        "nominal_forward_boundflow_vs_onnx": nominal_check,
        "boundflow": backend_rows,
        "benchmark_contract": {
            "same_process": True,
            "same_input_box": True,
            "same_linear_spec_c": True,
            "same_method": "plain-CROWN",
            "same_requested_outputs": requested_outputs_match,
            "performance_compliant": requested_outputs_match,
            "performance_noncompliance_reason": (
                None
                if requested_outputs_match
                else "external_lower_only_but_boundflow_computes_lower_and_upper"
            ),
            "requires_external_bound_equivalence_before_benchmark": True,
            "compile_excluded_from_warm_samples": True,
            "scope": "initial_bound_only_not_complete_verifier_e2e",
        },
        "command": sys.argv,
    }
    torch.save(tensor_payload, output_dir / "payload.pt")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "workload_name": args.workload_name,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    if not all_correct:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
