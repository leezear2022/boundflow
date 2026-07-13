#!/usr/bin/env python
"""Run PR-12I region-runtime and end-to-end baseline contracts."""

# mypy: disable-error-code=import-untyped
# pylint: disable=broad-exception-caught,duplicate-code,import-outside-toplevel
# pylint: disable=too-many-lines,too-many-locals,too-many-statements

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Optional, Sequence

import torch

from boundflow.benchmarks.contracts import (
    END_TO_END_FINAL_BOUND_CONTRACT,
    REGION_RUNTIME_CONTRACT,
    BenchmarkContract,
    BenchmarkContractLevel,
)
from boundflow.planner.execution_candidate import BackendVariant
from boundflow.runtime.crown_ibp import _relu_backward_mode, run_crown_ibp_mlp
from boundflow.runtime.fused_crown import (
    FusedReluAffineRequest,
    TVMFusedCrownExecutor,
    TVMUnfusedCrownExecutor,
    TorchChunkedFusedCrownExecutor,
    TorchDenseFusedCrownReference,
    build_fused_crown_runtime_selection,
)
from scripts.benchmark_phase7a_pr12_runtime_pareto import (
    RuntimeWorkload,
    _build_query,
    _clear_fused_compile_cache,
    _environment,
    _event_call,
    _max_abs_diff,
    _max_rel_diff,
    _measure_memory,
    _sha256,
    _summary,
    _warm_groups,
    _workload,
    _write_jsonl,
)

SCHEMA_VERSION = "boundflow.pr12i-baseline/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.pr12i-baseline-manifest/v1"
DEFAULT_CASE_IDS = (
    "linear-memory-sensitive",
    "conv-unseen-width",
    "mini-resnet-unseen-width",
)
DEFAULT_BACKENDS = (
    BackendVariant.PYTORCH_EAGER,
    BackendVariant.PYTORCH_STRUCTURED,
    BackendVariant.PYTORCH_CHUNKED,
    BackendVariant.TVM_TIR_UNFUSED,
    BackendVariant.TVM_FUSED_TIR,
    BackendVariant.TORCH_COMPILE,
)


@dataclass(frozen=True)
class _CompiledBoundPair:
    """Adapter keeping the benchmark result interface outside Dynamo capture."""

    lower: torch.Tensor
    upper: torch.Tensor


def _clear_unfused_compile_cache() -> None:
    from boundflow.backends.tvm.unfused_crown import (
        build_unfused_crown_conv2d_module,
        build_unfused_crown_linear_module,
    )

    build_unfused_crown_linear_module.cache_clear()
    build_unfused_crown_conv2d_module.cache_clear()


def _request(workload: RuntimeWorkload, device: torch.device) -> FusedReluAffineRequest:
    """Build one deterministic representative region from a full-query workload."""

    torch.manual_seed(13000 + sum(ord(char) for char in workload.case_id))
    domain, spec = workload.domain, workload.spec
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    attrs: dict[str, object]
    if workload.family == "linear":
        input_shape = (int(workload.config["previous"]),)
        output_shape = (int(workload.config["current"]),)
        weight = torch.randn(*output_shape, *input_shape, device=device)
        bias = torch.randn(output_shape[0], device=device)
        attrs = {}
    elif workload.family == "conv2d":
        channels = int(workload.config["channels"])
        height, width = int(workload.config["height"]), int(workload.config["width"])
        kernel = int(workload.config.get("kernel", 3))
        input_shape = (channels, height, width)
        output_shape = input_shape
        weight = torch.randn(channels, channels, kernel, kernel, device=device)
        bias = torch.randn(channels, device=device)
        attrs = {
            "stride": (1, 1),
            "padding": (kernel // 2, kernel // 2),
            "dilation": (1, 1),
            "groups": 1,
            "output_padding": (0, 0),
        }
    elif workload.family == "mini_resnet":
        # Use the stem Affine->ReLU region; full topology remains in the E2E row.
        output_channels = int(workload.config["width"]) // 2
        input_shape = (3, 16, 16)
        output_shape = (output_channels, 16, 16)
        weight = torch.randn(output_channels, 3, 3, 3, device=device)
        bias = torch.randn(output_channels, device=device)
        attrs = {
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "output_padding": (0, 0),
        }
    else:
        raise ValueError(f"unsupported PR-12I region workload: {workload.family}")
    features = 1
    for extent in output_shape:
        features *= int(extent)
    return FusedReluAffineRequest(
        kind="linear" if workload.family == "linear" else "conv2d",
        A_u=torch.randn(domain, spec, features, device=device),
        A_l=torch.randn(domain, spec, features, device=device),
        alpha_u=torch.rand(domain, features, device=device),
        alpha_l=torch.rand(domain, features, device=device),
        beta_u=torch.randn(domain, features, device=device),
        beta_l=torch.randn(domain, features, device=device),
        weight=weight,
        bias=bias,
        input_shape=input_shape,
        output_shape=output_shape,
        attrs=attrs,
    )


def _region_executor(backend: BackendVariant, *, chunk_rows: int):
    if backend == BackendVariant.PYTORCH_EAGER:
        return TorchDenseFusedCrownReference()
    if backend == BackendVariant.PYTORCH_CHUNKED:
        return TorchChunkedFusedCrownExecutor(chunk_rows=chunk_rows)
    if backend == BackendVariant.TVM_TIR_UNFUSED:
        return TVMUnfusedCrownExecutor()
    if backend == BackendVariant.TVM_FUSED_TIR:
        return TVMFusedCrownExecutor()
    return None


def _fused_result_bytes(result: Any) -> int:
    return sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in (
            result.A_prev_u,
            result.A_prev_l,
            result.bias_delta_u,
            result.bias_delta_l,
        )
    )


def _measure_region_memory(
    call: Callable[[], Any], stream: torch.cuda.Stream, device: torch.device
) -> dict[str, int]:
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    stream.synchronize()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.cuda.stream(stream):
        result = call()
    stream.synchronize()
    output_bytes = _fused_result_bytes(result)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del result
    return {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": max(0, peak_allocated - baseline_allocated),
        "peak_reserved_delta_bytes": max(0, peak_reserved - baseline_reserved),
        "output_bytes": output_bytes,
        "temporary_workspace_upper_bound_bytes": max(
            0, peak_allocated - baseline_allocated - output_bytes
        ),
    }


def _correctness(actual: Any, expected: Any) -> dict[str, Any]:
    actual_tensors = (
        actual.A_prev_u,
        actual.A_prev_l,
        actual.bias_delta_u,
        actual.bias_delta_l,
    )
    expected_tensors = (
        expected.A_prev_u,
        expected.A_prev_l,
        expected.bias_delta_u,
        expected.bias_delta_l,
    )
    max_abs = max(
        float((actual_tensor - expected_tensor).abs().max().item())
        for actual_tensor, expected_tensor in zip(actual_tensors, expected_tensors)
    )
    max_rel = max(
        float(
            (
                (actual_tensor - expected_tensor).abs()
                / expected_tensor.abs().clamp_min(1e-12)
            )
            .max()
            .item()
        )
        for actual_tensor, expected_tensor in zip(actual_tensors, expected_tensors)
    )
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in actual_tensors)
    allclose = all(
        bool(torch.allclose(actual_tensor, expected_tensor, rtol=2e-4, atol=2e-4))
        for actual_tensor, expected_tensor in zip(actual_tensors, expected_tensors)
    )
    return {
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "finite": finite,
        "allclose": allclose,
        "rtol": 2e-4,
        "atol": 2e-4,
    }


def _runtime_fields(
    call: Callable[[], Any],
    stream: torch.cuda.Stream,
    *,
    warmup: int,
    groups: int,
    repeats: int,
) -> tuple[dict[str, Any], Any]:
    first_wall, first_event, first_result = _event_call(call, stream)
    cold_wall, cold_event, cold_result = _event_call(call, stream)
    del cold_result
    host_groups, event_samples = _warm_groups(
        call, stream, warmup=warmup, groups=groups, repeats=repeats
    )
    return (
        {
            "compile_first_run_wall_ms": first_wall,
            "compile_first_run_cuda_event_ms": first_event,
            "cold_wall_ms": cold_wall,
            "cold_cuda_event_ms": cold_event,
            "estimated_compile_overhead_ms": max(0.0, first_wall - cold_wall),
            "host_group_per_query": _summary(host_groups),
            "cuda_event_per_query": _summary(event_samples),
            "warmup": warmup,
            "independent_groups": groups,
            "repeats_per_group": repeats,
            "host_group_samples_ms": host_groups,
            "cuda_event_samples_ms": event_samples,
        },
        first_result,
    )


def _base_row(
    workload: RuntimeWorkload,
    backend: BackendVariant,
    stream_name: str,
    contract: BenchmarkContract,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_contract": contract.to_dict(),
        "workload": {
            "case_id": workload.case_id,
            "family": workload.family,
            "domain": workload.domain,
            "spec": workload.spec,
            "budget_bytes": workload.budget_bytes,
            "config": workload.config,
        },
        "candidate": {"backend": backend.value, "stream": stream_name},
    }


def _not_applicable_row(
    workload: RuntimeWorkload,
    backend: BackendVariant,
    stream_name: str,
    contract: BenchmarkContract,
    reason: str,
) -> dict[str, Any]:
    return {
        **_base_row(workload, backend, stream_name, contract),
        "status": "not_applicable",
        "error": {"error_type": "ContractNotApplicable", "message": reason},
    }


def _region_row(  # pylint: disable=too-many-arguments
    workload: RuntimeWorkload,
    backend: BackendVariant,
    stream_name: str,
    *,
    warmup: int,
    groups: int,
    repeats: int,
    chunk_rows: int,
) -> dict[str, Any]:
    contract = REGION_RUNTIME_CONTRACT
    if backend in {BackendVariant.PYTORCH_STRUCTURED, BackendVariant.TORCH_COMPILE}:
        reason = (
            "torch.compile is conditionally probed only on the complete final-bound path"
            if backend == BackendVariant.TORCH_COMPILE
            else "structured representation crosses dense region boundaries; evaluate at E2E"
        )
        return _not_applicable_row(
            workload,
            backend,
            stream_name,
            contract,
            reason,
        )
    device = torch.device("cuda")
    request = _request(workload, device)
    expected = TorchDenseFusedCrownReference().run(request)
    executor = _region_executor(backend, chunk_rows=chunk_rows)
    if executor is None:
        return _not_applicable_row(
            workload, backend, stream_name, contract, "unsupported region backend"
        )
    if backend == BackendVariant.TVM_FUSED_TIR:
        _clear_fused_compile_cache()
    elif backend == BackendVariant.TVM_TIR_UNFUSED:
        _clear_unfused_compile_cache()
    stream = (
        torch.cuda.default_stream(device)
        if stream_name == "default"
        else torch.cuda.Stream(device=device)
    )

    def call() -> Any:
        return executor.run(request, stream=stream)

    runtime, first_result = _runtime_fields(
        call, stream, warmup=warmup, groups=groups, repeats=repeats
    )
    correctness = _correctness(first_result, expected)
    del first_result
    memory = _measure_region_memory(call, stream, device)
    correct = bool(correctness["finite"] and correctness["allclose"])
    return {
        **_base_row(workload, backend, stream_name, contract),
        "status": "ok" if correct else "fail",
        "error": None if correct else {"error_type": "CorrectnessFailure"},
        "runtime": runtime,
        "memory": memory,
        "correctness": correctness,
    }


def _e2e_row(  # pylint: disable=too-many-arguments
    workload: RuntimeWorkload,
    backend: BackendVariant,
    stream_name: str,
    *,
    warmup: int,
    groups: int,
    repeats: int,
    chunk_rows: int,
) -> dict[str, Any]:
    contract = END_TO_END_FINAL_BOUND_CONTRACT
    device = torch.device("cuda")
    module, input_spec = _build_query(workload, device)
    expected = run_crown_ibp_mlp(module, input_spec)
    persistent_executor = _region_executor(backend, chunk_rows=chunk_rows)
    if backend == BackendVariant.TVM_FUSED_TIR:
        _clear_fused_compile_cache()
    elif backend == BackendVariant.TVM_TIR_UNFUSED:
        _clear_unfused_compile_cache()
    stream = (
        torch.cuda.default_stream(device)
        if stream_name == "default"
        else torch.cuda.Stream(device=device)
    )

    def eager_call() -> Any:
        # Fixed-backend selection is the baseline Planner and region matching is
        # deliberately rebuilt inside every measured complete query.
        context: AbstractContextManager[None]
        if backend == BackendVariant.PYTORCH_STRUCTURED:
            context = _relu_backward_mode("structured")
            selection_backend = BackendVariant.PYTORCH_STRUCTURED.value
        elif backend in {BackendVariant.PYTORCH_EAGER, BackendVariant.TORCH_COMPILE}:
            context = _relu_backward_mode("dense")
            selection_backend = BackendVariant.PYTORCH_EAGER.value
        else:
            context = nullcontext()
            selection_backend = backend.value
        selection = build_fused_crown_runtime_selection(
            module.get_entry_task().ops,
            backend=selection_backend,
            chunk_rows=chunk_rows,
        )
        executor = persistent_executor if selection.executor is not None else None
        with context:
            return run_crown_ibp_mlp(
                module,
                input_spec,
                fused_crown_executor=executor,
                fused_crown_steps=selection.steps,
            )

    call = eager_call
    compile_settings: Optional[dict[str, Any]] = None
    if backend == BackendVariant.TORCH_COMPILE:
        compile_settings = {"fullgraph": True, "dynamic": False, "backend": "inductor"}

        def compile_target() -> tuple[torch.Tensor, torch.Tensor]:
            result = eager_call()
            return result.lower, result.upper

        compiled = torch.compile(compile_target, fullgraph=True, dynamic=False)

        def compiled_call() -> _CompiledBoundPair:
            lower, upper = compiled()
            return _CompiledBoundPair(lower=lower, upper=upper)

        call = compiled_call

    try:
        runtime, first_result = _runtime_fields(
            call, stream, warmup=warmup, groups=groups, repeats=repeats
        )
    except Exception as error:
        if backend != BackendVariant.TORCH_COMPILE:
            raise
        row = _not_applicable_row(
            workload,
            backend,
            stream_name,
            contract,
            "fullgraph torch.compile probe failed on the unmodified final-bound workload",
        )
        row["probe"] = {
            "settings": compile_settings,
            "stage": "capture_or_compile_first_run",
            "error_type": type(error).__name__,
            "message": str(error)[:8000],
        }
        return row
    max_abs = _max_abs_diff(first_result, expected)
    max_rel = _max_rel_diff(first_result, expected)
    finite = bool(
        torch.isfinite(first_result.lower).all()
        and torch.isfinite(first_result.upper).all()
    )
    ordered = bool((first_result.lower <= first_result.upper).all())
    allclose = bool(
        torch.allclose(first_result.lower, expected.lower, rtol=2e-4, atol=2e-4)
        and torch.allclose(first_result.upper, expected.upper, rtol=2e-4, atol=2e-4)
    )
    del first_result
    memory = _measure_memory(call, stream, device)
    correct = finite and ordered and allclose
    return {
        **_base_row(workload, backend, stream_name, contract),
        "status": "ok" if correct else "fail",
        "error": None if correct else {"error_type": "CorrectnessFailure"},
        "candidate": {
            "backend": backend.value,
            "stream": stream_name,
            "planner": "fixed_backend_baseline",
            "compile_settings": compile_settings,
        },
        "runtime": runtime,
        "memory": memory,
        "correctness": {
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "finite": finite,
            "lower_le_upper": ordered,
            "allclose": allclose,
            "rtol": 2e-4,
            "atol": 2e-4,
        },
    }


def _error_row(
    workload: RuntimeWorkload,
    backend: BackendVariant,
    stream_name: str,
    contract: BenchmarkContract,
    error: Exception,
) -> dict[str, Any]:
    return {
        **_base_row(workload, backend, stream_name, contract),
        "status": "error",
        "error": {"error_type": type(error).__name__, "message": str(error)},
    }


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:  # pylint: disable=too-many-branches
    """Run the frozen PR-12I baseline matrix and retain every failure row."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASE_IDS))
    parser.add_argument(
        "--contracts",
        default=",".join(
            (
                BenchmarkContractLevel.REGION_RUNTIME.value,
                BenchmarkContractLevel.END_TO_END_FINAL_BOUND.value,
            )
        ),
    )
    parser.add_argument(
        "--backends",
        default=",".join(backend.value for backend in DEFAULT_BACKENDS),
    )
    parser.add_argument("--streams", default="default,custom")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.chunk_rows, args.warmup, args.groups, args.repeats) <= 0:
        parser.error("chunk-rows/warmup/groups/repeats must be positive")
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    requested_cases = {item for item in args.case_ids.split(",") if item}
    records = [
        record
        for record in split["calibration"]
        if record.get("case_id") in requested_cases
    ]
    missing = requested_cases - {str(record["case_id"]) for record in records}
    if missing:
        parser.error(f"case ids are not consumed calibration cases: {sorted(missing)}")
    workloads = [_workload(record, split_role="baseline") for record in records]
    try:
        backends = tuple(
            BackendVariant(item) for item in args.backends.split(",") if item
        )
        levels = tuple(
            BenchmarkContractLevel(item) for item in args.contracts.split(",") if item
        )
    except ValueError as error:
        parser.error(str(error))
    streams = tuple(item for item in args.streams.split(",") if item)
    if any(stream not in {"default", "custom"} for stream in streams):
        parser.error("streams must be default and/or custom")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for workload in workloads:
        for level in levels:
            contract = (
                REGION_RUNTIME_CONTRACT
                if level == BenchmarkContractLevel.REGION_RUNTIME
                else END_TO_END_FINAL_BOUND_CONTRACT
            )
            for backend in backends:
                for stream_name in streams:
                    try:
                        row = (
                            _region_row(
                                workload,
                                backend,
                                stream_name,
                                warmup=args.warmup,
                                groups=args.groups,
                                repeats=args.repeats,
                                chunk_rows=args.chunk_rows,
                            )
                            if level == BenchmarkContractLevel.REGION_RUNTIME
                            else _e2e_row(
                                workload,
                                backend,
                                stream_name,
                                warmup=args.warmup,
                                groups=args.groups,
                                repeats=args.repeats,
                                chunk_rows=args.chunk_rows,
                            )
                        )
                    except Exception as error:  # pylint: disable=broad-exception-caught
                        row = _error_row(
                            workload, backend, stream_name, contract, error
                        )
                    rows.append(row)
    raw_path = args.out_dir / "raw.jsonl"
    _write_jsonl(raw_path, rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_id": split["split_id"],
        "split_sha256": _sha256(args.split_file),
        "environment": _environment(),
        "contracts": {
            contract.contract_id: {
                "sha256": contract.sha256(),
                "payload": contract.to_dict(),
            }
            for contract in (REGION_RUNTIME_CONTRACT, END_TO_END_FINAL_BOUND_CONTRACT)
            if contract.level in levels
        },
        "case_ids": [workload.case_id for workload in workloads],
        "backends": [backend.value for backend in backends],
        "streams": list(streams),
        "measurement": {
            "warmup": args.warmup,
            "groups": args.groups,
            "repeats": args.repeats,
            "chunk_rows": args.chunk_rows,
        },
        "row_count": len(rows),
        "status_counts": status_counts,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {"raw.jsonl": _sha256(raw_path)},
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
