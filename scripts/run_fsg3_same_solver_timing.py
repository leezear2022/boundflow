#!/usr/bin/env python3
"""Run one real FSG3 B0/B1/B2 same-solver timing control worker."""

# pylint: disable=wrong-import-position,protected-access,import-outside-toplevel
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-instance-attributes,import-error
# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, MutableMapping, Optional, Sequence, cast
import xml.etree.ElementTree as ET

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import (
    canonical_hash,
    FSG3Configuration,
    FSG3EnvironmentGate,
    FSG3ExecutionCounters,
    FSG3Mode,
    FSG3SemanticResult,
    FSG3TimingMetrics,
    FSG3TimingRun,
)
from scripts import run_rvir_v4_live_return_capture as candidate_runner
from scripts import run_rvir_v4_production_state_capture as capture_runner

WORKER_ENVELOPE_SCHEMA = "boundflow.fsg3-same-solver-worker-envelope/v1"
ALLOWED_GRAPHICS_PROCESSES = ("kwin_wayland",)


def _one_final_bound(value: object, *, label: str) -> tuple[str, Any]:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise ValueError(f"FSG3 {label} inventory differs")
    name, tensor = next(iter(value.items()))
    if not isinstance(name, str):
        raise TypeError(f"FSG3 {label} name differs")
    return name, tensor


class _QueueObserver:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.events: list[dict[str, object]] = []

    @contextmanager
    def instrument(self, domain_list_class: Any) -> Iterator[None]:
        original = domain_list_class.add

        def wrapped(
            instance: Any,
            ret: MutableMapping[str, object],
            host: Mapping[str, object],
            *positional: object,
            **keyword: object,
        ) -> object:
            before = len(instance)
            lower_name, lower = _one_final_bound(ret.get("lower_bounds"), label="lower")
            upper_name, upper = _one_final_bound(ret.get("upper_bounds"), label="upper")
            depths = host.get("depths")
            thresholds = host.get("thresholds")
            history = host.get("history")
            target = (
                lower_name == upper_name
                and self.torch.is_tensor(lower)
                and self.torch.is_tensor(upper)
                and self.torch.is_tensor(depths)
                and self.torch.is_tensor(thresholds)
                and isinstance(history, list)
                and tuple(cast(Any, depths).shape) == (6,)
                and tuple(cast(Any, thresholds).shape) == (6, 1)
                and int(lower.shape[0]) == 6
            )
            result = original(instance, ret, host, *positional, **keyword)
            if target:
                after = len(instance)
                self.events.append(
                    {
                        "before": before,
                        "input": int(lower.shape[0]),
                        "accepted": after - before,
                        "after": after,
                        "lower": lower,
                        "upper": upper,
                        "depths": depths,
                        "history": history,
                        "thresholds": thresholds,
                    }
                )
            return result

        domain_list_class.add = wrapped
        try:
            yield
        finally:
            domain_list_class.add = original


class _ProviderObserver:
    def __init__(self, core_observer: "_CoreObserver") -> None:
        self.core_observer = core_observer
        self.compute_bounds_call_count = 0

    @contextmanager
    def instrument(self, bounded_module_class: Any) -> Iterator[None]:
        original = bounded_module_class.compute_bounds

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if self.core_observer.active:
                self.compute_bounds_call_count += 1
            return original(instance, *args, **kwargs)

        bounded_module_class.compute_bounds = wrapped
        try:
            yield
        finally:
            bounded_module_class.compute_bounds = original


class _CoreObserver:  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        *,
        configuration: FSG3Configuration,
        torch_module: Any,
        arguments_module: Any,
    ) -> None:
        self.configuration = configuration
        self.torch = torch_module
        self.arguments = arguments_module
        self.core_count = 0
        self.provider_update_bounds_call_count = 0
        self.typed_validation_count = 0
        self.typed_snapshot_hash: Optional[str] = None
        self.active = False
        self.last_result: Any = None
        self.host_start_ns: Optional[int] = None
        self.host_end_ns: Optional[int] = None
        self.start_event: Any = None
        self.end_event: Any = None

    @contextmanager
    def instrument(self, stage_solve_module: Any) -> Iterator[None]:
        original = stage_solve_module.update_bounds_core

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.core_count != 0:
                raise RuntimeError("FSG3 requires exactly one update_bounds_core")
            net = kwargs.get("net", args[0] if args else None)
            pre_result = kwargs.get("pre_result", args[1] if len(args) > 1 else None)
            if net is None or pre_result is None:
                raise TypeError("FSG3 core call schema differs")
            stream = self.torch.cuda.current_stream()
            self.start_event = self.torch.cuda.Event(enable_timing=True)
            self.end_event = self.torch.cuda.Event(enable_timing=True)
            self.start_event.record(stream)
            self.host_start_ns = time.perf_counter_ns()
            self.active = True
            heuristic: Any = None
            original_compute: Any = None
            try:
                if self.configuration == FSG3Configuration.B1:
                    policy = capture_runner._optimizer_policy(self.arguments, kwargs)
                    snapshot = capture_runner._build_core_pre_snapshot(
                        pre_result,
                        net=net,
                        core_id=0,
                        policy=policy,
                    )
                    self.typed_snapshot_hash = snapshot.stable_hash()
                    self.typed_validation_count += 1
                if self.configuration in {
                    FSG3Configuration.B0,
                    FSG3Configuration.B1,
                }:
                    heuristic = kwargs.get("branching_heuristic")
                    original_compute = getattr(
                        heuristic, "compute_branching_decisions", None
                    )
                    if not callable(original_compute):
                        raise TypeError("FSG3 provider branching is unavailable")

                    def counted_compute(*inner_args: Any, **inner_kwargs: Any) -> Any:
                        previous_profile = sys.getprofile()

                        def count_update(frame: Any, event: str, value: object) -> None:
                            if previous_profile is not None:
                                cast(Any, previous_profile)(frame, event, value)
                            if (
                                event == "call"
                                and frame.f_code.co_name == "update_bounds"
                                and frame.f_code.co_filename.replace(
                                    "\\", "/"
                                ).endswith("/beta_CROWN_solver.py")
                            ):
                                self.provider_update_bounds_call_count += 1

                        sys.setprofile(count_update)
                        try:
                            return original_compute(*inner_args, **inner_kwargs)
                        finally:
                            sys.setprofile(previous_profile)

                    heuristic.compute_branching_decisions = counted_compute
                result = original(*args, **kwargs)
            finally:
                self.host_end_ns = time.perf_counter_ns()
                self.end_event.record(stream)
                self.active = False
                if heuristic is not None:
                    heuristic.compute_branching_decisions = original_compute
            self.last_result = result
            self.core_count += 1
            return result

        stage_solve_module.update_bounds_core = wrapped
        try:
            yield
        finally:
            stage_solve_module.update_bounds_core = original

    def timings(self) -> tuple[int, int]:
        if (
            self.core_count != 1
            or self.host_start_ns is None
            or self.host_end_ns is None
            or self.start_event is None
            or self.end_event is None
        ):
            raise ValueError("FSG3 core timing is incomplete")
        wall = self.host_end_ns - self.host_start_ns
        gpu = int(round(self.start_event.elapsed_time(self.end_event) * 1e6))
        if wall <= 0 or gpu <= 0:
            raise ValueError("FSG3 core timing is non-positive")
        return wall, gpu


class _PostObserver:
    def __init__(self) -> None:
        self.last_result: Any = None
        self.count = 0

    @contextmanager
    def instrument(self, stage_postprocess_module: Any) -> Iterator[None]:
        original = stage_postprocess_module.update_bounds_post

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.last_result = result
            self.count += 1
            return result

        stage_postprocess_module.update_bounds_post = wrapped
        try:
            yield
        finally:
            stage_postprocess_module.update_bounds_post = original


def _visited_domains(result: Any) -> tuple[int, ...]:
    stats = getattr(result, "stats", None)
    if not isinstance(stats, dict) or not isinstance(stats.get("bab"), list):
        return ()
    return tuple(
        int(row[2])
        for row in stats["bab"]
        if isinstance(row, (tuple, list)) and len(row) >= 3
    )


def _tensor_values(value: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point() and not bool(tensor.isfinite().all()):
        raise ValueError("FSG3 finite tensor projection differs")
    return tuple(int(item) for item in tensor.shape), tuple(
        float(item) for item in tensor.reshape(-1).tolist()
    )


def _upper_values(
    value: Any,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[bool, ...]]:
    tensor = value.detach().cpu().contiguous()
    if bool(tensor.isnan().any()) or bool(tensor.isneginf().any()):
        raise ValueError("FSG3 upper contains NaN or negative infinity")
    mask = tensor.isposinf()
    if not bool((tensor.isfinite() | mask).all()):
        raise ValueError("FSG3 upper sentinel differs")
    encoded = tensor.masked_fill(mask, 0.0)
    return (
        tuple(int(item) for item in tensor.shape),
        tuple(float(item) for item in encoded.reshape(-1).tolist()),
        tuple(bool(item) for item in mask.reshape(-1).tolist()),
    )


def _sequence_ints(value: object, torch_module: Any) -> tuple[int, ...]:
    if torch_module.is_tensor(value):
        tensor = cast(Any, value)
        return tuple(int(item) for item in tensor.detach().cpu().reshape(-1).tolist())
    if isinstance(value, Sequence):
        return tuple(int(item) for item in value)
    raise TypeError("FSG3 integer sequence differs")


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"FSG3 {label} must be an integer")
    return value


def _semantic_result(
    *,
    solver_result: Any,
    core_result: Any,
    queue_event: Mapping[str, object],
    torch_module: Any,
) -> FSG3SemanticResult:
    lower_shape, lower_values = _tensor_values(queue_event["lower"])
    upper_shape, upper_values, upper_mask = _upper_values(queue_event["upper"])
    decision = core_result.branching_decision
    input_count = _integer(queue_event["input"], "queue input")
    accepted = _integer(queue_event["accepted"], "queue accepted")
    result = FSG3SemanticResult(
        status=str(solver_result.status),
        success=bool(solver_result.success),
        visited_domains=_visited_domains(solver_result),
        queue_before=_integer(queue_event["before"], "queue before"),
        queue_input=input_count,
        queue_accepted=accepted,
        queue_pruned=input_count - accepted,
        queue_after=_integer(queue_event["after"], "queue after"),
        depths=_sequence_ints(queue_event["depths"], torch_module),
        history_count=len(cast(list[object], queue_event["history"])),
        lower_shape=lower_shape,
        lower_values=lower_values,
        upper_shape=upper_shape,
        upper_values=upper_values,
        upper_positive_infinity_mask=upper_mask,
        final_decision=tuple(
            (int(layer), int(index)) for layer, index in decision.branching_decision
        ),
        split_depth=int(decision.split_depth),
        batch_size=int(decision.batch_size),
        n_verified=int(core_result.n_verified),
        n_splits=int(core_result.n_splits),
    )
    result.validate()
    return result


def _run_command(*command: str) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def _nvidia_snapshot() -> dict[str, object]:
    xml = ET.fromstring(_run_command("nvidia-smi", "-q", "-x"))
    gpu = xml.find("gpu")
    if gpu is None:
        raise ValueError("FSG3 NVIDIA GPU XML is unavailable")

    def text_at(path: str) -> str:
        value = gpu.findtext(path)
        if value is None:
            raise ValueError(f"FSG3 NVIDIA field is unavailable: {path}")
        return value

    counters = text_at(
        "clocks_event_reasons_counters/clocks_event_reasons_counters_sw_therm_slowdown"
    )
    counter_us = int(counters.split()[0])
    return {
        "uuid": text_at("uuid"),
        "name": text_at("product_name"),
        "performance_state": text_at("performance_state"),
        "temperature": text_at("temperature/gpu_temp"),
        "power_draw": text_at("gpu_power_readings/instant_power_draw"),
        "sm_clock": text_at("clocks/sm_clock"),
        "memory_clock": text_at("clocks/mem_clock"),
        "sw_thermal_slowdown": text_at(
            "clocks_event_reasons/clocks_event_reason_sw_thermal_slowdown"
        ),
        "hw_thermal_slowdown": text_at(
            "clocks_event_reasons/clocks_event_reason_hw_thermal_slowdown"
        ),
        "sw_thermal_slowdown_counter_us": counter_us,
    }


def _compute_processes() -> list[dict[str, object]]:
    output = _run_command(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) != 3:
            raise ValueError("FSG3 NVIDIA process row differs")
        rows.append(
            {
                "pid": int(fields[0]),
                "name": fields[1],
                "used_memory_mib": int(fields[2]),
            }
        )
    return rows


def _ac_powered() -> bool:
    path = Path("/sys/class/power_supply/ACAD/online")
    return path.is_file() and path.read_text(encoding="utf-8").strip() == "1"


def _environment_gate(
    before: Mapping[str, object],
    after: Mapping[str, object],
    before_processes: Sequence[Mapping[str, object]],
    after_processes: Sequence[Mapping[str, object]],
) -> FSG3EnvironmentGate:
    external: list[str] = []
    by_identity = {
        (
            _integer(row["pid"], "process PID"),
            str(row["name"]),
            _integer(row["used_memory_mib"], "process memory"),
        )
        for row in (*before_processes, *after_processes)
        if _integer(row["pid"], "process PID") != os.getpid()
    }
    for pid, name, memory in sorted(by_identity):
        basename = Path(name).name
        identity = f"{pid}:{name}:{memory}MiB"
        if basename in ALLOWED_GRAPHICS_PROCESSES and memory < 64:
            continue
        external.append(identity)
    thermal_active = any(
        value != "Not Active"
        for value in (
            before["sw_thermal_slowdown"],
            before["hw_thermal_slowdown"],
            after["sw_thermal_slowdown"],
            after["hw_thermal_slowdown"],
        )
    )
    counter_increased = _integer(
        after["sw_thermal_slowdown_counter_us"], "thermal counter"
    ) > _integer(before["sw_thermal_slowdown_counter_us"], "thermal counter")
    return FSG3EnvironmentGate(
        gpu_uuid=str(before["uuid"]),
        gpu_name=str(before["name"]),
        external_compute_processes=tuple(external),
        thermal_slowdown=thermal_active or counter_increased,
        worker_overlap=bool(external),
        device_identity_stable=(
            before["uuid"] == after["uuid"] and before["name"] == after["name"]
        ),
        ac_powered=_ac_powered(),
    )


def _protocol_identity(_configuration: FSG3Configuration) -> str:
    return canonical_hash(
        {
            "device": "cuda",
            "seed": 100,
            "timeout_seconds": 60,
            "max_iterations": 1,
            "batch_size": 64,
            "auto_enlarge_batch_size": False,
            "alpha_steps": 5,
            "beta_steps": 10,
            "attack": "skip",
            "property_cache": "cold_isolated_copy",
        }
    )


def _source_identity(args: argparse.Namespace) -> str:
    return canonical_hash(
        {
            "boundflow_commit": capture_runner._git_value(
                REPOSITORY_ROOT, "rev-parse", "HEAD"
            ),
            "abcrown_commit": capture_runner._git_value(
                args.abcrown_root, "rev-parse", "HEAD"
            ),
            "auto_lirpa_commit": capture_runner._git_value(
                args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD"
            ),
            "vnncomp_commit": capture_runner._git_value(
                args.benchmark_root, "rev-parse", "HEAD"
            ),
            "model_sha256": capture_runner.file_sha256(args.model),
            "property_sha256": capture_runner.file_sha256(args.property),
        }
    )


def _worker(args: argparse.Namespace) -> None:  # pylint: disable=too-many-locals
    if args.mode != FSG3Mode.CONTROL.value:
        raise ValueError("FSG3 profile worker is not implemented in FSG3-2")
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import (  # type: ignore[import-not-found]
        ABCrownSolver,
        ConfigBuilder,
        IOConstraints,
    )
    import arguments  # type: ignore[import-not-found]
    from activation_split import stage_postprocess, stage_solve  # type: ignore[import-not-found]
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped]
    from branching_domains import BatchedDomainList  # type: ignore[import-not-found]

    capture_runner._validate_inputs(
        args.benchmark_root, args.abcrown_root, Path(sys.executable)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("FSG3 worker requires CUDA")
    configuration = FSG3Configuration(args.configuration)
    mode = FSG3Mode(args.mode)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    environment_before = _nvidia_snapshot()
    processes_before = _compute_processes()
    cold_started_ns = time.perf_counter_ns()
    program: Any = None
    module: Any = None
    compile_ns = 0
    if configuration == FSG3Configuration.B2:
        compile_started_ns = time.perf_counter_ns()
        from boundflow.frontends.onnx.frontend import import_onnx
        from boundflow.planner import plan_interval_ibp_v0

        program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
        module = plan_interval_ibp_v0(program)
        compile_ns = time.perf_counter_ns() - compile_started_ns

    core = _CoreObserver(
        configuration=configuration,
        torch_module=torch,
        arguments_module=arguments,
    )
    provider = _ProviderObserver(core)
    post = _PostObserver()
    queue = _QueueObserver(torch)
    executor: Any = None
    if configuration == FSG3Configuration.B2:
        executor = candidate_runner._LiveExecutor(
            model=args.model,
            torch_module=torch,
            arguments_module=arguments,
            precompiled_program=program,
            precompiled_module=module,
            capture_payloads=False,
        )

    with tempfile.TemporaryDirectory(prefix="boundflow-fsg3-property-") as raw:
        isolated_property = Path(raw) / args.property.name
        shutil.copy2(args.property, isolated_property)
        config = (
            ConfigBuilder.from_defaults()
            .set("general/device", "cuda")
            .set("general/seed", 100)
            .set("general/reset_seed_after_precompile", True)
            .set("general/complete_verifier", "bab")
            .set("attack/pgd_order", "skip")
            .set("bab/timeout", 60)
            .set("bab/max_iterations", 1)
            .set("solver/batch_size", 64)
            .set("solver/auto_enlarge_batch_size", False)
            .set("solver/alpha-crown/iteration", 5)
            .set("solver/beta-crown/iteration", 10)
        )
        with ExitStack() as stack:
            if executor is not None:
                stack.enter_context(
                    executor.instrument(
                        stage_solve=stage_solve,
                        stage_postprocess=stage_postprocess,
                    )
                )
            stack.enter_context(provider.instrument(BoundedModule))
            stack.enter_context(core.instrument(stage_solve))
            stack.enter_context(post.instrument(stage_postprocess))
            stack.enter_context(queue.instrument(BatchedDomainList))
            query_start_event = torch.cuda.Event(enable_timing=True)
            query_end_event = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream()
            query_start_event.record(stream)
            query_started_ns = time.perf_counter_ns()
            solver = ABCrownSolver(str(args.model), config=config)
            solver_result = solver.verify(
                constraints=IOConstraints(vnnlib_path=str(isolated_property))
            )
            query_end_event.record(stream)
            torch.cuda.synchronize()
            query_wall_ns = time.perf_counter_ns() - query_started_ns
            query_gpu_ns = int(
                round(query_start_event.elapsed_time(query_end_event) * 1e6)
            )
    cold_outer_ns = time.perf_counter_ns() - cold_started_ns
    core_wall_ns, core_gpu_ns = core.timings()
    if len(queue.events) != 1 or post.count != 1:
        raise ValueError("FSG3 post/queue event count differs")
    post_validation_started_ns = time.perf_counter_ns()
    semantic = _semantic_result(
        solver_result=solver_result,
        core_result=core.last_result,
        queue_event=queue.events[0],
        torch_module=torch,
    )
    post_validation_ns = time.perf_counter_ns() - post_validation_started_ns
    environment_after = _nvidia_snapshot()
    processes_after = _compute_processes()
    environment = _environment_gate(
        environment_before, environment_after, processes_before, processes_after
    )
    if configuration == FSG3Configuration.B2:
        if executor is None or executor.core_count != 1:
            raise ValueError("FSG3 B2 executor count differs")
        typed_count = 1
        provider_core = 0
        provider_update = executor.provider_update_bounds_callback_count
        fallback = executor.fallback_dispatch_count
        backend = "torch-eager-reference"
        replacement = "whole_call"
    else:
        typed_count = core.typed_validation_count
        provider_core = core.core_count
        provider_update = core.provider_update_bounds_call_count
        fallback = 0
        backend = "auto-lirpa"
        replacement = (
            "original_provider"
            if configuration == FSG3Configuration.B0
            else "rvir_passthrough"
        )
    metrics = FSG3TimingMetrics(
        cold_total_ns=(
            compile_ns + query_wall_ns
            if configuration == FSG3Configuration.B2
            else query_wall_ns
        ),
        boundflow_compile_ns=compile_ns,
        query_wall_ns=query_wall_ns,
        query_gpu_ns=query_gpu_ns,
        core_wall_ns=core_wall_ns,
        core_gpu_ns=core_gpu_ns,
        post_validation_ns=post_validation_ns,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )
    execution = FSG3ExecutionCounters(
        typed_validation_count=typed_count,
        provider_core_call_count=provider_core,
        provider_compute_bounds_call_count=provider.compute_bounds_call_count,
        provider_update_bounds_call_count=provider_update,
        fallback_dispatch_count=fallback,
        backend_kind=backend,
        replacement_mode=replacement,
    )
    run = FSG3TimingRun(
        run_id=args.run_id,
        block_index=args.block_index,
        sequence_position=args.sequence_position,
        configuration=configuration,
        mode=mode,
        source_identity=_source_identity(args),
        protocol_identity=_protocol_identity(configuration),
        metrics=metrics,
        semantics=semantic,
        execution=execution,
        environment=environment,
        profile_closure_error=None,
        profile_residual_share=None,
    )
    run.validate()
    envelope = {
        "schema_version": WORKER_ENVELOPE_SCHEMA,
        "run": run.to_dict(),
        "diagnostics": {
            "typed_snapshot_hash": core.typed_snapshot_hash,
            "pre_state_identities": (
                [] if executor is None else executor.pre_state_identities
            ),
            "assembly_metadata": (
                [] if executor is None else executor.assembly_metadata
            ),
            "commit_receipts": ([] if executor is None else executor.commit_receipts),
            "environment_before": environment_before,
            "environment_after": environment_after,
            "compute_processes_before": processes_before,
            "compute_processes_after": processes_after,
            "allowed_graphics_processes": list(ALLOWED_GRAPHICS_PROCESSES),
            "cold_outer_ns": cold_outer_ns,
            "cold_total_is_compile_plus_query_composite": True,
            "cold_scope_includes_hook_setup": False,
            "post_validation_excluded_from_timing": True,
        },
        "performance_claimed": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "configuration": configuration.value,
                "query_wall_ns": query_wall_ns,
                "core_wall_ns": core_wall_ns,
                "environment_admitted": environment.admitted,
                "performance_claimed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("B0", "B1", "B2"), required=True)
    parser.add_argument("--mode", choices=("control", "profile"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--sequence-position", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one fail-closed real control worker."""

    args = _parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    args.abcrown_root = args.abcrown_root.resolve()
    args.model = args.model.resolve()
    args.property = args.property.resolve()
    args.result = args.result.resolve()
    _worker(args)


if __name__ == "__main__":
    main()
