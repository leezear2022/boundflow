"""Provider-neutral helpers for the FSG3 real same-solver worker."""

# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring
# pylint: disable=too-many-arguments,duplicate-code

from types import SimpleNamespace
import os

import pytest
import torch

from boundflow.runtime.fsg3_same_solver_timing import FSG3Configuration
from scripts import run_fsg3_same_solver_timing as runner
from scripts import run_rvir_v4_live_return_capture as live_runner


def _gpu_snapshot(
    *,
    sw_thermal: str = "Not Active",
    sw_power: str = "Not Active",
    hw_thermal: str = "Not Active",
    sw_thermal_counter: int = 0,
    sw_power_counter: int = 0,
    hw_thermal_counter: int = 0,
) -> dict[str, object]:
    return {
        "uuid": "GPU-1",
        "name": "RTX 4060",
        "temperature": "44 C",
        "sw_thermal_slowdown": sw_thermal,
        "sw_power_cap": sw_power,
        "hw_thermal_slowdown": hw_thermal,
        "sw_thermal_slowdown_counter_us": sw_thermal_counter,
        "sw_power_cap_counter_us": sw_power_counter,
        "hw_thermal_slowdown_counter_us": hw_thermal_counter,
    }


def test_upper_projection_encodes_only_positive_infinity() -> None:
    shape, values, mask = runner._upper_values(
        torch.tensor([[float("inf"), 2.0]], dtype=torch.float32)
    )
    assert shape == (1, 2)
    assert values == (0.0, 2.0)
    assert mask == (True, False)
    with pytest.raises(ValueError, match="NaN or negative"):
        runner._upper_values(torch.tensor([float("-inf")]))


def test_live_executor_requires_paired_precompiled_inputs() -> None:
    with pytest.raises(ValueError, match="must be paired"):
        live_runner._LiveExecutor(
            model=runner.Path("model.onnx"),
            torch_module=torch,
            arguments_module=SimpleNamespace(),
            precompiled_program=object(),
            capture_payloads=False,
        )
    executor = live_runner._LiveExecutor(
        model=runner.Path("model.onnx"),
        torch_module=torch,
        arguments_module=SimpleNamespace(),
        precompiled_program=object(),
        precompiled_module=object(),
        capture_payloads=False,
    )
    assert executor.capture_payloads is False
    assert executor.last_core_result is None
    assert executor.last_post_result is None
    with pytest.raises(ValueError, match="cache/hash must be paired"):
        live_runner._LiveExecutor(
            model=runner.Path("model.onnx"),
            torch_module=torch,
            arguments_module=SimpleNamespace(),
            prepared_core_cache=object(),
            capture_payloads=False,
        )
    with pytest.raises(ValueError, match="conflicts with precompiled"):
        live_runner._LiveExecutor(
            model=runner.Path("model.onnx"),
            torch_module=torch,
            arguments_module=SimpleNamespace(),
            precompiled_program=object(),
            precompiled_module=object(),
            prepared_core_cache=object(),
            prepared_core_template_hash="f" * 64,
            capture_payloads=False,
        )
    with pytest.raises(ValueError, match="schedule requires prepared core"):
        live_runner._LiveExecutor(
            model=runner.Path("model.onnx"),
            torch_module=torch,
            arguments_module=SimpleNamespace(),
            terminal_optimizer_schedule=object(),
            capture_payloads=False,
        )


def test_profile_recorder_emits_non_overlapping_cuda_span() -> None:
    class FakeEvent:
        def record(self, _stream: object) -> None:
            return None

        def elapsed_time(self, _other: object) -> float:
            return 0.25

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=lambda **_kwargs: FakeEvent(),
            current_stream=object,
        )
    )
    recorder = runner._ProfileRecorder(fake_torch)
    with recorder.span(
        scope="core",
        name="provider_core",
        stack_layer="solver/provider",
        solver_phase="official_update_bounds_core",
        resource="host+cuda",
        cache_state="process-hit",
    ):
        pass
    spans = recorder.finalize()
    assert len(spans) == 1
    assert spans[0].wall_ns > 0
    assert spans[0].gpu_ns == 250_000


def test_protocol_identity_is_common_across_configurations() -> None:
    identities = {
        runner._protocol_identity(configuration) for configuration in FSG3Configuration
    }
    assert len(identities) == 1


def test_query_phase_timing_closes_and_rejects_invalid_nesting() -> None:
    timing = runner._query_phase_timing(
        query_wall_ns=1_000,
        solver_init_ns=100,
        constraint_prepare_ns=25,
        verify_started_ns=10_000,
        verify_ended_ns=10_700,
        core_started_ns=10_200,
        core_ended_ns=10_500,
        final_sync_ns=150,
        update_bounds_post_ns=40,
        official_post_queue_ns=80,
    )
    assert timing == {
        "query_wall_ns": 1_000,
        "solver_init_ns": 100,
        "constraint_prepare_ns": 25,
        "verify_wall_ns": 700,
        "pre_core_ns": 200,
        "core_wall_ns": 300,
        "post_core_ns": 200,
        "final_sync_ns": 150,
        "update_bounds_post_ns": 40,
        "official_post_queue_ns": 80,
        "verify_closure_ns": 0,
        "query_unattributed_ns": 25,
        "post_queue_residual_ns": 40,
    }
    with pytest.raises(ValueError, match="ordering"):
        runner._query_phase_timing(
            query_wall_ns=1_000,
            solver_init_ns=100,
            constraint_prepare_ns=25,
            verify_started_ns=10_000,
            verify_ended_ns=10_700,
            core_started_ns=9_999,
            core_ended_ns=10_500,
            final_sync_ns=150,
            update_bounds_post_ns=40,
            official_post_queue_ns=80,
        )


def test_host_phase_observer_restores_targets_and_requires_one_call() -> None:
    owner = SimpleNamespace(transform=lambda value: value + 1)
    original = owner.transform
    observer = runner._HostPhaseObserver()
    with observer.instrument(((owner, "transform", "transform"),)):
        assert owner.transform(4) == 5
    assert owner.transform is original
    snapshot = observer.snapshot(("transform",))
    assert snapshot["transform"]["call_count"] == 1
    assert snapshot["transform"]["wall_ns"] > 0

    unused = runner._HostPhaseObserver()
    with unused.instrument(((owner, "transform", "transform"),)):
        pass
    with pytest.raises(ValueError, match="count differs"):
        unused.snapshot(("transform",))
    with pytest.raises(ValueError, match="post phase nesting"):
        runner._query_phase_timing(
            query_wall_ns=1_000,
            solver_init_ns=100,
            constraint_prepare_ns=25,
            verify_started_ns=10_000,
            verify_ended_ns=10_700,
            core_started_ns=10_200,
            core_ended_ns=10_500,
            final_sync_ns=150,
            update_bounds_post_ns=81,
            official_post_queue_ns=80,
        )


def test_nested_phase_observer_closes_inclusive_and_exclusive_time() -> None:
    owner = SimpleNamespace()
    owner.inner = lambda value: value + 1
    owner.outer = lambda value: owner.inner(value) * 2
    original_inner = owner.inner
    original_outer = owner.outer
    observer = runner._NestedPhaseObserver()
    with observer.instrument(((owner, "outer", "root"), (owner, "inner", "child"))):
        assert owner.outer(4) == 10
    assert owner.inner is original_inner
    assert owner.outer is original_outer
    snapshot = observer.snapshot(root_name="root", required_names=("root", "child"))
    events = snapshot["events"]
    assert isinstance(events, list)
    assert [row["name"] for row in events] == ["root", "child"]
    assert events[1]["parent_event_id"] == events[0]["event_id"]
    assert events[0]["wall_ns"] >= events[1]["wall_ns"]
    assert events[0]["exclusive_ns"] >= 0
    aggregates = snapshot["aggregates"]
    assert isinstance(aggregates, dict)
    assert aggregates["root"]["call_count"] == 1
    assert aggregates["child"]["call_count"] == 1


def test_worker_post_init_preflight_admission_is_recomputed() -> None:
    payload = {
        "worker_pid": os.getpid(),
        "temperature_limit_celsius": 50,
        "poll_seconds": 5,
        "timeout_seconds": 900,
        "sample_count": 1,
        "wait_ns": 1,
        "samples": [
            {
                "elapsed_ns": 0,
                "temperature_celsius": 50,
                "independent_thermal_active": False,
                "gpu_snapshot": _gpu_snapshot(),
                "compute_processes": [
                    {
                        "pid": os.getpid(),
                        "name": "/python",
                        "used_memory_mib": 200,
                    }
                ],
                "ac_powered": True,
            }
        ],
        "admitted": True,
    }
    runner._validate_worker_preflight(payload)
    payload["samples"][0]["temperature_celsius"] = 51
    with pytest.raises(ValueError, match="admission differs"):
        runner._validate_worker_preflight(payload)
    payload["samples"][0]["temperature_celsius"] = 50
    payload["samples"][0]["independent_thermal_active"] = True
    with pytest.raises(ValueError, match="thermal projection differs"):
        runner._validate_worker_preflight(payload)


def test_gpu_snapshot_thermal_projection_excludes_only_exact_power_alias() -> None:
    coupled = _gpu_snapshot(
        sw_thermal="Active",
        sw_power="Active",
        sw_thermal_counter=12,
        sw_power_counter=12,
    )
    assert runner._snapshot_independent_thermal_active(coupled) is False

    mismatched = dict(coupled, sw_power_cap_counter_us=11)
    assert runner._snapshot_independent_thermal_active(mismatched) is True

    hardware = dict(coupled, hw_thermal_slowdown="Active")
    assert runner._snapshot_independent_thermal_active(hardware) is True


def test_environment_gate_recomputes_coupled_power_thermal_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ac_powered() -> bool:
        return True

    monkeypatch.setattr(runner, "_ac_powered", ac_powered)
    before = _gpu_snapshot(
        sw_thermal="Active",
        sw_power="Active",
        sw_thermal_counter=100,
        sw_power_counter=100,
    )
    after = _gpu_snapshot(
        sw_thermal="Active",
        sw_power="Active",
        sw_thermal_counter=120,
        sw_power_counter=120,
    )
    gate = runner._environment_gate(
        before,
        after,
        (),
        (),
        "runtime",
        worker_pid=123,
    )
    assert gate.software_thermal_signal is True
    assert gate.software_power_cap_signal is True
    assert gate.software_thermal_power_counters_coupled is True
    assert gate.independent_thermal_slowdown is False
    assert gate.admitted is True

    independent_after = dict(after, sw_power_cap_counter_us=119)
    rejected = runner._environment_gate(
        before,
        independent_after,
        (),
        (),
        "runtime",
        worker_pid=123,
    )
    assert rejected.software_thermal_power_counters_coupled is False
    assert rejected.independent_thermal_slowdown is True
    assert rejected.admitted is False

    offset_before = _gpu_snapshot(
        sw_thermal="Not Active",
        sw_power="Not Active",
        sw_thermal_counter=1_100,
        sw_power_counter=1_000,
    )
    offset_after = _gpu_snapshot(
        sw_thermal="Not Active",
        sw_power="Not Active",
        sw_thermal_counter=1_120,
        sw_power_counter=1_020,
    )
    offset_coupled = runner._environment_gate(
        offset_before,
        offset_after,
        (),
        (),
        "runtime",
        worker_pid=123,
    )
    assert offset_coupled.software_thermal_power_counters_coupled is True
    assert offset_coupled.independent_thermal_slowdown is False
    assert offset_coupled.admitted is True

    offset_after["sw_power_cap_counter_us"] = 1_019
    offset_independent = runner._environment_gate(
        offset_before,
        offset_after,
        (),
        (),
        "runtime",
        worker_pid=123,
    )
    assert offset_independent.software_thermal_power_counters_coupled is False
    assert offset_independent.independent_thermal_slowdown is True
    assert offset_independent.admitted is False
