"""Preregistered protocol and decision tests for FSG4/B4-A formal timing."""

# pylint: disable=protected-access,missing-function-docstring,duplicate-code

from argparse import Namespace
from pathlib import Path
import subprocess

import pytest

from scripts import run_fsg4_b4a_formal_timing as timing


def _pairs(*, core: float = 1.03, query: float = 0.98):
    return [
        {
            "ratios": {
                name: (
                    core
                    if name == "core_wall_ns"
                    else query if name == "query_wall_ns" else 1.0
                )
                for name in timing.METRICS
            }
        }
        for _ in range(6)
    ]


def _profiles(*, closure: float = 0.01, residual: float = 0.03):
    return [{"closure_error": closure, "residual_share": residual} for _ in range(12)]


def _preflight(*, temperature: int = 45, software_thermal: str = "Not Active"):
    active = software_thermal == "Active"
    return {
        "temperature_limit_celsius": timing.B4A_PREFLIGHT_TEMPERATURE_LIMIT_C,
        "nvidia_powerd_state": timing.NVIDIA_POWERD_REQUIRED_STATE,
        "gpu_power_limit_watts": timing.GPU_POWER_LIMIT_WATTS,
        "poll_seconds": timing.base_experiment.PREFLIGHT_POLL_SECONDS,
        "timeout_seconds": timing.base_experiment.PREFLIGHT_TIMEOUT_SECONDS,
        "sample_count": 1,
        "wait_ns": 1,
        "samples": [
            {
                "elapsed_ns": 0,
                "temperature_celsius": temperature,
                "independent_thermal_active": False,
                "gpu_snapshot": {
                    "temperature": f"{temperature} C",
                    "sw_thermal_slowdown": software_thermal,
                    "sw_power_cap": "Active" if active else "Not Active",
                    "hw_thermal_slowdown": "Not Active",
                    "sw_thermal_slowdown_counter_us": 42 if active else 0,
                    "sw_power_cap_counter_us": 42 if active else 0,
                    "hw_thermal_slowdown_counter_us": 0,
                },
                "compute_processes": [],
                "ac_powered": True,
            }
        ],
        "admitted": True,
    }


def test_formal_sequence_is_fixed_unique_and_counterbalanced() -> None:
    flattened = [entry for block in timing.SEQUENCE for entry in block]
    assert len(flattened) == 24
    assert len(timing._sequence_payload()) == 6
    assert all(
        flattened.count((configuration, mode)) == 6
        for configuration in ("B3", "B4-A")
        for mode in ("control", "profile")
    )
    assert (
        len(
            {
                timing._run_id(block, position, *entry)
                for block, entries in enumerate(timing.SEQUENCE)
                for position, entry in enumerate(entries)
            }
        )
        == 24
    )


def test_formal_preflight_requires_cool_and_fully_inactive_software_thermal() -> None:
    timing._validate_formal_preflight(_preflight())
    assert timing.FORMAL_PREFLIGHT_CONTRACT["temperature_limit_celsius"] == 45
    assert (
        timing.FORMAL_PREFLIGHT_CONTRACT["software_thermal_signal_must_be_inactive"]
        is True
    )
    with pytest.raises(ValueError, match="strict preflight admission differs"):
        timing._validate_formal_preflight(_preflight(temperature=46))
    with pytest.raises(ValueError, match="strict preflight admission differs"):
        timing._validate_formal_preflight(_preflight(software_thermal="Active"))
    wrong_power = _preflight()
    wrong_power["nvidia_powerd_state"] = "active"
    with pytest.raises(ValueError, match="strict preflight payload differs"):
        timing._validate_formal_preflight(wrong_power)


def test_formal_decision_accepts_exact_preregistered_boundaries() -> None:
    summary = timing._summary(_pairs(), _profiles())
    assert summary["core_wall_geomean_passed"] is True
    assert summary["query_wall_worst_pair_passed"] is True
    assert summary["profile_closed"] is True
    assert summary["performance_candidate_admitted"] is True
    assert summary["status"] == "validated-b4a-performance-candidate"
    assert summary["performance_claimed"] is False


@pytest.mark.parametrize(
    ("pairs", "profiles"),
    (
        (_pairs(core=1.029999), _profiles()),
        (_pairs(query=0.979999), _profiles()),
        (_pairs(), _profiles(closure=0.010001)),
        (_pairs(), _profiles(residual=0.030001)),
    ),
)
def test_formal_decision_fails_closed_below_each_gate(pairs, profiles) -> None:
    summary = timing._summary(pairs, profiles)
    assert summary["performance_candidate_admitted"] is False
    assert summary["status"] == "validated-no-go-b4a-performance"
    assert summary["performance_claimed"] is False


def test_formal_worker_command_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to("/usr/bin/python3")
    args = Namespace(
        abcrown_python=interpreter,
        benchmark_root=tmp_path / "benchmark",
        abcrown_root=tmp_path / "abcrown",
        model=tmp_path / "model.onnx",
        property=tmp_path / "property.vnnlib",
    )
    command = timing._worker_command(
        args=args,
        result=tmp_path / "worker.json",
        block=0,
        position=0,
        configuration="B3",
        mode="control",
    )
    assert command[0] == str(interpreter.absolute())
    assert command[0] != str(interpreter.resolve())


def test_formal_log_sanitization_removes_host_roots(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    abcrown = tmp_path / "abcrown"
    python = abcrown / ".venv" / "bin" / "python"
    value = f"{timing.REPOSITORY_ROOT} {benchmark}/model {abcrown} {python}"
    sanitized = timing._sanitize(
        value,
        benchmark=benchmark,
        abcrown=abcrown,
        python=python,
    )
    assert "$BOUNDFLOW_ROOT" in sanitized
    assert "$VNNCOMP_ROOT/model" in sanitized
    assert "$ABCROWN_ROOT" in sanitized
    assert "$PYTHON/python" in sanitized
    assert str(tmp_path) not in sanitized


def test_formal_worker_timeout_writes_fail_closed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("worker", 1, output="", stderr="")

    monkeypatch.setattr(timing.subprocess, "run", timed_out)
    monkeypatch.setattr(timing.base_experiment, "_host_snapshot", lambda: {})
    with pytest.raises(RuntimeError, match="timed out"):
        timing._run_worker(
            artifact=tmp_path / "artifact",
            index=0,
            command=("python",),
            projection={"worker": "worker.py"},
            preflight={},
            benchmark=tmp_path / "benchmark",
            abcrown=tmp_path / "abcrown",
            python=tmp_path / "abcrown" / "bin" / "python",
        )
    failure = timing._load_json(tmp_path / "artifact" / "failed_worker.json")
    assert failure["timed_out"] is True
    assert failure["returncode"] is None
    assert failure["performance_claimed"] is False
