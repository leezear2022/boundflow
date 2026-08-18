"""Contracts for B4-A same-solver worker and five-fresh replay."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from copy import deepcopy
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from boundflow.runtime.fsg3_same_solver_timing import FSG3Mode
from boundflow.runtime.fsg4_b3_explicit_counters import COUNTER_NAMES
from scripts import run_fsg4_b4a_correctness_pairs as pairs
from scripts import run_fsg4_b4a_same_solver_worker as worker
from scripts.run_rvir_v4_live_return_capture import _audited_tensor_payload


def _export_payload(offset: float = 0.0):
    value = torch.tensor([[1.0 + offset, -2.0 + offset]], dtype=torch.float32)
    return {
        "lower": _audited_tensor_payload(value),
        "lAs": {"/activation": _audited_tensor_payload(value.reshape(1, 1, 2))},
        "intermediates": {
            "/preactivation": {
                "lower": _audited_tensor_payload(value),
                "upper": _audited_tensor_payload(value + 3.0),
            }
        },
    }


def test_b4a_export_pair_compares_raw_float_payloads() -> None:
    result = pairs._export_pair(_export_payload(), _export_payload(1e-6))
    assert result["tensor_count"] == 4
    assert 0.0 < result["maximum_absolute_difference"] < 2e-6
    assert result["all_sign_exact"] is True

    with pytest.raises(ValueError, match="numeric differs"):
        pairs._export_pair(_export_payload(), _export_payload(1.0))


def test_b4a_pair_schedule_is_fixed_and_counterbalanced() -> None:
    assert pairs.PAIR_SCHEDULE == (
        ("B3", "B4-A"),
        ("B4-A", "B3"),
        ("B3", "B4-A"),
        ("B4-A", "B3"),
        ("B3", "B4-A"),
    )
    assert len(pairs.CODE_PATHS) == len(set(pairs.CODE_PATHS))


def test_b4a_worker_command_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
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
    command = pairs._command(
        args,
        pair_index=0,
        position=0,
        configuration="B3",
        result=tmp_path / "worker.json",
    )
    assert command[0] == str(interpreter.absolute())
    assert command[0] != str(interpreter.resolve())


def test_b4a_worker_logs_replace_host_local_roots(tmp_path: Path) -> None:
    args = Namespace(
        abcrown_python=tmp_path / "abcrown" / ".venv" / "bin" / "python",
        abcrown_root=tmp_path / "abcrown",
        benchmark_root=tmp_path / "benchmark",
    )
    value = (
        f"python={args.abcrown_python.absolute()} "
        f"model={args.benchmark_root.resolve()}/model.onnx "
        f"repo={pairs.REPOSITORY_ROOT}/script.py"
    )
    sanitized = pairs._sanitize_log(value, args)
    assert "$ABCROWN_PYTHON" in sanitized
    assert "$BENCHMARK_ROOT/model.onnx" in sanitized
    assert "$BOUNDFLOW_ROOT/script.py" in sanitized
    assert str(tmp_path) not in sanitized


def test_b4a_worker_activation_rejects_lineage_and_rerun_tamper() -> None:
    names = ("17", "19", "23", "25", "28", "31")
    handoff_payload = {
        "runtime_handoff_hash": "a" * 64,
        "handoff_hash": "b" * 64,
        "terminal_lower_adjoint_handoff_count": 1,
        "provider_core_callback_count": 0,
        "provider_compute_bounds_callback_count": 0,
        "provider_update_bounds_callback_count": 0,
        "fallback_dispatch_count": 0,
        "lower_adjoints": {name: {} for name in names},
        "lineages": {
            name: {
                "shape_source": "correlation-parent-boundflow-operator",
                "kernel_shape_inferred": False,
                "producer_op_ordinal": ordinal,
                "producer_op_name": f"op-{ordinal}",
                "lineage_hash": str(ordinal) * 64,
            }
            for ordinal, name in enumerate(names, start=1)
        },
    }
    diagnostics = {
        "native_backward_export_metadata": [
            {
                "schema_version": "boundflow.rvir-v4-native-backward-export/v1",
                "export_hash": "c" * 64,
            }
        ],
        "terminal_lower_adjoint_handoff_metadata": [
            {
                "optimizer_evaluation_count": 10,
                "optimizer_update_count": 9,
                "terminal_lower_adjoint_handoff_count": 1,
                "terminal_export_crown_rerun_count": 0,
                "runtime_handoff_hash": "a" * 64,
                "handoff": handoff_payload,
            }
        ],
        "terminal_export_assembly_metadata": [
            {
                "terminal_lower_adjoint_handoff_count": 1,
                "terminal_export_crown_rerun_count": 0,
                "provider_core_callback_count": 0,
                "provider_compute_bounds_callback_count": 0,
                "provider_update_bounds_callback_count": 0,
                "fallback_dispatch_count": 0,
                "handoff_hash": "a" * 64,
                "export_schema_version": "boundflow.rvir-v4-native-backward-export/v1",
                "assembly_hash": "d" * 64,
            }
        ],
    }
    activation = worker._activation("B4-A", diagnostics, actual_profile_counts=None)
    assert activation["terminal_lower_adjoint_handoff_count"] == 1
    assert activation["terminal_export_crown_rerun_count"] == 0
    assert activation["lineage_count"] == 6

    tampered = deepcopy(diagnostics)
    tampered["terminal_lower_adjoint_handoff_metadata"][0]["handoff"]["lineages"]["17"][
        "kernel_shape_inferred"
    ] = True
    with pytest.raises(ValueError, match="lineage receipt differs"):
        worker._activation("B4-A", tampered, actual_profile_counts=None)

    tampered = deepcopy(diagnostics)
    tampered["terminal_export_assembly_metadata"][0][
        "terminal_export_crown_rerun_count"
    ] = 1
    with pytest.raises(ValueError, match="activation receipt differs"):
        worker._activation("B4-A", tampered, actual_profile_counts=None)


def test_b4a_profile_counter_contract_preserves_physical_forward_four() -> None:
    counts = {name: 1 for name in COUNTER_NAMES}
    counts.update(worker.EXPECTED_B3C_FIXED_COUNTERS)
    payload = {"activation": {"detailed_counts": counts}}
    assert worker._actual_profile_counts(payload, FSG3Mode.PROFILE) == counts
    assert counts["forward_trace_build_count"] == 4
    with pytest.raises(ValueError, match="control counter receipt differs"):
        worker._actual_profile_counts(payload, FSG3Mode.CONTROL)
