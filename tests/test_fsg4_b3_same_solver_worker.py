"""Provider-neutral contracts for the FSG4 B3 timing worker."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from argparse import Namespace
from pathlib import Path

import pytest

from boundflow.runtime.fsg3_same_solver_timing import FSG3Mode
from boundflow.runtime.fsg4_b3_explicit_counters import (
    COUNTER_NAMES,
    EXPECTED_B2_FIXED_COUNTERS,
    EXPECTED_B3C_FIXED_COUNTERS,
    Fsg4B3CounterRecorder,
)
from boundflow.runtime.fsg4_b3_same_solver_timing import (
    FSG4B3TimingConfiguration,
)
from scripts import run_fsg4_b3_same_solver_timing as worker


def _counts(configuration: FSG4B3TimingConfiguration) -> dict[str, int]:
    values = {name: 0 for name in COUNTER_NAMES}
    values.update(
        EXPECTED_B2_FIXED_COUNTERS
        if configuration == FSG4B3TimingConfiguration.B2
        else EXPECTED_B3C_FIXED_COUNTERS
    )
    for name in (
        "tensor_content_hash_count",
        "gpu_tensor_content_hash_count",
        "typed_validate_call_count",
        "stable_hash_call_count",
    ):
        values[name] = max(values[name], 1)
    return values


def _recorder(configuration: FSG4B3TimingConfiguration) -> Fsg4B3CounterRecorder:
    return Fsg4B3CounterRecorder(
        retain_events=False,
        _direct_counts=_counts(configuration),
    )


def _diagnostics(configuration: FSG4B3TimingConfiguration) -> dict[str, object]:
    b3 = configuration == FSG4B3TimingConfiguration.B3
    replacement = configuration != FSG4B3TimingConfiguration.B0
    return {
        "prepared_core_template_hashes": (["a" * 64] if b3 else []),
        "prepared_core_instance_hashes": (["b" * 64] if b3 else []),
        "terminal_optimizer_schedule_hashes": (["c" * 64] if b3 else []),
        "assembly_metadata": (
            [{"headline_content_digest_count": 0}] if replacement else []
        ),
        "commit_receipts": ([{"candidate_d2h_copy_count": 0}] if replacement else []),
        "device_commit_audits": ([{"audit_hash": "d" * 64}] if b3 else []),
        "post_query_audit_ns": (1 if b3 else 0),
        "post_query_audit_excluded_from_timing": True,
    }


def test_protocol_binds_five_fresh_admission() -> None:
    identity = worker._protocol_identity()
    assert len(identity) == 64
    assert worker.FIVE_FRESH_MANIFEST_FILE_SHA256.startswith("bf8b3ecc")
    assert worker.FIVE_FRESH_MANIFEST_HASH.startswith("457ab1ad")


def test_b3_underlying_measurement_is_b2_but_output_is_not_relabelled_silently() -> (
    None
):
    args = Namespace(
        configuration="B3",
        mode="control",
        run_id="run",
        block_index=0,
        sequence_position=0,
        benchmark_root=Path("bench"),
        abcrown_root=Path("abcrown"),
        model=Path("model.onnx"),
        property=Path("property.vnnlib"),
    )
    namespace = worker._base_namespace(args, Path("result.json"))
    assert namespace.configuration == "B2"
    assert FSG4B3TimingConfiguration(args.configuration).value == "B3"


@pytest.mark.parametrize(
    "configuration",
    [FSG4B3TimingConfiguration.B2, FSG4B3TimingConfiguration.B3],
)
def test_profile_activation_requires_direct_receipts_and_counters(
    configuration: FSG4B3TimingConfiguration,
) -> None:
    activation = worker._activation_receipt(
        configuration=configuration,
        mode=FSG3Mode.PROFILE,
        diagnostics=_diagnostics(configuration),
        recorder=_recorder(configuration),
    )
    activation.validate(configuration, FSG3Mode.PROFILE)
    assert activation.detailed_counts is not None


def test_b3_control_requires_device_receipt_without_counter_instrumentation() -> None:
    activation = worker._activation_receipt(
        configuration=FSG4B3TimingConfiguration.B3,
        mode=FSG3Mode.CONTROL,
        diagnostics=_diagnostics(FSG4B3TimingConfiguration.B3),
        recorder=None,
    )
    assert activation.candidate_d2h_copy_count == 0
    assert activation.detailed_counts is None


def test_missing_terminal_schedule_fails_closed() -> None:
    diagnostics = _diagnostics(FSG4B3TimingConfiguration.B3)
    diagnostics["terminal_optimizer_schedule_hashes"] = []
    with pytest.raises(ValueError, match="activation receipt"):
        worker._activation_receipt(
            configuration=FSG4B3TimingConfiguration.B3,
            mode=FSG3Mode.CONTROL,
            diagnostics=diagnostics,
            recorder=None,
        )


def test_lightweight_counter_recorder_keeps_no_event_journal() -> None:
    recorder = Fsg4B3CounterRecorder(retain_events=False)
    recorder.add("optimizer_evaluation_count", amount=10, detail="optimizer")
    assert not recorder.events
    assert recorder.counts()["optimizer_evaluation_count"] == 10


def test_diagnostics_normalization_removes_host_executable_paths() -> None:
    diagnostics = {
        "runtime_environment": {"python_executable": "/home/user/env/bin/python"},
        "compute_processes_before": [
            {"pid": 1, "name": "/home/user/env/bin/python", "used_memory_mib": 1}
        ],
        "compute_processes_after": [],
        "worker_preflight": {
            "samples": [
                {
                    "compute_processes": [
                        {
                            "pid": 1,
                            "name": "/home/user/env/bin/python",
                            "used_memory_mib": 1,
                        }
                    ]
                }
            ]
        },
    }
    normalized = worker._normalize_diagnostics(diagnostics)
    assert normalized["runtime_environment"]["python_executable"] == "python"
    assert normalized["compute_processes_before"][0]["name"] == "python"
    assert (
        normalized["worker_preflight"]["samples"][0]["compute_processes"][0]["name"]
        == "python"
    )
