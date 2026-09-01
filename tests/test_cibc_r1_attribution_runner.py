"""Artifact and re-signed tamper tests for the CIBC R1-A smoke runner."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, cast

import pytest

from boundflow.ir.task import BoundTask, TaskKind, TaskOp
from boundflow.runtime.cibc_r1_attribution import (
    CalibrationTriplet,
    canonical_hash,
    derive_clock_calibration,
    topology_from_task,
)
from boundflow.runtime.cibc_r1_nsys import (
    R1NsightExportReceipt,
    nsys_export_receipt_from_dict,
)
from scripts import run_cibc_r1_attribution_artifact as artifact
from scripts.run_cibc_r1_attribution_worker import WORKER_SCHEMA


def _topology() -> dict[str, object]:
    task = BoundTask(
        task_id="r1a-runner-test",
        kind=TaskKind.INTERVAL_IBP,
        ops=[TaskOp("relu", "relu0", ["input"], ["output"])],
        input_values=["input"],
        output_values=["output"],
    )
    return topology_from_task(
        task,
        external_values=("input",),
        value_shapes={"input": (1, 2), "output": (1, 2)},
        value_dtypes={"input": "torch.float32", "output": "torch.float32"},
        value_devices={"input": "cuda:0", "output": "cuda:0"},
        single_stream=True,
    ).to_dict()


def _calibration() -> dict[str, object]:
    rows = []
    for phase, base in (("before", 1_000_000_000), ("after", 2_000_000_000)):
        for ordinal in range(64):
            gpu = base + ordinal * 100_000
            rows.append(
                CalibrationTriplet(
                    phase=phase,
                    ordinal=ordinal,
                    host_before_ns=gpu + 99_500,
                    gpu_timestamp_ns=gpu,
                    host_after_ns=gpu + 100_500,
                )
            )
    return derive_clock_calibration(rows).to_dict()


def _worker(mode: str, *, profile_ratio: float = 1.02) -> dict[str, object]:
    topology = _topology()
    topology_hash = str(topology["topology_hash"])
    marker = f"boundflow.r1/graph/0/relu/{topology_hash[:12]}"
    group_count = 20
    median = profile_ratio if mode == "profile" else 1.0
    calibration = _calibration() if mode == "profile" else None
    value: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "mode": mode,
        "pair_ordinal": 0,
        "topology": topology,
        "marker_receipt": {
            "markers": [marker],
            "invocations": {"0": 4},
            "expected_invocations_per_marker": 4,
            "capture_only": True,
        },
        "semantic_receipt": {
            "maximum_absolute_difference": artifact.EXPECTED_SEMANTIC_MAXIMUM,
            "sign_exact": True,
            "element_count": artifact.EXPECTED_SEMANTIC_ELEMENT_COUNT,
            "atol": 3.0e-4,
            "rtol": 3.0e-4,
            "baseline_launch_count": 0,
            "candidate_launch_count": 6,
            "fallback_count": 0,
            "eager_shadow_count": 0,
        },
        "groups_ms": [median] * group_count,
        "median_ms": median,
        "warmup_count": 10,
        "group_count": group_count,
        "repeats_per_group": 5 if mode == "profile" else 50,
        "input_copy_included": True,
        "threads_per_block": 128,
        "calibration_receipt": calibration,
        "profile_inventory": (
            {
                "backend": "torch_profiler_smoke",
                "device_event_counts": {"DeviceType.CPU": 1},
                "event_name_counts": {"graph": 1},
                "event_count": 1,
                "owner_ledger_available": False,
                "formal_attribution_available": False,
                "reason": "nsys_export_unavailable",
            }
            if mode == "profile"
            else None
        ),
        "profiler_epoch_warmup_excluded": mode == "profile",
        "environment": {"device": "test"},
        "cupti_admitted": bool(calibration and calibration["cupti_admitted"]),
        "formal_attribution_admitted": False,
        "performance_claimed": False,
    }
    value["worker_hash"] = canonical_hash(value)
    return value


def _protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    source = tmp_path / "source.pt"
    model = tmp_path / "model.onnx"
    source.write_bytes(b"source")
    model.write_bytes(b"model")
    monkeypatch.setattr(artifact, "SOURCE_CAPTURE_SHA256", artifact.file_sha256(source))
    monkeypatch.setattr(artifact, "MODEL_SHA256", artifact.file_sha256(model))
    monkeypatch.setattr(
        artifact, "PRODUCTION_TOPOLOGY_HASH", str(_topology()["topology_hash"])
    )
    return artifact.protocol(source, model, smoke=True)


def test_worker_and_smoke_summary_recompute_from_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol(tmp_path, monkeypatch)
    control = _worker("control")
    profile = _worker("profile")
    artifact.validate_worker(control, mode="control", pair_ordinal=0)
    artifact.validate_worker(profile, mode="profile", pair_ordinal=0)
    summary = artifact.derive_summary(
        protocol, {(0, "control"): control, (0, "profile"): profile}
    )
    assert summary["pair_count"] == 1
    assert summary["all_cupti_admitted"] is True
    assert summary["all_perturbation_admitted"] is True
    assert summary["formal_attribution_admitted"] is False
    assert summary["status"] == "smoke-only-r1a-attribution-closed"


def test_protocol_rejects_resigned_scope_target_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol(tmp_path, monkeypatch)
    target = cast(dict[str, Any], protocol["target_contract"])
    target["query_research"] = 1.10
    protocol["protocol_hash"] = canonical_hash(
        {key: value for key, value in protocol.items() if key != "protocol_hash"}
    )
    with pytest.raises(ValueError, match="target contract differs"):
        artifact.validate_protocol(protocol)


def test_profile_perturbation_outside_gate_is_disclosed_not_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = artifact.derive_summary(
        _protocol(tmp_path, monkeypatch),
        {
            (0, "control"): _worker("control"),
            (0, "profile"): _worker("profile", profile_ratio=1.17),
        },
    )
    pair = summary["pairs"][0]
    assert pair["profile_perturbation"] == pytest.approx(1.17)
    assert pair["perturbation_admitted"] is False
    assert summary["formal_attribution_admitted"] is False


def test_worker_rejects_outer_resigned_latency_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact, "PRODUCTION_TOPOLOGY_HASH", str(_topology()["topology_hash"])
    )
    worker = _worker("control")
    worker["median_ms"] = 2.0
    worker["worker_hash"] = canonical_hash(
        {key: value for key, value in worker.items() if key != "worker_hash"}
    )
    with pytest.raises(ValueError, match="identity differs"):
        artifact.validate_worker(worker, mode="control", pair_ordinal=0)


def test_worker_rejects_outer_resigned_calibration_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact, "PRODUCTION_TOPOLOGY_HASH", str(_topology()["topology_hash"])
    )
    worker = _worker("profile")
    calibration = cast(dict[str, Any], worker["calibration_receipt"])
    calibration["slope"] = 1.5
    worker["worker_hash"] = canonical_hash(
        {key: value for key, value in worker.items() if key != "worker_hash"}
    )
    with pytest.raises(ValueError, match="calibration derivation"):
        artifact.validate_worker(worker, mode="profile", pair_ordinal=0)


def test_formal_preflight_requires_nsys_before_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    model = tmp_path / "model.onnx"
    source.write_bytes(b"source")
    model.write_bytes(b"model")
    monkeypatch.setattr(artifact.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="requires Nsight Systems"):
        artifact.generate(
            tmp_path / "formal",
            source_capture=source,
            model=model,
            smoke=False,
        )


def test_porcelain_parser_preserves_dot_prefixed_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=("git",),
        returncode=0,
        stdout=" M .docops/ev.jsonl\nM  tracked.py\n",
        stderr="",
    )
    monkeypatch.setattr(artifact.subprocess, "run", lambda *_args, **_kwargs: completed)
    assert artifact._tracked_dirty_paths() == (".docops/ev.jsonl", "tracked.py")


def test_nsys_export_receipt_round_trip_and_anchor_gate() -> None:
    receipt = R1NsightExportReceipt(
        anchor_errors_ns=(221, 445, 224),
        graph_node_count=42,
        cloned_graph_node_count=138,
        profile_group_count=20,
        replay_count=100,
        kernel_count=4_200,
        memcpy_count=200,
        runtime_api_count=520,
        graph_launch_count=100,
        stream_ids=(7,),
        unowned_event_count=0,
        temporal_fallback_count=0,
        graph_node_owner_hash="a" * 64,
        formal_admitted=True,
    )
    assert nsys_export_receipt_from_dict(receipt.to_dict()) == receipt
    payload = receipt.to_dict()
    payload["formal_admitted"] = "true"
    with pytest.raises(ValueError, match="admission differs"):
        nsys_export_receipt_from_dict(payload)
    rejected = R1NsightExportReceipt(**{**receipt.__dict__, "formal_admitted": False})
    assert nsys_export_receipt_from_dict(rejected.to_dict()) == rejected


def test_smoke_artifact_replay_rebuilds_summary_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    protocol = _protocol(tmp_path, monkeypatch)
    control = _worker("control")
    profile = _worker("profile")
    summary = artifact.derive_summary(
        protocol, {(0, "control"): control, (0, "profile"): profile}
    )
    artifact._write_json(root / "protocol.json", protocol)
    artifact._write_json(root / "raw/pair_00_control.json", control)
    artifact._write_json(root / "raw/pair_00_profile.json", profile)
    artifact._write_json(root / "summary.json", summary)
    artifact._write_json(
        root / "manifest.json", artifact._manifest(root, protocol, summary)
    )
    assert artifact.replay(root) == summary


def test_generate_creates_exact_missing_parent_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol(tmp_path, monkeypatch)
    source = tmp_path / "source.pt"
    model = tmp_path / "model.onnx"
    control = _worker("control")
    profile = _worker("profile")
    monkeypatch.setattr(artifact, "protocol", lambda _source, _model, smoke: protocol)
    monkeypatch.setattr(
        artifact,
        "_run_worker",
        lambda **kwargs: control if kwargs["mode"] == "control" else profile,
    )
    root = tmp_path / "missing" / "nested" / "artifact"
    summary = artifact.generate(root, source_capture=source, model=model, smoke=True)
    assert root.is_dir()
    assert artifact.replay(root) == summary


def test_generate_preserves_partial_raw_on_worker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol(tmp_path, monkeypatch)
    source = tmp_path / "source.pt"
    model = tmp_path / "model.onnx"
    monkeypatch.setattr(artifact, "protocol", lambda _source, _model, smoke: protocol)
    monkeypatch.setattr(
        artifact,
        "_run_worker",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("fresh worker failed")),
    )
    root = tmp_path / "failed" / "artifact"
    with pytest.raises(RuntimeError, match="fresh worker failed"):
        artifact.generate(root, source_capture=source, model=model, smoke=True)
    assert not root.exists()
    failure = artifact._load_json(root.with_name("artifact.failed") / "failure.json")
    assert failure["error"] == "fresh worker failed"
    assert failure["performance_claimed"] is False
