#!/usr/bin/env python3
"""Run synchronized-rehash tamper probes against an RVIR-v4 step artifact."""

# pylint: disable=import-outside-toplevel,protected-access,wrong-import-position
# pylint: disable=too-many-locals
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_rvir_v4_optimizer_step_artifact as artifact_runner

REPORT_SCHEMA_VERSION = "boundflow.rvir-v4-optimizer-step-tamper-report/v1"
Mutator = Callable[[dict[str, Any]], None]


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root differs: {path}")
    return value


def _load_capture(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("RVIR-v4 optimizer tamper capture root differs")
    return value


def _trace(capture: Mapping[str, Any]) -> dict[str, Any]:
    traces = capture.get("optimizer_step_traces")
    if (
        not isinstance(traces, list)
        or len(traces) != 1
        or not isinstance(traces[0], dict)
    ):
        raise ValueError("RVIR-v4 optimizer tamper trace inventory differs")
    return cast(dict[str, Any], traces[0])


def _step_metadata(step: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {key: value for key, value in step.items() if key != "lower"}
    tensors = metadata.get("state_tensors")
    if not isinstance(tensors, list):
        raise TypeError("RVIR-v4 optimizer tamper state tensor inventory differs")
    metadata["state_tensors"] = [
        {key: value for key, value in tensor.items() if key != "value"}
        for tensor in tensors
    ]
    return metadata


def _resign_step_state(step: dict[str, Any]) -> None:
    tensors = step.get("state_tensors")
    if not isinstance(tensors, list):
        raise TypeError("RVIR-v4 optimizer tamper state tensor inventory differs")
    step["state_hash"] = _canonical_hash(
        [
            {key: value for key, value in tensor.items() if key != "value"}
            for tensor in tensors
        ]
    )


def _resign_trace(trace: dict[str, Any]) -> None:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise TypeError("RVIR-v4 optimizer tamper step inventory differs")
    semantic = {key: value for key, value in trace.items() if key != "trace_hash"}
    semantic["steps"] = [_step_metadata(step) for step in steps]
    trace["trace_hash"] = _canonical_hash(semantic)


def _mutate_state(capture: dict[str, Any]) -> None:
    trace = _trace(capture)
    step = trace["steps"][3]
    tensor = step["state_tensors"][0]
    value = tensor["value"].clone()
    value.reshape(-1)[0] += 0.125
    tensor["value"] = value
    tensor["content_sha256"] = production_tensor_sha256(value)
    _resign_step_state(step)
    _resign_trace(trace)


def _mutate_lower(capture: dict[str, Any]) -> None:
    trace = _trace(capture)
    step = trace["steps"][3]
    lower = step["lower"].clone()
    lower.reshape(-1)[0] += 1.0
    step["lower"] = lower
    step["lower_sha256"] = production_tensor_sha256(lower)
    _resign_trace(trace)


def _mutate_call_result(capture: dict[str, Any]) -> None:
    trace = _trace(capture)
    target_call = trace["steps"][3]["call_id"]
    calls = capture.get("calls")
    if not isinstance(calls, list):
        raise TypeError("RVIR-v4 optimizer tamper call inventory differs")
    call = next(row for row in calls if row.get("call_id") == target_call)
    result = next(
        row for row in call["result_tensors"] if row.get("path") == "result[0]"
    )
    result["content_sha256"] = "0" * 64


def _mutate_step_lineage(capture: dict[str, Any]) -> None:
    trace = _trace(capture)
    trace["steps"][-1]["call_id"] = 99
    _resign_trace(trace)


def _mutate_policy(capture: dict[str, Any]) -> None:
    trace = _trace(capture)
    policy = trace["mutation_policy"]
    policy["controls"]["lr_decay"] = 0.97
    trace["mutation_policy_hash"] = _canonical_hash(policy)
    _resign_trace(trace)


PROBES: tuple[tuple[str, str, Mutator], ...] = (
    ("state-internal-rehash", "trace/call state binding differs", _mutate_state),
    ("lower-internal-rehash", "trace/call lower binding differs", _mutate_lower),
    ("call-result-resign", "trace/call lower binding differs", _mutate_call_result),
    (
        "step-lineage-internal-rehash",
        "trace/call lineage differs",
        _mutate_step_lineage,
    ),
    (
        "policy-internal-rehash",
        "production mutation policy is not admitted",
        _mutate_policy,
    ),
)


def _resign_artifact(artifact: Path, capture: Mapping[str, Any]) -> str:
    import torch

    capture_path = artifact / artifact_runner.WORKER_CAPTURE_FILE
    torch.save(dict(capture), capture_path)
    manifest_path = artifact / "manifest.json"
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("RVIR-v4 optimizer tamper manifest files differ")
    files[artifact_runner.WORKER_CAPTURE_FILE] = _file_sha256(capture_path)
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = _canonical_hash(semantic)
    _write_json(manifest_path, manifest)
    return cast(str, manifest["manifest_hash"])


def _run_replay(artifact: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_rvir_v4_optimizer_step_artifact.py"),
            "replay",
            "--artifact-dir",
            str(artifact),
        ),
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )


def run_probe_suite(artifact: Path) -> dict[str, object]:
    """Replay the original and five fully re-signed semantic tamper copies."""

    artifact = artifact.resolve()
    original = _run_replay(artifact)
    if original.returncode != 0:
        raise RuntimeError(f"original RVIR-v4 replay failed: {original.stdout}")
    original_result = json.loads(original.stdout.strip().splitlines()[-1])
    manifest_sha256 = _file_sha256(artifact / "manifest.json")
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-tamper-") as temporary:
        root = Path(temporary)
        for name, expected_error, mutate in PROBES:
            probe = root / name
            shutil.copytree(artifact, probe)
            capture = deepcopy(
                _load_capture(probe / artifact_runner.WORKER_CAPTURE_FILE)
            )
            mutate(capture)
            resigned_hash = _resign_artifact(probe, capture)
            completed = _run_replay(probe)
            passed = completed.returncode != 0 and expected_error in completed.stdout
            if not passed:
                raise AssertionError(
                    f"tamper probe {name} did not fail closed as expected: "
                    f"rc={completed.returncode}\n{completed.stdout}"
                )
            results.append(
                {
                    "name": name,
                    "status": "rejected-as-expected",
                    "expected_error": expected_error,
                    "outer_manifest_resigned": True,
                    "resigned_manifest_hash": resigned_hash,
                }
            )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "probe_code_sha256": _file_sha256(Path(__file__).resolve()),
        "artifact_manifest_sha256": manifest_sha256,
        "artifact_source_git_head": _load_json(artifact / "manifest.json")[
            "source_git_head"
        ],
        "original_replay": original_result,
        "probe_count": len(results),
        "probes": results,
        "all_probes_rejected": True,
        "performance_claimed": False,
    }
    report["report_hash"] = _canonical_hash(report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the formal probe suite and write its canonical report."""

    args = _parse_args()
    report = run_probe_suite(args.artifact_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.report, report)
    print(_canonical_json(report))


if __name__ == "__main__":
    main()
