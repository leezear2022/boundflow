#!/usr/bin/env python3
"""Generate or replay the formal R3-3 isolated attribution artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=duplicate-code,wrong-import-position

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-3-isolated-attribution-v1"
WORKER = ROOT / "scripts/run_r3_3_isolated_attribution_worker.py"
CAPTURE = ROOT / "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
RUN_COUNT = 5
WARMUP_COUNT = 10
SAMPLE_COUNT = 30
CALIBRATION_ELEMENT_COUNT = 1 << 20
CODE_PATHS = (
    "boundflow/runtime/r3_3_isolated_attribution.py",
    "scripts/run_r3_3_isolated_attribution_worker.py",
    "scripts/run_r3_3_isolated_attribution_artifact.py",
    "scripts/probe_r3_3_isolated_attribution_tamper.py",
    "gemini_doc/BOUNDFLOW_R3_3_ISOLATED_MICROPHYSICS_ATTRIBUTION_PLAN_2026_08_26.md",
    "gemini_doc/BOUNDFLOW_R3_3_ISOLATED_ATTRIBUTION_IMPLEMENTATION_CHANGE_2026_08_26.md",
)

from boundflow.runtime.r3_3_isolated_attribution import (  # noqa: E402
    CURRENT_SPEEDUP,
    MAX_PROFILE_PERTURBATION,
    MAX_REQUIRED_BUCKET_SPEEDUP,
    MAX_UNEXPLAINED_SHARE,
    TARGET_SPEEDUP,
    canonical_hash,
    derive_ledger,
    derive_route_or_stop,
    event_from_dict,
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("R3-3 attribution JSON root differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _portable_log(value: str) -> str:
    return value.replace(str(ROOT), "<repo>").replace(sys.prefix, "<python-prefix>")


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _git_blob_hash(revision: str, path: str) -> str:
    content = subprocess.check_output(("git", "show", f"{revision}:{path}"), cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def _clean() -> None:
    ignored = (
        "docs/CIBC_for_DAC.pdf",
        ".docops/ev.jsonl",
        ".docops/s.md",
        "gemini_doc/README.md",
        "gemini_doc/asplos_claims_map.md",
        "gemini_doc/asplos_execution_memo_v1_0.md",
        "gemini_doc/current_status_after_pr13.md",
        "gemini_doc/BOUNDFLOW_R3_3_ISOLATED_ATTRIBUTION_FORMAL_STOP_CLOSURE_2026_08_26.md",
    )
    dirty = [
        row
        for row in _git("status", "--porcelain").splitlines()
        if not any(row.endswith(name) for name in ignored)
    ]
    if dirty:
        raise RuntimeError(f"R3-3 attribution source is dirty: {dirty}")


def _validate_worker(
    raw: Mapping[str, Any], ordinal: int, protocol: Mapping[str, Any]
) -> dict[str, object]:
    unsigned = dict(raw)
    worker_hash = unsigned.pop("worker_hash", None)
    expected = {
        "schema_version",
        "run_index",
        "capture_sha256",
        "warmup_count",
        "sample_count",
        "latency_ns",
        "median_latency_ns",
        "profiled_cuda_event_ns",
        "calibration_cuda_event_ns",
        "calibration_element_count",
        "parity",
        "template_hash",
        "schedule_hash",
        "module_receipt_hash",
        "events",
        "event_hash",
        "ledger",
        "output_shapes",
        "gpu_before",
        "gpu_after",
        "environment",
        "performance_claimed",
        "worker_hash",
    }
    captures = protocol.get("capture_sha256")
    samples = raw.get("latency_ns")
    parity = raw.get("parity")
    events_raw = raw.get("events")
    ledger = raw.get("ledger")
    if (
        set(raw) != expected
        or worker_hash != canonical_hash(unsigned)
        or raw.get("schema_version") != "boundflow.r3-3-isolated-attribution-worker/v1"
        or raw.get("run_index") != ordinal
        or not isinstance(captures, dict)
        or raw.get("capture_sha256") != captures[f"run_{ordinal:02d}.pt"]
        or raw.get("warmup_count") != WARMUP_COUNT
        or raw.get("sample_count") != SAMPLE_COUNT
        or raw.get("calibration_element_count") != CALIBRATION_ELEMENT_COUNT
        or not isinstance(samples, list)
        or len(samples) != SAMPLE_COUNT
        or any(not isinstance(value, int) or value <= 0 for value in samples)
        or raw.get("median_latency_ns") != round(statistics.median(samples))
        or not isinstance(parity, dict)
        or parity.get("allclose") is not True
        or parity.get("sign_exact") is not True
        or float(parity.get("maximum_absolute_difference", math.inf)) > 2.0e-4
        or not isinstance(events_raw, list)
        or not events_raw
        or not isinstance(ledger, dict)
        or raw.get("performance_claimed") is not False
    ):
        raise ValueError("R3-3 attribution worker envelope differs")
    events = tuple(event_from_dict(row) for row in events_raw)
    if raw.get("event_hash") != canonical_hash(events_raw):
        raise ValueError("R3-3 attribution worker event hash differs")
    rebuilt = derive_ledger(
        events,
        unprofiled_median_ns=int(raw["median_latency_ns"]),
        profiled_cuda_event_ns=int(raw["profiled_cuda_event_ns"]),
        calibration_cuda_event_ns=int(raw["calibration_cuda_event_ns"]),
    )
    if rebuilt != ledger:
        raise ValueError("R3-3 attribution worker ledger differs")
    return rebuilt


def _summary(raws: list[dict[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    if len(raws) != RUN_COUNT:
        raise ValueError("R3-3 attribution worker count differs")
    ledgers = [
        _validate_worker(raw, ordinal, protocol) for ordinal, raw in enumerate(raws)
    ]
    identities = (
        {raw["template_hash"] for raw in raws},
        {raw["schedule_hash"] for raw in raws},
        {raw["module_receipt_hash"] for raw in raws},
    )
    if any(len(values) != 1 for values in identities):
        raise ValueError("R3-3 attribution compiler identity differs")
    route = derive_route_or_stop(ledgers)
    admitted = all(ledger["attribution_admitted"] is True for ledger in ledgers)
    verdict = (
        f"VALIDATED-R3-3-ISOLATED-ATTRIBUTION-ROUTE-{route['route']}"
        if admitted
        else "VALIDATED-R3-3-ISOLATED-ATTRIBUTION-STOP-QUALITY"
    )
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-3-isolated-attribution-summary/v1",
        "worker_count": RUN_COUNT,
        "sample_count": RUN_COUNT * SAMPLE_COUNT,
        "attribution_admitted": admitted,
        "admitted_worker_count": sum(
            ledger["attribution_admitted"] is True for ledger in ledgers
        ),
        "worker_median_latency_ns": [
            ledger["unprofiled_median_ns"] for ledger in ledgers
        ],
        "worker_profile_perturbation_ratios": [
            ledger["profile_perturbation_ratio"] for ledger in ledgers
        ],
        "worker_calibration_residual_ns": [
            ledger["calibration_residual_ns"] for ledger in ledgers
        ],
        "worker_unexplained_shares": [
            cast(Mapping[str, Any], ledger["bucket_share"])["unexplained"]
            for ledger in ledgers
        ],
        "worker_admission_failures": [
            ledger["admission_failures"] for ledger in ledgers
        ],
        "route_decision": route,
        "provisional_verdict": verdict,
        "template_hash": next(iter(identities[0])),
        "schedule_hash": next(iter(identities[1])),
        "module_receipt_hash": next(iter(identities[2])),
        "attribution_closure_pending_tamper": True,
        "diagnostic_shares_are_performance_claim": False,
        "r3_4_open": False,
        "same_solver_open": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = canonical_hash(result)
    return result


def _protocol(revision: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-3-isolated-attribution-protocol/v1",
        "source_revision": revision,
        "worker_count": RUN_COUNT,
        "capture_ordinals": list(range(RUN_COUNT)),
        "warmup_count": WARMUP_COUNT,
        "samples_per_worker": SAMPLE_COUNT,
        "calibration_element_count": CALIBRATION_ELEMENT_COUNT,
        "current_speedup": CURRENT_SPEEDUP,
        "target_speedup": TARGET_SPEEDUP,
        "maximum_required_bucket_speedup": MAX_REQUIRED_BUCKET_SPEEDUP,
        "maximum_profile_perturbation": MAX_PROFILE_PERTURBATION,
        "maximum_unexplained_share": MAX_UNEXPLAINED_SHARE,
        "route_priority": ["KERNEL", "BRIDGE", "AUTOGRAD", "CUMULATIVE", "STOP"],
        "r3_4_open": False,
        "same_solver_open": False,
        "performance_claimed": False,
        "capture_sha256": {
            f"run_{ordinal:02d}.pt": _file_hash(CAPTURE / f"run_{ordinal:02d}.pt")
            for ordinal in range(RUN_COUNT)
        },
        "code_revision": {name: _file_hash(ROOT / name) for name in CODE_PATHS},
    }
    result["protocol_hash"] = canonical_hash(result)
    return result


def generate(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3-3 attribution artifact exists: {output}")
    _clean()
    protocol = _protocol(_git("rev-parse", "HEAD"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="r3-3-attribution-", dir=output.parent))
    try:
        raw_dir, log_dir = temporary / "raw", temporary / "logs"
        raw_dir.mkdir()
        log_dir.mkdir()
        raws = []
        for ordinal in range(RUN_COUNT):
            target = raw_dir / f"run_{ordinal:02d}.json"
            completed = subprocess.run(
                (
                    sys.executable,
                    str(WORKER),
                    "--run-index",
                    str(ordinal),
                    "--result",
                    str(target),
                ),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            (log_dir / f"run_{ordinal:02d}.stdout.txt").write_text(
                _portable_log(completed.stdout), encoding="utf-8"
            )
            (log_dir / f"run_{ordinal:02d}.stderr.txt").write_text(
                _portable_log(completed.stderr), encoding="utf-8"
            )
            raws.append(_json(target))
        summary = _summary(raws, protocol)
        _write(temporary / "protocol.json", protocol)
        _write(temporary / "summary.json", summary)
        files = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, Any] = {
            "schema_version": "boundflow.r3-3-isolated-attribution-manifest/v1",
            "source_revision": protocol["source_revision"],
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": files,
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        _write(temporary / "manifest.json", manifest)
        replay(temporary)
        temporary.rename(output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay(artifact: Path) -> dict[str, Any]:
    manifest = _json(artifact / "manifest.json")
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if manifest.get(
        "schema_version"
    ) != "boundflow.r3-3-isolated-attribution-manifest/v1" or manifest_hash != canonical_hash(
        unsigned_manifest
    ):
        raise ValueError("R3-3 attribution manifest differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(
        _file_hash(artifact / name) != digest for name, digest in files.items()
    ):
        raise ValueError("R3-3 attribution file digest differs")
    for name in files:
        if str(name).endswith(".txt"):
            text = (artifact / str(name)).read_text(encoding="utf-8")
            if _portable_log(text) != text or any(
                token in text for token in ("/home/", "file://", "\\Users\\")
            ):
                raise ValueError("R3-3 attribution log leaks a local path")
    protocol = _json(artifact / "protocol.json")
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if protocol_hash != canonical_hash(
        unsigned_protocol
    ) or protocol_hash != manifest.get("protocol_hash"):
        raise ValueError("R3-3 attribution protocol hash differs")
    frozen = {
        "worker_count": RUN_COUNT,
        "capture_ordinals": list(range(RUN_COUNT)),
        "warmup_count": WARMUP_COUNT,
        "samples_per_worker": SAMPLE_COUNT,
        "calibration_element_count": CALIBRATION_ELEMENT_COUNT,
        "current_speedup": CURRENT_SPEEDUP,
        "target_speedup": TARGET_SPEEDUP,
        "maximum_required_bucket_speedup": MAX_REQUIRED_BUCKET_SPEEDUP,
        "maximum_profile_perturbation": MAX_PROFILE_PERTURBATION,
        "maximum_unexplained_share": MAX_UNEXPLAINED_SHARE,
        "route_priority": ["KERNEL", "BRIDGE", "AUTOGRAD", "CUMULATIVE", "STOP"],
        "r3_4_open": False,
        "same_solver_open": False,
        "performance_claimed": False,
    }
    if any(protocol.get(name) != value for name, value in frozen.items()):
        raise ValueError("R3-3 attribution frozen protocol differs")
    revision, code = protocol.get("source_revision"), protocol.get("code_revision")
    if (
        not isinstance(revision, str)
        or not isinstance(code, dict)
        or set(code) != set(CODE_PATHS)
        or any(
            _git_blob_hash(revision, name) != digest for name, digest in code.items()
        )
    ):
        raise ValueError("R3-3 attribution source binding differs")
    raws = [_json(path) for path in sorted((artifact / "raw").glob("*.json"))]
    summary = _summary(raws, protocol)
    if summary != _json(artifact / "summary.json") or summary[
        "summary_hash"
    ] != manifest.get("summary_hash"):
        raise ValueError("R3-3 attribution semantic replay differs")
    print(
        f"R3-3 attribution replay PASS: admitted="
        f"{summary['admitted_worker_count']}/{summary['worker_count']} "
        f"route={summary['route_decision']['route']} "
        f"verdict={summary['provisional_verdict']}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.output.absolute())
    else:
        generate(args.output.absolute())


if __name__ == "__main__":
    main()
