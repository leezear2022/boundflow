#!/usr/bin/env python3
"""Run or replay five fresh B2/B3-C correctness pairs without timing claims."""

# pylint: disable=protected-access,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-arguments,too-many-lines
# pylint: disable=wrong-import-position,missing-function-docstring
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import (
    _semantic_pair_failures,
    canonical_hash,
    FSG3TimingRun,
    fsg3_timing_run_from_dict,
)
from scripts import run_fsg4_b3_counter_diagnostic as diagnostic
from scripts import run_rvir_v4_production_state_capture as capture_runner

PROTOCOL_SCHEMA = "boundflow.fsg4-b3-five-fresh-protocol/v1"
REPORT_SCHEMA = "boundflow.fsg4-b3-five-fresh-report/v1"
MANIFEST_SCHEMA = "boundflow.fsg4-b3-five-fresh-manifest/v1"
PAIR_SCHEDULE = (
    ("B2", "B3-C"),
    ("B3-C", "B2"),
    ("B2", "B3-C"),
    ("B3-C", "B2"),
    ("B2", "B3-C"),
)
CODE_PATHS = diagnostic.B3C_CODE_PATHS + ("scripts/run_fsg4_b3_correctness_pairs.py",)


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"FSG4/B3 five-fresh JSON root differs: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _code_revision() -> dict[str, str]:
    return {path: _file_sha256(REPOSITORY_ROOT / path) for path in CODE_PATHS}


def _verify_code_revision(manifest: Mapping[str, Any]) -> None:
    source = manifest.get("source_git_head")
    revision = manifest.get("code_revision")
    if not isinstance(source, str) or not isinstance(revision, Mapping):
        raise ValueError("FSG4/B3 five-fresh source provenance differs")
    if set(revision) != set(CODE_PATHS):
        raise ValueError("FSG4/B3 five-fresh code inventory differs")
    if _git("rev-parse", "HEAD") == source:
        observed = _code_revision()
    else:
        observed = {
            path: hashlib.sha256(
                subprocess.run(
                    ("git", "show", f"{source}:{path}"),
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).stdout
            ).hexdigest()
            for path in CODE_PATHS
        }
    if dict(revision) != observed:
        raise ValueError("FSG4/B3 five-fresh code revision differs")


def _schedule_payload() -> list[dict[str, object]]:
    return [
        {"pair_index": index, "positions": list(configurations)}
        for index, configurations in enumerate(PAIR_SCHEDULE)
    ]


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    protocol: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "schedule": _schedule_payload(),
        "pair_count": 5,
        "worker_count": 10,
        "process_isolation": "one-diagnostic-subprocess-per-position",
        "mode": "control",
        "benchmark_id": "vnncomp2021/cifar10_resnet",
        "solver_id": "alpha-beta-crown-pinned-by-worker-source",
        "model_name": args.model.name,
        "property_name": args.property.name,
        "model_sha256": capture_runner.file_sha256(args.model.resolve()),
        "property_sha256": capture_runner.file_sha256(args.property.resolve()),
        "resume_policy": "accept-complete-replay-only",
        "direct_semantic_atol": 2e-4,
        "direct_semantic_rtol": 2e-4,
        "timing_admitted": False,
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def _validate_protocol(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("protocol_hash", None)
    if (
        value.get("schema_version") != PROTOCOL_SCHEMA
        or value.get("source_git_head") is None
        or value.get("schedule") != _schedule_payload()
        or value.get("pair_count") != 5
        or value.get("worker_count") != 10
        or value.get("process_isolation") != "one-diagnostic-subprocess-per-position"
        or value.get("mode") != "control"
        or value.get("resume_policy") != "accept-complete-replay-only"
        or value.get("direct_semantic_atol") != 2e-4
        or value.get("direct_semantic_rtol") != 2e-4
        or value.get("timing_admitted") is not False
        or value.get("performance_claimed") is not False
        or claimed != canonical_hash(payload)
    ):
        raise ValueError("FSG4/B3 five-fresh protocol differs")


def _run_directory(root: Path, pair_index: int, position: int, config: str) -> Path:
    return (
        root
        / "runs"
        / f"pair-{pair_index:02d}"
        / f"position-{position}-{config.lower()}"
    )


def _diagnostic_command(
    args: argparse.Namespace,
    *,
    configuration: str,
    target: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/run_fsg4_b3_counter_diagnostic.py"),
        "run",
        "--artifact-dir",
        str(target),
        "--benchmark-root",
        str(args.benchmark_root.resolve()),
        "--abcrown-root",
        str(args.abcrown_root.resolve()),
        "--model",
        str(args.model.resolve()),
        "--property",
        str(args.property.resolve()),
        "--configuration",
        configuration,
    )


def _load_raw_run(root: Path) -> tuple[FSG3TimingRun, dict[str, Any], dict[str, Any]]:
    report = _load_json(root / "report.json")
    worker = _load_json(root / "worker.json")
    raw_run = worker.get("run")
    if not isinstance(raw_run, Mapping):
        raise TypeError("FSG4/B3 five-fresh worker run differs")
    return fsg3_timing_run_from_dict(cast(Mapping[str, Any], raw_run)), report, worker


def _zero_provider(run: FSG3TimingRun) -> bool:
    return (
        run.execution.provider_core_call_count == 0
        and run.execution.provider_compute_bounds_call_count == 0
        and run.execution.provider_update_bounds_call_count == 0
        and run.execution.fallback_dispatch_count == 0
    )


def _snapshot_counts(report: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshot = report.get("snapshot")
    if not isinstance(snapshot, Mapping) or not isinstance(
        snapshot.get("counts"), Mapping
    ):
        raise TypeError("FSG4/B3 five-fresh counter snapshot differs")
    return cast(Mapping[str, Any], snapshot["counts"])


def _validate_b3c_diagnostics(worker: Mapping[str, Any]) -> dict[str, object]:
    diagnostics = worker.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise TypeError("FSG4/B3 five-fresh diagnostics differs")
    assemblies = diagnostics.get("assembly_metadata")
    receipts = diagnostics.get("commit_receipts")
    audits = diagnostics.get("device_commit_audits")
    if (
        diagnostics.get("post_query_audit_excluded_from_timing") is not True
        or not isinstance(diagnostics.get("post_query_audit_ns"), int)
        or int(diagnostics["post_query_audit_ns"]) <= 0
        or not isinstance(assemblies, list)
        or len(assemblies) != 1
        or not isinstance(receipts, list)
        or len(receipts) != 1
        or not isinstance(audits, list)
        or len(audits) != 1
        or not all(
            isinstance(item, Mapping) for item in (*assemblies, *receipts, *audits)
        )
    ):
        raise ValueError("FSG4/B3 five-fresh audit cardinality differs")
    assembly = cast(Mapping[str, Any], assemblies[0])
    receipt = cast(Mapping[str, Any], receipts[0])
    audit = cast(Mapping[str, Any], audits[0])
    path_digests = audit.get("path_digests")
    if (
        assembly.get("headline_content_digest_count") != 0
        or assembly.get("candidate_device_resident") is not True
        or receipt.get("candidate_d2h_copy_count") != 0
        or receipt.get("committed_path_count") != 12
        or receipt.get("device_rollback_backup_count") != 12
        or audit.get("commit_hash") != receipt.get("commit_hash")
        or audit.get("headline_timing_excluded") is not True
        or audit.get("content_audit_complete") is not True
        or not isinstance(path_digests, list)
        or len(path_digests) != 12
    ):
        raise ValueError("FSG4/B3 five-fresh audit binding differs")
    return {
        "assembly_hash": assembly["assembly_hash"],
        "commit_hash": receipt["commit_hash"],
        "audit_hash": audit["audit_hash"],
        "post_query_audit_ns": diagnostics["post_query_audit_ns"],
        "headline_content_digest_count": 0,
    }


def _pair_row(root: Path, pair_index: int) -> dict[str, object]:
    by_configuration: dict[
        str, tuple[Path, FSG3TimingRun, dict[str, Any], dict[str, Any]]
    ] = {}
    for position, configuration in enumerate(PAIR_SCHEDULE[pair_index]):
        run_root = _run_directory(root, pair_index, position, configuration)
        diagnostic._replay(run_root)
        run, report, worker = _load_raw_run(run_root)
        if (
            report.get("configuration") != configuration
            or report.get("source_git_head") != _git("rev-parse", "HEAD")
            and report.get("source_git_head")
            != _load_json(root / "protocol.json").get("source_git_head")
            or not run.environment.admitted
            or not _zero_provider(run)
        ):
            raise ValueError("FSG4/B3 five-fresh raw run gate differs")
        by_configuration[configuration] = (run_root, run, report, worker)
    if set(by_configuration) != {"B2", "B3-C"}:
        raise ValueError("FSG4/B3 five-fresh pair inventory differs")
    b2_root, b2, b2_report, _b2_worker = by_configuration["B2"]
    b3c_root, b3c, b3c_report, b3c_worker = by_configuration["B3-C"]
    failures = _semantic_pair_failures(
        b2.semantics, b3c.semantics, label=f"pair-{pair_index:02d}"
    )
    if (
        failures
        or b2.source_identity != b3c.source_identity
        or b2.protocol_identity != b3c.protocol_identity
        or b2.environment.gpu_uuid != b3c.environment.gpu_uuid
        or b2.environment.gpu_name != b3c.environment.gpu_name
        or b2.environment.runtime_identity != b3c.environment.runtime_identity
    ):
        raise ValueError(
            "FSG4/B3 five-fresh direct pair differs: " + ",".join(failures)
        )
    b2_counts = _snapshot_counts(b2_report)
    b3c_counts = _snapshot_counts(b3c_report)
    if (
        b2_counts.get("timed_candidate_d2h_copy_count") != 12
        or b2_counts.get("full_optimizer_step_snapshot_count") != 10
        or b2_counts.get("forward_trace_build_count") != 5
        or b3c_counts.get("timed_candidate_d2h_copy_count") != 0
        or b3c_counts.get("candidate_snapshot_materialization_count") != 12
        or b3c_counts.get("committed_mutable_path_count") != 12
        or b3c_counts.get("device_rollback_backup_count") != 12
        or b3c_counts.get("commit_copy_call_count") != 12
        or b3c_counts.get("optimizer_evaluation_count") != 10
        or b3c_counts.get("optimizer_update_count") != 9
        or b3c_counts.get("full_optimizer_step_snapshot_count") != 0
        or b3c_counts.get("forward_trace_build_count") != 4
        or b3c_counts.get("kfsb_candidate_count") != 3
        or b3c_counts.get("kfsb_child_batch_count") != 3
    ):
        raise ValueError("FSG4/B3 five-fresh physical counter differs")
    audit = _validate_b3c_diagnostics(b3c_worker)
    return {
        "pair_index": pair_index,
        "schedule": list(PAIR_SCHEDULE[pair_index]),
        "b2_manifest_hash": _load_json(b2_root / "manifest.json")["manifest_hash"],
        "b3c_manifest_hash": _load_json(b3c_root / "manifest.json")["manifest_hash"],
        "source_identity": b2.source_identity,
        "protocol_identity": b2.protocol_identity,
        "gpu_uuid": b2.environment.gpu_uuid,
        "runtime_identity": b2.environment.runtime_identity,
        "b2_semantic_hash": canonical_hash(b2.semantics.to_dict()),
        "b3c_semantic_hash": canonical_hash(b3c.semantics.to_dict()),
        "semantic_failures": [],
        "environment_admitted": True,
        "provider_fallback_zero": True,
        "b2_counter_gate_passed": True,
        "b3c_counter_gate_passed": True,
        "b3c_audit": audit,
        "timing_admitted": False,
        "performance_claimed": False,
    }


def _all_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    }


def _complete(root: Path) -> dict[str, object]:
    protocol = _load_json(root / "protocol.json")
    _validate_protocol(protocol)
    pairs = [_pair_row(root, index) for index in range(5)]
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "source_git_head": protocol["source_git_head"],
        "protocol_hash": protocol["protocol_hash"],
        "pair_count": len(pairs),
        "worker_count": 10,
        "pairs": pairs,
        "all_direct_semantic_pairs_passed": True,
        "all_environments_admitted": True,
        "all_provider_fallback_zero": True,
        "all_counter_gates_passed": True,
        "all_post_query_audits_passed": True,
        "timing_admitted": False,
        "performance_claimed": False,
    }
    report["report_hash"] = canonical_hash(report)
    _write_json(root / "report.json", report)
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "source_git_head": protocol["source_git_head"],
        "code_revision": protocol["code_revision"],
        "protocol_hash": protocol["protocol_hash"],
        "report_hash": report["report_hash"],
        "files": _all_files(root),
        "pair_count": 5,
        "worker_count": 10,
        "timing_admitted": False,
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(root / "manifest.json", manifest)
    return _replay(root)


def _run(args: argparse.Namespace) -> dict[str, object]:
    root = args.artifact_dir.resolve()
    dirty = _git("status", "--porcelain=v1", "--", *CODE_PATHS)
    if dirty:
        raise RuntimeError("FSG4/B3 five-fresh code paths must be committed")
    expected_protocol = _protocol(args)
    if root.exists():
        if not args.resume:
            raise FileExistsError(f"FSG4/B3 five-fresh artifact exists: {root}")
        if (root / "manifest.json").exists():
            return _replay(root)
        observed_protocol = _load_json(root / "protocol.json")
        if observed_protocol != expected_protocol:
            raise ValueError("FSG4/B3 five-fresh resume protocol differs")
    else:
        root.mkdir(parents=True)
        _write_json(root / "protocol.json", expected_protocol)
    for pair_index, configurations in enumerate(PAIR_SCHEDULE):
        for position, configuration in enumerate(configurations):
            target = _run_directory(root, pair_index, position, configuration)
            if target.exists():
                diagnostic._replay(target)
                report = _load_json(target / "report.json")
                if report.get("configuration") != configuration:
                    raise ValueError("FSG4/B3 five-fresh resume configuration differs")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                _diagnostic_command(args, configuration=configuration, target=target),
                cwd=REPOSITORY_ROOT,
                check=True,
            )
            diagnostic._replay(target)
    return _complete(root)


def _replay(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = _load_json(root / "manifest.json")
    report = _load_json(root / "report.json")
    protocol = _load_json(root / "protocol.json")
    manifest_payload = dict(manifest)
    claimed_manifest_hash = manifest_payload.pop("manifest_hash", None)
    report_payload = dict(report)
    claimed_report_hash = report_payload.pop("report_hash", None)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or report.get("schema_version") != REPORT_SCHEMA
        or manifest.get("pair_count") != 5
        or manifest.get("worker_count") != 10
        or manifest.get("timing_admitted") is not False
        or manifest.get("performance_claimed") is not False
        or claimed_manifest_hash != canonical_hash(manifest_payload)
        or claimed_report_hash != canonical_hash(report_payload)
        or manifest.get("report_hash") != claimed_report_hash
    ):
        raise ValueError("FSG4/B3 five-fresh root hash differs")
    _verify_code_revision(manifest)
    _validate_protocol(protocol)
    if (
        protocol.get("protocol_hash") != manifest.get("protocol_hash")
        or protocol.get("protocol_hash") != report.get("protocol_hash")
        or protocol.get("source_git_head") != manifest.get("source_git_head")
        or protocol.get("source_git_head") != report.get("source_git_head")
    ):
        raise ValueError("FSG4/B3 five-fresh protocol binding differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or dict(files) != _all_files(root):
        raise ValueError("FSG4/B3 five-fresh file inventory differs")
    pairs = [_pair_row(root, index) for index in range(5)]
    expected_report = {
        "schema_version": REPORT_SCHEMA,
        "source_git_head": protocol["source_git_head"],
        "protocol_hash": protocol["protocol_hash"],
        "pair_count": 5,
        "worker_count": 10,
        "pairs": pairs,
        "all_direct_semantic_pairs_passed": True,
        "all_environments_admitted": True,
        "all_provider_fallback_zero": True,
        "all_counter_gates_passed": True,
        "all_post_query_audits_passed": True,
        "timing_admitted": False,
        "performance_claimed": False,
    }
    expected_report["report_hash"] = canonical_hash(expected_report)
    if report != expected_report:
        raise ValueError("FSG4/B3 five-fresh report projection differs")
    result: dict[str, object] = {
        "status": "replay-passed",
        "source_git_head": manifest["source_git_head"],
        "pair_count": 5,
        "worker_count": 10,
        "report_hash": claimed_report_hash,
        "manifest_hash": claimed_manifest_hash,
        "timing_admitted": False,
        "performance_claimed": False,
    }
    print(_canonical_json(result), flush=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--artifact-dir", type=Path, required=True)
    run.add_argument("--benchmark-root", type=Path, required=True)
    run.add_argument("--abcrown-root", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--property", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        result = _run(args)
    else:
        result = _replay(args.artifact_dir)
    print(_canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
