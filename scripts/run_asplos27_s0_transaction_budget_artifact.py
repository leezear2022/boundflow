#!/usr/bin/env python3
"""Generate or replay the S0 explicit-transaction 10x research budget."""

# pylint: disable=missing-function-docstring,protected-access,wrong-import-position
# pylint: disable=too-many-branches

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.asplos27_transaction_budget import (  # noqa: E402
    DEFAULT_AXIS_POLICIES,
    TRANSACTION_BUDGET_SCHEMA_VERSION,
    derive_transaction_budgets,
)
from boundflow.runtime.gpu_attribution import canonical_hash  # noqa: E402
from scripts import run_asplos27_s0_transaction_markers as transactions  # noqa: E402

ARTIFACT_SCHEMA_VERSION = "boundflow.asplos27-s0-transaction-budget-artifact/v1"
SOURCE_ARTIFACT = Path("artifacts/asplos27-s0-transactions/official-b0-five-pair-v1")
DEFAULT_ARTIFACT = Path(
    "artifacts/asplos27-s0-transaction-budget/official-b0-five-pair-v1"
)
SOURCE_FILES = (
    "manifest.json",
    "protocol.json",
    "worker_runs.jsonl",
    "pairs.jsonl",
    "summary.json",
)
OUTPUT_FILES = (
    "protocol.json",
    "budget_report.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CODE_FILES = (
    Path("boundflow/runtime/asplos27_transaction_budget.py"),
    Path("scripts/run_asplos27_s0_transaction_budget_artifact.py"),
)


def canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"transaction budget JSON root differs: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"transaction budget JSONL row differs: {path}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _protocol() -> dict[str, object]:
    protocol: dict[str, object] = {
        "schema_version": TRANSACTION_BUDGET_SCHEMA_VERSION,
        "source_artifact": str(SOURCE_ARTIFACT),
        "source_profile_count": 10,
        "source_repeats_per_workload": 5,
        "target_speedup": 10.0,
        "integration_overhead_share": 0.0,
        "pooling": "sum_category_ns_over_sum_profile_scope_ns_per_workload",
        "unresolved_policy": "immutable_at_1x",
        "axis_policies": [policy.to_dict() for policy in DEFAULT_AXIS_POLICIES],
        "decision_rules": [
            "source_explicit_transaction_artifact_must_replay_and_be_admitted",
            "every_exact_category_has_exactly_one_optimization_axis_owner",
            "unresolved_share_is_immutable_and_never_renamed_as_solver_control",
            "projection_assumes_integration_overhead_h_equals_zero",
            "research_targets_open_implementation_but_not_performance_claims",
            "every_axis_target_requires_direct_same_scope_validation",
        ],
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def _summary(report: Mapping[str, Any]) -> dict[str, object]:
    workloads = report.get("workloads")
    if not isinstance(workloads, Sequence) or isinstance(workloads, (str, bytes)):
        raise TypeError("transaction budget workload report differs")
    projected = [
        float(workload["projected_speedup_hypothesis"])
        for workload in workloads
        if isinstance(workload, Mapping)
    ]
    unresolved = [
        float(workload["unresolved_share"])
        for workload in workloads
        if isinstance(workload, Mapping)
    ]
    if len(projected) != len(workloads) or not projected:
        raise TypeError("transaction budget workload projection differs")
    summary: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": report["status"],
        "source_evidence_status": "s0-explicit-transactions-admitted",
        "workload_count": report["workload_count"],
        "profile_count": report["profile_count"],
        "target_speedup": report["target_speedup"],
        "integration_overhead_share": report["integration_overhead_share"],
        "minimum_projected_speedup_hypothesis": min(projected),
        "maximum_unresolved_share": max(unresolved),
        "all_workloads_tenx_feasible_hypothesis": report[
            "all_workloads_tenx_feasible_hypothesis"
        ],
        "s1_implementation_open": report["s1_implementation_open"],
        "s1_performance_gate_open": False,
        "required_next_action": (
            "implement_S1_canonical_CIBC_Primal_Bound_Plan_TIR_prepared_path_and_"
            "validate_O1_O3_directly"
        ),
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "workload_count": summary["workload_count"],
        "s1_implementation_open": summary["s1_implementation_open"],
        "s1_performance_gate_open": False,
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# ASPLOS'27 S0 Explicit-Transaction 10x Research Budget\n\n"
        "This artifact pools five admitted profile runs per workload into exclusive "
        "optimization axes. Axis speedups are preregistered research targets, not "
        "measurements. The projection explicitly assumes integration overhead h=0. "
        "It opens S1 implementation only; performance claims remain closed until "
        "direct same-scope candidates validate each target and measure h.\n"
    )


def _derived_outputs(root: Path) -> dict[str, object]:
    source_dir = root / SOURCE_ARTIFACT
    replay = transactions.replay_artifact(source_dir)
    if (
        replay.get("evidence_status") != "s0-explicit-transactions-admitted"
        or replay.get("budget_recompute_open") is not True
        or replay.get("performance_claimed") is not False
    ):
        raise ValueError("transaction budget source evidence is not admitted")
    records = _read_jsonl(source_dir / "worker_runs.jsonl")
    report = derive_transaction_budgets(
        records,
        repeats=5,
        target_speedup=10.0,
        integration_overhead_share=0.0,
    )
    summary = _summary(report)
    return {
        "protocol.json": _protocol(),
        "budget_report.json": report,
        "summary.json": summary,
        "replay_stdout.txt": canonical_json(_replay_result(summary)) + "\n",
        "README.md": _readme(),
    }


def _output_text(name: str, value: object) -> str:
    if name.endswith(".json"):
        return canonical_json(value, indent=2) + "\n"
    if not isinstance(value, str):
        raise TypeError(f"transaction budget text output differs: {name}")
    return value


def _write_outputs(artifact_dir: Path, outputs: Mapping[str, object]) -> None:
    for name, value in outputs.items():
        (artifact_dir / name).write_text(_output_text(name, value), encoding="utf-8")


def _write_manifest(root: Path, artifact_dir: Path, summary: Mapping[str, Any]) -> None:
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": summary["status"],
        "source_files": {
            name: file_sha256(root / SOURCE_ARTIFACT / name) for name in SOURCE_FILES
        },
        "code_revision": {str(path): file_sha256(root / path) for path in CODE_FILES},
        "files": {name: file_sha256(artifact_dir / name) for name in OUTPUT_FILES},
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact_dir / "manifest.json", manifest)


def generate_artifact(root: Path, artifact_dir: Path) -> Mapping[str, Any]:
    root = root.resolve()
    artifact_dir = artifact_dir.resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outputs = _derived_outputs(root)
    _write_outputs(artifact_dir, outputs)
    summary = cast(Mapping[str, Any], outputs["summary.json"])
    _write_manifest(root, artifact_dir, summary)
    return _replay_result(summary)


def refresh_artifact(root: Path, artifact_dir: Path) -> Mapping[str, Any]:
    """Regenerate this derived budget from the admitted transaction raw."""

    root = root.resolve()
    artifact_dir = artifact_dir.resolve()
    outputs = _derived_outputs(root)
    _write_outputs(artifact_dir, outputs)
    summary = cast(Mapping[str, Any], outputs["summary.json"])
    _write_manifest(root, artifact_dir, summary)
    return _replay_result(summary)


def replay_artifact(root: Path, artifact_dir: Path) -> Mapping[str, Any]:
    root = root.resolve()
    artifact_dir = artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
        or manifest.get("manifest_hash") != canonical_hash(unsigned)
    ):
        raise ValueError("transaction budget manifest envelope differs")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != set(SOURCE_FILES):
        raise ValueError("transaction budget source inventory differs")
    for name in SOURCE_FILES:
        if source_files[name] != file_sha256(root / SOURCE_ARTIFACT / name):
            raise ValueError("transaction budget source digest differs")
    code_revision = manifest.get("code_revision")
    if not isinstance(code_revision, Mapping) or set(code_revision) != {
        str(path) for path in CODE_FILES
    }:
        raise ValueError("transaction budget code inventory differs")
    for path in CODE_FILES:
        if code_revision[str(path)] != file_sha256(root / path):
            raise ValueError("transaction budget code revision differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(OUTPUT_FILES):
        raise ValueError("transaction budget output inventory differs")
    for name in OUTPUT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError("transaction budget output digest differs")
    expected = _derived_outputs(root)
    for name, value in expected.items():
        if (artifact_dir / name).read_text(encoding="utf-8") != _output_text(
            name, value
        ):
            raise ValueError(f"transaction budget semantic replay differs: {name}")
    summary = cast(Mapping[str, Any], expected["summary.json"])
    if (
        manifest.get("status") != summary["status"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("transaction budget summary projection differs")
    return _replay_result(summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--root", type=Path, default=ROOT)
    generate.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--root", type=Path, default=ROOT)
    replay.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    refresh = subparsers.add_parser("refresh-derived")
    refresh.add_argument("--root", type=Path, default=ROOT)
    refresh.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        result = generate_artifact(args.root, args.artifact_dir)
    elif args.command == "refresh-derived":
        result = refresh_artifact(args.root, args.artifact_dir)
    else:
        result = replay_artifact(args.root, args.artifact_dir)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
