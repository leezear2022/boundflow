#!/usr/bin/env python3
"""Generate or replay the ASPLOS'27 S0 attribution and 10x budget artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from boundflow.runtime.asplos27_tenx_budget import (
    ASPLOS27_TENX_BUDGET_SCHEMA_VERSION,
    DirectCumulativeObservation,
    EvidenceScope,
    derive_fsg1_diagnostic_budgets,
    derive_fsg1_transaction_inventory,
    validate_direct_observation_ledger,
)
from boundflow.runtime.gpu_attribution import canonical_hash

ARTIFACT_SCHEMA_VERSION = "boundflow.asplos27-s0-tenx-artifact/v1"
DEFAULT_ARTIFACT = Path(
    "artifacts/asplos27-s0-tenx-budget/fsg1-diagnostic-and-history-v1"
)
SOURCE_FILES = {
    "fsg1_closure": Path(
        "artifacts/fsg1-official-control/"
        "resnet2b-mnistfc2-rtx4060-five-repeat-v1/closure.json"
    ),
    "fsg1_worker_runs": Path(
        "artifacts/fsg1-official-control/resnet2b-mnistfc2-rtx4060-"
        "five-repeat-v1/worker_runs.jsonl"
    ),
    "fsg4_b3_summary": Path("artifacts/fsg4-b3-same-solver-timing")
    / "resnet2b-prop0-v1/summary.json",
    "mr5_summary": Path(
        "artifacts/measurement-recovery/mr5-multi-conv-timing-v1/summary.json"
    ),
    "mr6_summary": Path(
        "artifacts/measurement-recovery/mr6-hot-path-guard-attribution-v1/summary.json"
    ),
    "cibc_summary": Path(
        "artifacts/cibc-ibp-horizontal-formal/resnet2b-prop0-v1/summary.json"
    ),
}
OUTPUT_FILES = (
    "protocol.json",
    "budget_report.json",
    "transaction_inventory.json",
    "direct_observation_ledger.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CODE_FILES = (
    Path("boundflow/runtime/asplos27_tenx_budget.py"),
    Path("scripts/run_asplos27_s0_tenx_budget_artifact.py"),
)


def canonical_json(value: object, *, indent: int | None = None) -> str:
    """Return deterministic JSON and reject non-finite values."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def file_sha256(path: Path) -> str:
    """Return a source or artifact file digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"S0 JSON root must be a mapping: {path}")
    return value


def _number(value: Mapping[str, Any], key: str) -> float:
    raw = value.get(key)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise TypeError(f"S0 numeric field differs: {key}")
    return float(raw)


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _protocol() -> dict[str, object]:
    protocol: dict[str, object] = {
        "schema_version": ASPLOS27_TENX_BUDGET_SCHEMA_VERSION,
        "target_speedup": 10.0,
        "operator_target_speedup": 12.795107698179335,
        "minimum_semantic_coverage": 0.97,
        "maximum_semantic_unclassified": 0.03,
        "official_b0_scope": "fixed-16-iteration-prefix",
        "claim_modes": ["fixed_trajectory_systems", "solved_query_ttv"],
        "rules": [
            "critical_path_buckets_are_exclusive_and_sum_to_one",
            "phase_semantic_coverage_is_a_separate_admission_gate",
            "local_operator_and_standalone_graph_ratios_are_not_query_ratios",
            "distinct_scope_ratios_are_never_multiplied",
            "s0_diagnostic_never_sets_performance_claimed",
        ],
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def _source_payloads(root: Path) -> dict[str, Mapping[str, Any]]:
    missing = [
        str(path) for path in SOURCE_FILES.values() if not (root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"S0 source artifacts are missing: {missing}")
    return {
        name: _load_json(root / path)
        for name, path in SOURCE_FILES.items()
        if path.suffix == ".json"
    }


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"S0 JSONL row must be a mapping: {path}")
        rows.append(value)
    return rows


def _direct_ledger(
    source: Mapping[str, Mapping[str, Any]], root: Path
) -> dict[str, object]:
    b3 = source["fsg4_b3_summary"]
    mr5 = source["mr5_summary"]
    mr6 = source["mr6_summary"]
    cibc = source["cibc_summary"]
    b3_decision = b3["decision_inputs"]
    mr5_gates = mr5["gates"]
    mr6_gates = mr6["gates"]
    if not isinstance(b3_decision, Mapping):
        raise TypeError("B3 decision inputs must be a mapping")
    if not isinstance(mr5_gates, Mapping) or not isinstance(mr6_gates, Mapping):
        raise TypeError("MR timing gates must be mappings")
    observations = (
        DirectCumulativeObservation(
            observation_id="historical-b3-vs-b0-query-prefix",
            scope_id="resnet2b-prop0-same-solver-fixed-prefix",
            evidence_scope=EvidenceScope.FIXED_PREFIX,
            baseline_id="official-alpha-beta-crown-B0",
            candidate_id="BoundFlow-B3",
            baseline_over_candidate=float(b3_decision["b0_over_b3_query_geomean"]),
            source_digest=file_sha256(root / SOURCE_FILES["fsg4_b3_summary"]),
            semantic_passed=(
                b3.get("correctness_passed") is True
                and b3.get("measurement_auditable") is True
            ),
        ),
        DirectCumulativeObservation(
            observation_id="historical-mr5-three-site-production-bridge",
            scope_id="resnet2b-three-site-wrapper-inclusive-region",
            evidence_scope=EvidenceScope.SAME_SOLVER_REGION,
            baseline_id="native-provider",
            candidate_id="BoundFlow-MR5-bridge",
            baseline_over_candidate=float(mr5["host_speedup_geomean"]),
            source_digest=file_sha256(root / SOURCE_FILES["mr5_summary"]),
            semantic_passed=mr5_gates.get("correctness") is True,
        ),
        DirectCumulativeObservation(
            observation_id="historical-mr6-guard-reduced-bridge",
            scope_id="resnet2b-three-site-wrapper-inclusive-region",
            evidence_scope=EvidenceScope.SAME_SOLVER_REGION,
            baseline_id="native-provider",
            candidate_id="BoundFlow-MR6-guard-diagnostic",
            baseline_over_candidate=float(mr6["provider_diagnostic_host_geomean"]),
            source_digest=file_sha256(root / SOURCE_FILES["mr6_summary"]),
            semantic_passed=mr6_gates.get("semantic_exact") is True,
        ),
        DirectCumulativeObservation(
            observation_id="historical-cibc-standalone-ibp-graph",
            scope_id="resnet2b-standalone-ibp-cuda-graph",
            evidence_scope=EvidenceScope.STANDALONE_GRAPH,
            baseline_id="pytorch-ibp-cuda-graph",
            candidate_id="BoundFlow-CIBC-ibp-cuda-graph",
            baseline_over_candidate=float(cibc["model_speedup_geomean"]),
            source_digest=file_sha256(root / SOURCE_FILES["cibc_summary"]),
            semantic_passed=(
                cibc.get("sign_exact") is True
                and cibc.get("performance_admitted") is True
            ),
        ),
    )
    return validate_direct_observation_ledger(observations)


def _derived_outputs(root: Path) -> dict[str, object]:
    protocol = _protocol()
    source = _source_payloads(root)
    budget_report = derive_fsg1_diagnostic_budgets(
        source["fsg1_closure"],
        operator_target_speedup=_number(protocol, "operator_target_speedup"),
        target_speedup=_number(protocol, "target_speedup"),
    )
    ledger = _direct_ledger(source, root)
    transaction_inventory = derive_fsg1_transaction_inventory(
        _load_jsonl(root / SOURCE_FILES["fsg1_worker_runs"])
    )
    runs = budget_report["runs"]
    if not isinstance(runs, list):
        raise TypeError("S0 budget runs must be a list")
    coverages = [float(run["semantic_coverage_share"]) for run in runs]
    unclassified = [float(run["semantic_unclassified_share"]) for run in runs]
    projections = [float(run["projected_speedup"]) for run in runs]
    ceilings = [float(run["operator_infinite_speedup_ceiling"]) for run in runs]
    summary: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": budget_report["status"],
        "official_b0_scope": protocol["official_b0_scope"],
        "target_speedup": protocol["target_speedup"],
        "run_count": budget_report["run_count"],
        "admitted_run_count": budget_report["admitted_run_count"],
        "tenx_feasible_run_count": budget_report["tenx_feasible_run_count"],
        "minimum_semantic_coverage": min(coverages),
        "median_semantic_coverage": median(coverages),
        "maximum_semantic_unclassified": max(unclassified),
        "maximum_existing_operator_only_projection": max(projections),
        "maximum_operator_infinite_speedup_ceiling": max(ceilings),
        "direct_observation_count": ledger["observation_count"],
        "transaction_topology_context_closed_count": transaction_inventory[
            "topology_context_closed_count"
        ],
        "transaction_mechanism_admitted_count": transaction_inventory[
            "mechanism_admitted_count"
        ],
        "direct_ratios_aggregated": False,
        "s1_performance_gate_open": False,
        "required_next_measurement": (
            "classify_resnet_solver_phase_transactions_then_measure_direct_"
            "cumulative_candidate_against_the_same_B0_scope"
        ),
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    replay = {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "run_count": summary["run_count"],
        "admitted_run_count": summary["admitted_run_count"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }
    readme = (
        "# ASPLOS'27 S0 10x Budget Diagnostic\n\n"
        "This artifact derives a semantic-coverage admission gate and Amdahl budget "
        "from the frozen official B0 FSG1 prefix. It also records historical direct "
        "ratios as a non-aggregated ledger. It does not claim performance.\n"
    )
    return {
        "protocol.json": protocol,
        "budget_report.json": budget_report,
        "transaction_inventory.json": transaction_inventory,
        "direct_observation_ledger.json": ledger,
        "summary.json": summary,
        "replay_stdout.txt": canonical_json(replay) + "\n",
        "README.md": readme,
    }


def _output_text(name: str, value: object) -> str:
    if name.endswith(".json"):
        return canonical_json(value, indent=2) + "\n"
    if not isinstance(value, str):
        raise TypeError(f"S0 text output differs: {name}")
    return value


def generate_artifact(root: Path, artifact_dir: Path) -> Mapping[str, Any]:
    """Generate one derived artifact without rerunning the GPU workload."""

    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outputs = _derived_outputs(root)
    for name, value in outputs.items():
        (artifact_dir / name).write_text(_output_text(name, value), encoding="utf-8")
    summary = _load_json(artifact_dir / "summary.json")
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": summary["status"],
        "source_files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(root / path),
            }
            for name, path in sorted(SOURCE_FILES.items())
        },
        "code_revision": {str(path): file_sha256(root / path) for path in CODE_FILES},
        "files": {name: file_sha256(artifact_dir / name) for name in OUTPUT_FILES},
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact_dir / "manifest.json", manifest)
    return _load_json(artifact_dir / "replay_stdout.txt")


def replay_artifact(  # pylint: disable=too-many-branches
    root: Path, artifact_dir: Path
) -> Mapping[str, Any]:
    """Verify the envelope, source chain, and semantic derivation."""

    manifest = _load_json(artifact_dir / "manifest.json")
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
        or manifest.get("manifest_hash") != canonical_hash(semantic_manifest)
    ):
        raise ValueError("S0 artifact manifest envelope differs")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != set(SOURCE_FILES):
        raise ValueError("S0 source file inventory differs")
    for name, relative_path in SOURCE_FILES.items():
        record = source_files[name]
        if not isinstance(record, Mapping):
            raise TypeError("S0 source file record must be a mapping")
        if record.get("path") != str(relative_path):
            raise ValueError("S0 source path differs")
        if record.get("sha256") != file_sha256(root / relative_path):
            raise ValueError("S0 source digest differs")
    code_revision = manifest.get("code_revision")
    if not isinstance(code_revision, Mapping) or set(code_revision) != {
        str(path) for path in CODE_FILES
    }:
        raise ValueError("S0 code revision inventory differs")
    for path in CODE_FILES:
        if code_revision[str(path)] != file_sha256(root / path):
            raise ValueError("S0 code revision differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(OUTPUT_FILES):
        raise ValueError("S0 output file inventory differs")
    for name in OUTPUT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError("S0 output file digest differs")
    expected = _derived_outputs(root)
    for name, value in expected.items():
        if (artifact_dir / name).read_text(encoding="utf-8") != _output_text(
            name, value
        ):
            raise ValueError(f"S0 semantic replay differs: {name}")
    summary = _load_json(artifact_dir / "summary.json")
    if manifest.get("status") != summary.get("status") or manifest.get(
        "summary_hash"
    ) != summary.get("summary_hash"):
        raise ValueError("S0 manifest summary projection differs")
    return _load_json(artifact_dir / "replay_stdout.txt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "replay"))
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    """Run generation or semantic replay."""

    args = _parse_args()
    root = args.root.resolve()
    artifact_dir = (
        args.artifact_dir
        if args.artifact_dir.is_absolute()
        else (root / args.artifact_dir)
    ).resolve()
    if args.command == "generate":
        result = generate_artifact(root, artifact_dir)
    else:
        result = replay_artifact(root, artifact_dir)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
