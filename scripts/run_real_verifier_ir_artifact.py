#!/usr/bin/env python3
"""Generate or replay the self-contained real-verifier IR artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from boundflow.runtime.verifier_ir_integration import (
    ExternalVerifierCallSpec,
    compile_external_verifier_call,
)

ARTIFACT_SCHEMA = "boundflow.real-verifier-ir-artifact/v1"
ACTIVATION_PHASE = "activation_bab_bound"
ARTIFACT_FILES = (
    "activation_calls.jsonl",
    "online_execution.json",
    "resnet_semantics.json",
)


def _canonical(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _activation_row(workload: str, query: Mapping[str, Any]) -> dict[str, Any]:
    spec = ExternalVerifierCallSpec.from_query_dict(query)
    compilation = compile_external_verifier_call(spec)
    options = query.get("execution_options")
    limitations = []
    if isinstance(options, Mapping):
        raw_limitations = options.get("identity_limitations", [])
        if isinstance(raw_limitations, list):
            limitations = [str(value) for value in raw_limitations]
        if "bound_lower_requested" not in options:
            limitations.append(
                "legacy_requested_bound_polarity_unresolved_assumed_both"
            )
        if options.get("adapter_schema_version") == "boundflow.abcrown-adapter/v1":
            limitations.append("legacy_parent_lineage_not_captured")
    return {
        "source_workload": workload,
        "query": dict(query),
        "observed_method": spec.observed_method,
        "effective_method": spec.effective_method.value,
        "identity_limitations": limitations,
        "ir_hashes": compilation.hashes(),
        "backend": "external_abcrown_exact_call/v1",
        "semantics_owner": "external_verifier",
        "performance_claimed": False,
    }


def _historical_rows(query_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = sorted(query_root.glob("*/queries.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no workload queries.jsonl found under {query_root}")
    rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for path in paths:
        workload = path.parent.name
        sources[f"{workload}/queries.jsonl"] = _sha256(path)
        for query in _read_jsonl(path):
            options = query.get("execution_options")
            phase = (
                options.get("solver_phase") if isinstance(options, Mapping) else None
            )
            if phase == ACTIVATION_PHASE:
                rows.append(_activation_row(workload, query))
    return rows, sources


def _bab_projection(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    stats = result.get("stats")
    raw_rows = stats.get("bab", []) if isinstance(stats, Mapping) else []
    projected = []
    for row in raw_rows if isinstance(raw_rows, list) else []:
        if isinstance(row, list) and len(row) >= 3:
            projected.append(
                {
                    "instance": int(row[0]),
                    "final_lower": str(row[1]),
                    "visited_domains": int(row[2]),
                }
            )
    return projected


def _online_summary(  # pylint: disable=too-many-locals
    run_dir: Path,
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    queries_path = run_dir / "queries.jsonl"
    records_path = run_dir / "typed_ir.jsonl"
    manifest = _read_json(manifest_path)
    queries = _read_jsonl(queries_path)
    records = _read_jsonl(records_path)
    query_ids = [str(row.get("query_id")) for row in queries]
    record_ids = [str(row.get("query_id")) for row in records]
    if query_ids != record_ids or not all(
        row.get("completed") is True for row in records
    ):
        raise ValueError("online typed-IR query/result accounting is incomplete")
    typed = manifest.get("typed_ir")
    comparison = manifest.get("baseline_comparison")
    if not isinstance(typed, Mapping) or typed.get("enabled") is not True:
        raise ValueError("online run did not enable typed IR")
    if not isinstance(comparison, Mapping) or not (
        comparison.get("status_match") is True
        and comparison.get("visited_domains_match") is True
    ):
        raise ValueError("online observer on/off comparison did not pass")
    result = manifest.get("result")
    baseline = manifest.get("baseline_result")
    if not isinstance(result, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError("online run lacks result/baseline result")
    if _bab_projection(result) != _bab_projection(baseline):
        raise ValueError("online final lower/domain projection differs from baseline")
    phase_methods = Counter()
    effective_methods = Counter()
    requested_outputs = Counter()
    seen_query_ids: set[str] = set()
    parent_link_count = 0
    for query in queries:
        query_id = str(query.get("query_id"))
        parent_query_id = query.get("parent_query_id")
        if parent_query_id is not None:
            if str(parent_query_id) not in seen_query_ids:
                raise ValueError("online query parent does not precede its child")
            parent_link_count += 1
        seen_query_ids.add(query_id)
        options = query.get("execution_options")
        phase = str(options.get("solver_phase")) if isinstance(options, Mapping) else ""
        method = str(query.get("bound_method"))
        phase_methods[(phase, method)] += 1
        requested = query.get("requested_outputs")
        requested_key = "+".join(str(value) for value in requested or [])
        requested_outputs[requested_key] += 1
        if phase == ACTIVATION_PHASE:
            effective = ExternalVerifierCallSpec.from_query_dict(
                query
            ).effective_method.value
            effective_methods[effective] += 1
    return {
        "schema_version": str(manifest.get("schema_version")),
        "workload_name": str(manifest.get("workload_name")),
        "abcrown_commit": str(manifest.get("abcrown_commit")),
        "model_sha256": str(manifest.get("model_sha256")),
        "vnnlib_sha256": str(manifest.get("vnnlib_sha256")),
        "device": manifest.get("config_overrides", {}).get("general/device"),
        "query_count": len(queries),
        "compiled_and_dispatched": len(records),
        "completed": sum(row.get("completed") is True for row in records),
        "root_query_count": len(queries) - parent_link_count,
        "parent_link_count": parent_link_count,
        "requested_output_counts": dict(sorted(requested_outputs.items())),
        "phase_method_counts": [
            {"phase": key[0], "method": key[1], "count": count}
            for key, count in sorted(phase_methods.items())
        ],
        "activation_effective_method_counts": dict(sorted(effective_methods.items())),
        "result_status": result.get("status"),
        "baseline_status": baseline.get("status"),
        "bab_projection": _bab_projection(result),
        "observer_comparison": dict(comparison),
        "semantics_owner": "external_verifier",
        "performance_claimed": False,
        "source_digests": {
            "manifest.json": _sha256(manifest_path),
            "queries.jsonl": _sha256(queries_path),
            "typed_ir.jsonl": _sha256(records_path),
        },
    }


def _resnet_summary(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    capture = manifest.get("capture")
    boundflow = manifest.get("boundflow")
    eager = boundflow.get("pytorch_eager") if isinstance(boundflow, Mapping) else None
    comparison = eager.get("lower_vs_external") if isinstance(eager, Mapping) else None
    if not isinstance(capture, Mapping) or not isinstance(comparison, Mapping):
        raise ValueError("ResNet manifest lacks capture/correctness evidence")
    if not (
        manifest.get("status") == "ok"
        and capture.get("intermediate_bound_source") == "external_verifier"
        and capture.get("relu_lower_slope_policy") == "adaptive"
        and int(capture.get("intermediate_bound_count", 0)) == 6
        and comparison.get("allclose") is True
        and int(comparison.get("sign_agreement", 0))
        == int(comparison.get("sign_total", -1))
        and float(comparison.get("max_abs_diff", float("inf"))) <= 2e-4
    ):
        raise ValueError("ResNet RVIR-1 correctness gate did not pass")
    contract = manifest.get("benchmark_contract")
    return {
        "schema_version": str(manifest.get("schema_version")),
        "workload_name": str(manifest.get("workload_name")),
        "abcrown_commit": str(manifest.get("abcrown_commit")),
        "model_sha256": str(manifest.get("model_sha256")),
        "vnnlib_sha256": str(manifest.get("vnnlib_sha256")),
        "device": manifest.get("config_overrides", {}).get("general/device"),
        "intermediate_bound_count": int(capture["intermediate_bound_count"]),
        "intermediate_bound_source": capture["intermediate_bound_source"],
        "intermediate_bounds_hash": capture["intermediate_bounds_hash"],
        "relu_lower_slope_policy": capture["relu_lower_slope_policy"],
        "lower_allclose": comparison["allclose"],
        "lower_max_abs_diff": comparison["max_abs_diff"],
        "sign_agreement": comparison["sign_agreement"],
        "sign_total": comparison["sign_total"],
        "performance_compliant": (
            contract.get("performance_compliant")
            if isinstance(contract, Mapping)
            else False
        ),
        "performance_claimed": False,
        "source_manifest_sha256": _sha256(manifest_path),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical(value, indent=2) + "\n", encoding="utf-8")


def generate_artifact(
    out_dir: Path,
    *,
    query_root: Path,
    online_run_dir: Path,
    resnet_manifest: Path,
    expected_activation_calls: int,
) -> dict[str, Any]:
    """Freeze historical admission plus online and ResNet correctness evidence."""

    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"artifact output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, historical_sources = _historical_rows(query_root)
    if len(rows) != expected_activation_calls:
        raise ValueError(
            f"activation call count mismatch: {len(rows)} != {expected_activation_calls}"
        )
    (out_dir / "activation_calls.jsonl").write_text(
        "".join(_canonical(row) + "\n" for row in rows), encoding="utf-8"
    )
    _write_json(out_dir / "online_execution.json", _online_summary(online_run_dir))
    _write_json(out_dir / "resnet_semantics.json", _resnet_summary(resnet_manifest))
    workloads = Counter(str(row["source_workload"]) for row in rows)
    methods = Counter(str(row["effective_method"]) for row in rows)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA,
        "activation_call_count": len(rows),
        "workload_counts": dict(sorted(workloads.items())),
        "effective_method_counts": dict(sorted(methods.items())),
        "historical_source_digests": historical_sources,
        "files": {name: _sha256(out_dir / name) for name in ARTIFACT_FILES},
        "semantics_owner": "external_verifier",
        "performance_claimed": False,
        "environment_boundary": "CPU correctness; no CUDA or performance claim",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def replay_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Verify digests and independently recompile every embedded activation query."""

    manifest = _read_json(artifact_dir / "manifest.json")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("real-verifier IR artifact schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("real-verifier IR artifact file set mismatch")
    for name, digest in files.items():
        if _sha256(artifact_dir / str(name)) != digest:
            raise ValueError(f"real-verifier IR artifact digest mismatch: {name}")
    rows = _read_jsonl(artifact_dir / "activation_calls.jsonl")
    if len(rows) != int(manifest.get("activation_call_count", -1)):
        raise ValueError("real-verifier activation call count mismatch")
    for index, row in enumerate(rows):
        query = row.get("query")
        if not isinstance(query, Mapping):
            raise TypeError(f"activation row {index} has no query identity")
        expected = _activation_row(str(row.get("source_workload")), query)
        if row != expected:
            raise ValueError(f"real-verifier IR replay mismatch at row {index}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--out-dir", type=Path, required=True)
    generate.add_argument("--query-root", type=Path, required=True)
    generate.add_argument("--online-run-dir", type=Path, required=True)
    generate.add_argument("--resnet-manifest", type=Path, required=True)
    generate.add_argument("--expected-activation-calls", type=int, default=394)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate a fresh artifact or replay an existing self-contained artifact."""

    args = _parse_args()
    if args.command == "generate":
        manifest = generate_artifact(
            args.out_dir,
            query_root=args.query_root,
            online_run_dir=args.online_run_dir,
            resnet_manifest=args.resnet_manifest,
            expected_activation_calls=args.expected_activation_calls,
        )
        status = "generated"
    else:
        manifest = replay_artifact(args.artifact_dir)
        status = "replayed"
    print(
        _canonical(
            {
                "status": status,
                "activation_call_count": manifest["activation_call_count"],
                "performance_claimed": manifest["performance_claimed"],
            }
        )
    )


if __name__ == "__main__":
    main()
