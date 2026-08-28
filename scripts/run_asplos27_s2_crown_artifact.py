#!/usr/bin/env python3
"""Generate or semantically replay the ASPLOS'27 S2 formal artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SCHEMA = "boundflow.asplos27-s2-crown-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.asplos27-s2-crown-protocol/v1"
MANIFEST_SCHEMA = "boundflow.asplos27-s2-crown-manifest/v1"
ORDERS = ("NDP", "NPD", "DNP", "DPN", "PND", "PDN")
CODE_PATHS = (
    "boundflow/backends/tvm/asplos27_s2_selected_value.py",
    "boundflow/runtime/asplos27_s2_crown_pipeline.py",
    "scripts/run_asplos27_s2_crown_worker.py",
    "scripts/run_asplos27_s2_crown_artifact.py",
    "tests/test_asplos27_s2_crown_pipeline.py",
    "env.sh",
    "scripts/install_dev.sh",
)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _git_blob_hash(revision: str, path: str) -> str:
    blob = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def code_revision(revision: str) -> dict[str, str]:
    return {path: _git_blob_hash(revision, path) for path in CODE_PATHS}


def protocol(source: Path, model: Path) -> dict[str, object]:
    revision = git("rev-parse", "HEAD")
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "source_revision": revision,
        "code_revision": code_revision(revision),
        "source_capture_sha256": file_sha256(source),
        "model_sha256": file_sha256(model),
        "orders": list(ORDERS),
        "fresh_process_count": 6,
        "warmup_groups": 5,
        "sample_groups": 30,
        "lower_atol": 2e-4,
        "gradient_atol": 2e-4,
        "p_over_d_geomean_min": 0.90,
        "p_over_n_geomean_min": 4.00,
        "p_over_n_worst_min": 3.50,
        "p_over_n_reduced_min": 1.20,
        "p_over_n_nogo_worst": 0.98,
        "same_solver_claimed": False,
        "complete_query_claimed": False,
        "tenx_claimed": False,
        "performance_claimed": False,
    }


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("S2 geomean input differs")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _sign(value: float) -> int:
    return int(value > 0.0) - int(value < 0.0)


def _receipt_hash_valid(receipt: dict[str, Any]) -> bool:
    claimed = receipt.get("receipt_hash")
    payload = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    return claimed == hashlib.sha256(canonical(payload).encode()).hexdigest()


def validate_records(
    records: list[dict[str, Any]], protocol_value: dict[str, Any]
) -> dict[str, object]:
    if (
        len(records) != 6
        or [row.get("order") for row in records] != list(ORDERS)
        or [row.get("run_index") for row in records] != list(range(6))
    ):
        raise ValueError("S2 raw process/order inventory differs")
    plan_hashes = {row.get("plan_hash") for row in records}
    trace_hashes = {row.get("trace_hash") for row in records}
    environments = {canonical(row.get("environment")) for row in records}
    if len(plan_hashes) != 1 or len(trace_hashes) != 1 or len(environments) != 1:
        raise ValueError("S2 raw identity/environment differs")
    max_lower_diff = 0.0
    max_gradient_diff = 0.0
    lower_sign_exact = True
    gradient_sign_exact = True
    n_over_p: list[float] = []
    d_over_p: list[float] = []
    cold_prepare_p: list[float] = []
    for row in records:
        if (
            row.get("schema_version") != "boundflow.asplos27-s2-crown-worker/v1"
            or row.get("source_capture_sha256")
            != protocol_value.get("source_capture_sha256")
            or row.get("model_sha256") != protocol_value.get("model_sha256")
            or row.get("warmup_groups") != protocol_value.get("warmup_groups")
            or row.get("sample_groups") != protocol_value.get("sample_groups")
            or any(
                row.get(flag) is not False
                for flag in (
                    "same_solver_claimed",
                    "complete_query_claimed",
                    "tenx_claimed",
                    "performance_claimed",
                )
            )
        ):
            raise ValueError("S2 raw protocol/claim boundary differs")
        latency = row.get("latency_ns")
        medians = row.get("median_latency_ns")
        if not isinstance(latency, dict) or not isinstance(medians, dict):
            raise ValueError("S2 raw latency payload differs")
        for mode in "NDP":
            values = latency.get(mode)
            if (
                not isinstance(values, list)
                or len(values) != 30
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in values
                )
                or medians.get(mode) != statistics.median(values)
            ):
                raise ValueError("S2 raw latency samples differ")
        n_over_p.append(float(medians["N"]) / float(medians["P"]))
        d_over_p.append(float(medians["D"]) / float(medians["P"]))
        cold_prepare_p.append(float(row["prepare_ns"]["P"]))
        semantic = row.get("semantic")
        if not isinstance(semantic, dict):
            raise ValueError("S2 semantic payload differs")
        native = semantic.get("N")
        if not isinstance(native, dict):
            raise ValueError("S2 native semantic payload differs")
        for mode in "DP":
            candidate = semantic.get(mode)
            if not isinstance(candidate, dict):
                raise ValueError("S2 candidate semantic payload differs")
            for field, expected_length, tolerance in (
                ("lower", 6, float(protocol_value["lower_atol"])),
                ("gradient", 1032, float(protocol_value["gradient_atol"])),
            ):
                expected = native.get(field)
                observed = candidate.get(field)
                if (
                    not isinstance(expected, list)
                    or not isinstance(observed, list)
                    or len(expected) != expected_length
                    or len(observed) != expected_length
                    or any(not isinstance(value, (int, float)) for value in observed)
                ):
                    raise ValueError("S2 semantic vector differs")
                differences = [
                    abs(float(left) - float(right))
                    for left, right in zip(expected, observed)
                ]
                maximum = max(differences, default=0.0)
                if maximum > tolerance:
                    raise ValueError("S2 semantic tolerance differs")
                signs = all(
                    _sign(float(left)) == _sign(float(right))
                    for left, right in zip(expected, observed)
                )
                if not signs:
                    raise ValueError("S2 semantic sign differs")
                if field == "lower":
                    max_lower_diff = max(max_lower_diff, maximum)
                    lower_sign_exact = lower_sign_exact and signs
                else:
                    max_gradient_diff = max(max_gradient_diff, maximum)
                    gradient_sign_exact = gradient_sign_exact and signs
        receipt = row.get("canonical_receipt")
        direct_receipt = row.get("direct_receipt")
        if (
            not isinstance(receipt, dict)
            or not _receipt_hash_valid(receipt)
            or receipt.get("production_plan_hash") != row.get("plan_hash")
            or receipt.get("trace_hash") != row.get("trace_hash")
            or receipt.get("cudnn_partition_function_count") != 4
            or receipt.get("cudnn_conv_call_count") != 5
            or receipt.get("selected_tir_count") != 4
            or receipt.get("forward_graph_replay_count") != 1
            or receipt.get("selected_graph_replay_count") != 1
            or receipt.get("custom_forward_count") != 1
            or receipt.get("custom_backward_count") != 1
            or receipt.get("existing_arena_count") != 2
            or receipt.get("active_beta") is not True
            or receipt.get("saved_dense_a_count") != 0
            or receipt.get("saved_autograd_history") is not False
            or receipt.get("warm_dlpack_view_count") != 0
            or receipt.get("fallback_count") != 0
            or receipt.get("eager_candidate_count") != 0
            or receipt.get("native_shadow_count") != 0
            or receipt.get("performance_claimed") is not False
            or not isinstance(direct_receipt, dict)
            or direct_receipt.get("performance_claimed") is not False
        ):
            raise ValueError("S2 execution receipt semantics differ")
        memory = row.get("memory")
        if (
            not isinstance(memory, dict)
            or memory.get("canonical_peak_dynamic_allocated") != 0
            or memory.get("canonical_peak_dynamic_reserved") != 0
        ):
            raise ValueError("S2 warm memory ownership differs")
    p_over_n_geomean = _geomean(n_over_p)
    p_over_d_geomean = _geomean(d_over_p)
    p_over_n_worst = min(n_over_p)
    qualifies_4x = (
        p_over_d_geomean >= float(protocol_value["p_over_d_geomean_min"])
        and p_over_n_geomean >= float(protocol_value["p_over_n_geomean_min"])
        and p_over_n_worst >= float(protocol_value["p_over_n_worst_min"])
    )
    reduced = p_over_n_geomean >= float(
        protocol_value["p_over_n_reduced_min"]
    ) and p_over_n_worst >= float(protocol_value["p_over_n_nogo_worst"])
    if qualifies_4x:
        status = "validated-s2-4x-canonical-crown"
    elif reduced:
        status = "validated-reduced-s2-canonical-crown"
    else:
        status = "validated-no-go-s2-canonical-crown"
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "status": status,
        "run_count": len(records),
        "orders": list(ORDERS),
        "plan_hash": next(iter(plan_hashes)),
        "trace_hash": next(iter(trace_hashes)),
        "p_over_n_by_run": n_over_p,
        "p_over_d_by_run": d_over_p,
        "p_over_n_geomean": p_over_n_geomean,
        "p_over_n_worst": p_over_n_worst,
        "p_over_d_geomean": p_over_d_geomean,
        "max_lower_abs_diff": max_lower_diff,
        "max_gradient_abs_diff": max_gradient_diff,
        "lower_sign_exact": lower_sign_exact,
        "gradient_sign_exact": gradient_sign_exact,
        "canonical_prepare_ns_geomean": _geomean(cold_prepare_p),
        "warm_dynamic_allocated_bytes": 0,
        "warm_dynamic_reserved_bytes": 0,
        "validated_s2_4x": qualifies_4x,
        "validated_reduced_s2": reduced,
        "optimizer_gate_open": p_over_n_worst >= 0.98,
        "same_solver_claimed": False,
        "complete_query_claimed": False,
        "tenx_claimed": False,
        "performance_claimed": False,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"S2 JSON object differs: {path}")
    return value


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("S2 JSONL row differs")
    return rows


def _write_manifest(artifact: Path) -> dict[str, object]:
    files = {
        str(path.relative_to(artifact)): file_sha256(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {"schema_version": MANIFEST_SCHEMA, "files": files}
    (artifact / "manifest.json").write_text(
        canonical(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def replay(artifact: Path, *, require_validated_4x: bool = True) -> dict[str, object]:
    manifest = load_json(artifact / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("S2 manifest schema differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("S2 manifest inventory differs")
    for name, digest in files.items():
        path = artifact / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"S2 manifest digest differs: {name}")
    protocol_value = load_json(artifact / "protocol.json")
    if (
        protocol_value.get("schema_version") != PROTOCOL_SCHEMA
        or protocol_value.get("orders") != list(ORDERS)
        or protocol_value.get("fresh_process_count") != 6
        or any(
            protocol_value.get(flag) is not False
            for flag in (
                "same_solver_claimed",
                "complete_query_claimed",
                "tenx_claimed",
                "performance_claimed",
            )
        )
    ):
        raise ValueError("S2 protocol differs")
    revision = protocol_value.get("source_revision")
    if not isinstance(revision, str) or protocol_value.get(
        "code_revision"
    ) != code_revision(revision):
        raise ValueError("S2 code revision differs")
    summary = validate_records(
        load_records(artifact / "raw/workers.jsonl"), protocol_value
    )
    if summary != load_json(artifact / "summary.json"):
        raise ValueError("S2 summary semantic replay differs")
    if require_validated_4x and not summary["validated_s2_4x"]:
        raise ValueError("S2 formal 4x gate differs")
    return {
        "status": "replay-passed",
        "summary_hash": hashlib.sha256(canonical(summary).encode()).hexdigest(),
        "validated_s2_4x": summary["validated_s2_4x"],
        "performance_claimed": False,
    }


def generate(artifact: Path, source_capture: Path, model: Path) -> dict[str, object]:
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(
            "S2 artifact must be generated raw-first into an empty path"
        )
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "raw").mkdir()
    (artifact / "logs").mkdir()
    protocol_value = protocol(source_capture, model)
    (artifact / "protocol.json").write_text(
        canonical(protocol_value) + "\n", encoding="utf-8"
    )
    records: list[dict[str, Any]] = []
    for run_index, order in enumerate(ORDERS):
        with tempfile.TemporaryDirectory(prefix="boundflow-s2-worker-") as directory:
            result = Path(directory) / "result.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_asplos27_s2_crown_worker.py"),
                    "--source-capture",
                    str(source_capture),
                    "--model",
                    str(model),
                    "--order",
                    order,
                    "--run-index",
                    str(run_index),
                    "--result",
                    str(result),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            (artifact / "logs" / f"{run_index}-{order}.log").write_text(
                process.stdout + process.stderr, encoding="utf-8"
            )
            records.append(load_json(result))
    (artifact / "raw/workers.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in records), encoding="utf-8"
    )
    summary = validate_records(records, protocol_value)
    (artifact / "summary.json").write_text(canonical(summary) + "\n", encoding="utf-8")
    _write_manifest(artifact)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        result = replay(args.artifact.resolve())
    else:
        if args.source_capture is None or args.model is None:
            parser.error("generation requires --source-capture and --model")
        result = generate(
            args.artifact.resolve(),
            args.source_capture.resolve(),
            args.model.resolve(),
        )
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
