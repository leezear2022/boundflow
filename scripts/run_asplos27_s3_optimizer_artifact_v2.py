#!/usr/bin/env python3
"""Generate or replay the 18-worker robust ASPLOS'27 S3 v2 artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,duplicate-code
# pylint: disable=too-many-boolean-expressions
# pylint: disable=protected-access

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

from scripts import run_asplos27_s3_optimizer_artifact as v1

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SCHEMA = "boundflow.asplos27-s3-optimizer-artifact/v2"
PROTOCOL_SCHEMA = "boundflow.asplos27-s3-optimizer-protocol/v2"
MANIFEST_SCHEMA = "boundflow.asplos27-s3-optimizer-manifest/v2"
ORDERS = v1.ORDERS
REPLICATES = 3
INTER_WORKER_COOLDOWN_SECONDS = 15
CODE_PATHS = (
    *v1.CODE_PATHS,
    "scripts/run_asplos27_s3_optimizer_artifact_v2.py",
    "scripts/probe_asplos27_s3_optimizer_v2_tamper.py",
    "tests/test_asplos27_s3_optimizer_artifact_v2.py",
)


def code_revision(revision: str) -> dict[str, str]:
    return {
        path: v1._git_blob_hash(revision, path) for path in CODE_PATHS
    }  # pylint: disable=protected-access


def protocol(source: Path, model: Path) -> dict[str, object]:
    revision = v1.git("rev-parse", "HEAD")
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "source_revision": revision,
        "code_revision": code_revision(revision),
        "source_capture_sha256": v1.file_sha256(source),
        "model_sha256": v1.file_sha256(model),
        "orders": list(ORDERS),
        "replicate_count_per_order": REPLICATES,
        "fresh_process_count": len(ORDERS) * REPLICATES,
        "inter_worker_cooldown_seconds": INTER_WORKER_COOLDOWN_SECONDS,
        "headline_estimator": "geomean-of-six-within-order-median-pair-speedups",
        "warmup_groups": 5,
        "sample_groups": 30,
        "lower_atol": 2e-4,
        "state_atol": 2e-5,
        "p_over_n_geomean_min": 3.00,
        "p_over_n_worst_min": 2.50,
        "p_over_d_geomean_min": 1.50,
        "p_over_n_reduced_min": 1.20,
        "p_over_n_nogo_worst": 0.98,
        "same_solver_claimed": False,
        "complete_query_claimed": False,
        "tenx_claimed": False,
        "performance_claimed": False,
    }


def validate_records(
    records: list[dict[str, Any]], protocol_value: dict[str, Any]
) -> dict[str, object]:
    expected = [
        (replicate, index, order)
        for replicate in range(REPLICATES)
        for index, order in enumerate(ORDERS)
    ]
    observed = [
        (row.get("replicate_index"), row.get("run_index"), row.get("order"))
        for row in records
    ]
    if observed != expected:
        raise ValueError("S3 v2 raw replicate/order inventory differs")

    replica_summaries = []
    for replicate in range(REPLICATES):
        group = [row for row in records if row.get("replicate_index") == replicate]
        replica_summaries.append(v1.validate_records(group, protocol_value))

    p_over_n_by_order: dict[str, list[float]] = {order: [] for order in ORDERS}
    p_over_d_by_order: dict[str, list[float]] = {order: [] for order in ORDERS}
    p_medians_by_order: dict[str, list[float]] = {order: [] for order in ORDERS}
    raw_p_over_n: list[float] = []
    raw_p_over_d: list[float] = []
    for row in records:
        medians = row["median_latency_ns"]
        order = str(row["order"])
        p_over_n = float(medians["N"]) / float(medians["P"])
        p_over_d = float(medians["D"]) / float(medians["P"])
        p_over_n_by_order[order].append(p_over_n)
        p_over_d_by_order[order].append(p_over_d)
        p_medians_by_order[order].append(float(medians["P"]))
        raw_p_over_n.append(p_over_n)
        raw_p_over_d.append(p_over_d)
    if any(
        len(values) != REPLICATES
        for values in (*p_over_n_by_order.values(), *p_over_d_by_order.values())
    ):
        raise ValueError("S3 v2 within-order replicate inventory differs")

    order_median_p_over_n = {
        order: statistics.median(values) for order, values in p_over_n_by_order.items()
    }
    order_median_p_over_d = {
        order: statistics.median(values) for order, values in p_over_d_by_order.items()
    }
    headline_n = list(order_median_p_over_n.values())
    headline_d = list(order_median_p_over_d.values())
    p_over_n_geomean = v1._geomean(headline_n)  # pylint: disable=protected-access
    p_over_n_worst = min(headline_n)
    p_over_d_geomean = v1._geomean(headline_d)  # pylint: disable=protected-access
    validates_3x = (
        p_over_n_geomean >= float(protocol_value["p_over_n_geomean_min"])
        and p_over_n_worst >= float(protocol_value["p_over_n_worst_min"])
        and p_over_d_geomean >= float(protocol_value["p_over_d_geomean_min"])
    )
    reduced = p_over_n_geomean >= float(
        protocol_value["p_over_n_reduced_min"]
    ) and p_over_n_worst >= float(protocol_value["p_over_n_nogo_worst"])
    status = (
        "validated-s3-3x-local-optimizer-v2"
        if validates_3x
        else (
            "validated-reduced-s3-local-optimizer-v2"
            if reduced
            else "validated-no-go-s3-local-optimizer-v2"
        )
    )
    max_differences = {
        field: max(
            float(summary["max_step_abs_diff"][field])  # type: ignore[index]
            for summary in replica_summaries
        )
        for field in v1.TENSOR_FIELDS
    }
    plan_hashes = {summary["plan_hash"] for summary in replica_summaries}
    trace_hashes = {summary["trace_hash"] for summary in replica_summaries}
    schedule_hashes = {summary["schedule_hash"] for summary in replica_summaries}
    if any(len(values) != 1 for values in (plan_hashes, trace_hashes, schedule_hashes)):
        raise ValueError("S3 v2 replica semantic identity differs")
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "status": status,
        "run_count": len(records),
        "replicate_count_per_order": REPLICATES,
        "orders": list(ORDERS),
        "plan_hash": next(iter(plan_hashes)),
        "trace_hash": next(iter(trace_hashes)),
        "schedule_hash": next(iter(schedule_hashes)),
        "p_over_n_geomean": p_over_n_geomean,
        "p_over_n_worst": p_over_n_worst,
        "p_over_d_geomean": p_over_d_geomean,
        "order_median_p_over_n": order_median_p_over_n,
        "order_median_p_over_d": order_median_p_over_d,
        "raw_p_over_n_by_order": p_over_n_by_order,
        "raw_p_over_d_by_order": p_over_d_by_order,
        "raw_p_over_n_geomean": v1._geomean(
            raw_p_over_n
        ),  # pylint: disable=protected-access
        "raw_p_over_n_worst": min(raw_p_over_n),
        "raw_p_over_d_geomean": v1._geomean(
            raw_p_over_d
        ),  # pylint: disable=protected-access
        "candidate_median_ns_by_order": p_medians_by_order,
        "max_step_abs_diff": max_differences,
        "lower_sign_exact": all(
            bool(summary["lower_sign_exact"]) for summary in replica_summaries
        ),
        "gradient_sign_exact": all(
            bool(summary["gradient_sign_exact"]) for summary in replica_summaries
        ),
        "candidate_peak_dynamic_allocated_bytes_max": max(
            int(
                summary["candidate_peak_dynamic_allocated_bytes_max"]
            )  # type: ignore[call-overload]
            for summary in replica_summaries
        ),
        "candidate_peak_dynamic_reserved_bytes_max": max(
            int(summary["candidate_peak_dynamic_reserved_bytes_max"])  # type: ignore[call-overload]
            for summary in replica_summaries
        ),
        "validated_s3_3x": validates_3x,
        "validated_reduced_s3": reduced,
        "same_solver_gate_open": validates_3x,
        "v1_no_go_preserved": True,
        "same_solver_claimed": False,
        "complete_query_claimed": False,
        "tenx_claimed": False,
        "performance_claimed": False,
    }


def _write_manifest(artifact: Path) -> dict[str, object]:
    files = {
        str(path.relative_to(artifact)): v1.file_sha256(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "files": files,
    }
    (artifact / "manifest.json").write_text(
        v1.canonical(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def replay(artifact: Path, *, require_validated_3x: bool = True) -> dict[str, object]:
    manifest = v1.load_json(artifact / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("S3 v2 manifest schema differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("S3 v2 manifest inventory differs")
    for name, digest in files.items():
        path = artifact / name
        if not path.is_file() or v1.file_sha256(path) != digest:
            raise ValueError(f"S3 v2 manifest digest differs: {name}")
    protocol_value = v1.load_json(artifact / "protocol.json")
    if (
        protocol_value.get("schema_version") != PROTOCOL_SCHEMA
        or protocol_value.get("fresh_process_count") != 18
        or protocol_value.get("replicate_count_per_order") != 3
        or protocol_value.get("inter_worker_cooldown_seconds") != 15
        or protocol_value.get("headline_estimator")
        != "geomean-of-six-within-order-median-pair-speedups"
        or protocol_value.get("p_over_n_geomean_min") != 3.0
        or protocol_value.get("p_over_n_worst_min") != 2.5
        or protocol_value.get("p_over_d_geomean_min") != 1.5
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
        raise ValueError("S3 v2 protocol differs")
    revision = protocol_value.get("source_revision")
    if not isinstance(revision, str) or protocol_value.get(
        "code_revision"
    ) != code_revision(revision):
        raise ValueError("S3 v2 code revision differs")
    summary = validate_records(
        v1.load_records(artifact / "raw/workers.jsonl"), protocol_value
    )
    if summary != v1.load_json(artifact / "summary.json"):
        raise ValueError("S3 v2 summary semantic replay differs")
    if require_validated_3x and not summary["validated_s3_3x"]:
        raise ValueError("S3 v2 formal 3x gate differs")
    return {
        "status": "replay-passed",
        "summary_hash": hashlib.sha256(v1.canonical(summary).encode()).hexdigest(),
        "validated_s3_3x": summary["validated_s3_3x"],
        "performance_claimed": False,
    }


def generate(artifact: Path, source_capture: Path, model: Path) -> dict[str, object]:
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError("S3 v2 artifact must be generated into an empty path")
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "raw").mkdir()
    (artifact / "logs").mkdir()
    protocol_value = protocol(source_capture, model)
    (artifact / "protocol.json").write_text(
        v1.canonical(protocol_value) + "\n", encoding="utf-8"
    )
    records: list[dict[str, Any]] = []
    for replicate in range(REPLICATES):
        for run_index, order in enumerate(ORDERS):
            with tempfile.TemporaryDirectory(
                prefix="boundflow-s3-v2-worker-"
            ) as directory:
                result = Path(directory) / "result.json"
                process = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/run_asplos27_s3_optimizer_worker.py"),
                        "--source-capture",
                        str(source_capture),
                        "--model",
                        str(model),
                        "--order",
                        order,
                        "--run-index",
                        str(run_index),
                        "--replicate-index",
                        str(replicate),
                        "--result",
                        str(result),
                    ],
                    cwd=ROOT,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                log_path = artifact / "logs" / f"r{replicate}-{run_index}-{order}.log"
                log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
                if process.returncode != 0:
                    raise RuntimeError(
                        "S3 v2 worker failed "
                        f"replicate={replicate} order={order} "
                        f"returncode={process.returncode} log={log_path}"
                    )
                records.append(v1.load_json(result))
                worker_ordinal = replicate * len(ORDERS) + run_index
                if worker_ordinal + 1 < len(ORDERS) * REPLICATES:
                    time.sleep(INTER_WORKER_COOLDOWN_SECONDS)
    (artifact / "raw/workers.jsonl").write_text(
        "".join(v1.canonical(row) + "\n" for row in records), encoding="utf-8"
    )
    summary = validate_records(records, protocol_value)
    (artifact / "summary.json").write_text(
        v1.canonical(summary) + "\n", encoding="utf-8"
    )
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
    print(v1.canonical(result), flush=True)


if __name__ == "__main__":
    main()
