#!/usr/bin/env python3
"""Generate or semantically replay the ASPLOS'27 S3 formal artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,duplicate-code
# pylint: disable=too-many-boolean-expressions,too-many-nested-blocks

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SCHEMA = "boundflow.asplos27-s3-optimizer-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.asplos27-s3-optimizer-protocol/v1"
MANIFEST_SCHEMA = "boundflow.asplos27-s3-optimizer-manifest/v1"
ORDERS = ("NDP", "NPD", "DNP", "DPN", "PND", "PDN")
CODE_PATHS = (
    "boundflow/backends/tvm/asplos27_s2_selected_value.py",
    "boundflow/runtime/asplos27_s2_crown_pipeline.py",
    "boundflow/runtime/asplos27_s3_optimizer_pipeline.py",
    "scripts/run_asplos27_s3_optimizer_worker.py",
    "scripts/run_asplos27_s3_optimizer_artifact.py",
    "scripts/probe_asplos27_s3_optimizer_tamper.py",
    "tests/test_asplos27_s3_optimizer_pipeline.py",
    "env.sh",
    "scripts/install_dev.sh",
)
TENSOR_FIELDS = (
    "alpha_before",
    "lower",
    "gradient",
    "alpha_after",
    "optimizer_exp_avg",
    "optimizer_exp_avg_sq",
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
        ["git", *arguments], cwd=ROOT, check=True, text=True, capture_output=True
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


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("S3 geomean input differs")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _sign(value: float) -> int:
    return int(value > 0.0) - int(value < 0.0)


def _receipt_hash_valid(receipt: dict[str, Any]) -> bool:
    claimed = receipt.get("receipt_hash")
    payload = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    return claimed == hashlib.sha256(canonical(payload).encode()).hexdigest()


def _tensor_hash(values: list[float], shape: list[int]) -> str:
    shape_tuple = tuple(shape)
    digest = hashlib.sha256()
    digest.update(b"torch.float32")
    digest.update(str(shape_tuple).encode())
    digest.update(struct.pack(f"<{len(values)}f", *values))
    return digest.hexdigest()


def _validate_tensor(payload: object, shape: tuple[int, ...]) -> list[float]:
    if not isinstance(payload, dict):
        raise TypeError("S3 step tensor payload differs")
    values = payload.get("values")
    raw_shape = payload.get("shape")
    if (
        not isinstance(values, list)
        or raw_shape != list(shape)
        or len(values) != math.prod(shape)
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        )
        or payload.get("sha256") != _tensor_hash(values, raw_shape)
    ):
        raise ValueError("S3 step tensor digest differs")
    return [float(value) for value in values]


def validate_records(
    records: list[dict[str, Any]], protocol_value: dict[str, Any]
) -> dict[str, object]:
    if (
        len(records) != 6
        or [row.get("order") for row in records] != list(ORDERS)
        or [row.get("run_index") for row in records] != list(range(6))
    ):
        raise ValueError("S3 raw process/order inventory differs")
    identity_fields = ("plan_hash", "trace_hash", "schedule_hash")
    identities = [
        {canonical(row.get(name)) for row in records} for name in identity_fields
    ]
    environments = {canonical(row.get("environment")) for row in records}
    if any(len(values) != 1 for values in identities) or len(environments) != 1:
        raise ValueError("S3 raw identity/environment differs")

    n_over_p: list[float] = []
    d_over_p: list[float] = []
    max_differences = {field: 0.0 for field in TENSOR_FIELDS}
    lower_sign_exact = True
    gradient_sign_exact = True
    dynamic_allocated: list[int] = []
    dynamic_reserved: list[int] = []
    prepare_ns: list[float] = []
    for row in records:
        if (
            row.get("schema_version") != "boundflow.asplos27-s3-optimizer-worker/v1"
            or row.get("source_capture_sha256")
            != protocol_value.get("source_capture_sha256")
            or row.get("model_sha256") != protocol_value.get("model_sha256")
            or row.get("warmup_groups") != 5
            or row.get("sample_groups") != 30
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
            raise ValueError("S3 raw protocol/claim boundary differs")
        latency = row.get("latency_ns")
        medians = row.get("median_latency_ns")
        if not isinstance(latency, dict) or not isinstance(medians, dict):
            raise TypeError("S3 latency payload differs")
        for mode in "NDP":
            samples = latency.get(mode)
            if (
                not isinstance(samples, list)
                or len(samples) != 30
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in samples
                )
                or medians.get(mode) != statistics.median(samples)
            ):
                raise ValueError("S3 latency samples differ")
        n_over_p.append(float(medians["N"]) / float(medians["P"]))
        d_over_p.append(float(medians["D"]) / float(medians["P"]))

        semantic = row.get("semantic")
        if not isinstance(semantic, dict):
            raise TypeError("S3 semantic payload differs")
        mode_steps: dict[str, list[dict[str, Any]]] = {}
        for mode in "NDP":
            steps = semantic.get(mode)
            if not isinstance(steps, list) or len(steps) != 10:
                raise ValueError("S3 semantic step inventory differs")
            mode_steps[mode] = steps
            for ordinal, step in enumerate(steps):
                if (
                    not isinstance(step, dict)
                    or step.get("evaluation_ordinal") != ordinal
                    or step.get("update_after") is not (ordinal < 9)
                    or step.get("optimizer_step") != float(min(ordinal + 1, 9))
                    or not math.isclose(
                        float(step.get("alpha_learning_rate", -1.0)),
                        0.01 * (0.98**ordinal),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ):
                    raise ValueError("S3 optimizer transition metadata differs")
                for field in TENSOR_FIELDS:
                    shape = (6, 1) if field == "lower" else (2, 1, 6, 86)
                    _validate_tensor(step.get(field), shape)
        for mode in "DP":
            for expected_step, observed_step in zip(mode_steps["N"], mode_steps[mode]):
                for field in TENSOR_FIELDS:
                    shape = (6, 1) if field == "lower" else (2, 1, 6, 86)
                    expected = _validate_tensor(expected_step[field], shape)
                    observed = _validate_tensor(observed_step[field], shape)
                    maximum = max(
                        (abs(left - right) for left, right in zip(expected, observed)),
                        default=0.0,
                    )
                    tolerance = (
                        float(protocol_value["lower_atol"])
                        if field == "lower"
                        else float(protocol_value["state_atol"])
                    )
                    if maximum > tolerance:
                        raise ValueError(f"S3 semantic tolerance differs: {field}")
                    max_differences[field] = max(max_differences[field], maximum)
                    if field in {"lower", "gradient"}:
                        exact = all(
                            _sign(left) == _sign(right)
                            for left, right in zip(expected, observed)
                        )
                        if not exact:
                            raise ValueError(f"S3 semantic sign differs: {field}")
                        if field == "lower":
                            lower_sign_exact = lower_sign_exact and exact
                        else:
                            gradient_sign_exact = gradient_sign_exact and exact

        receipt = row.get("candidate_receipt")
        direct_receipt = row.get("direct_receipt")
        if (
            not isinstance(receipt, dict)
            or not _receipt_hash_valid(receipt)
            or receipt.get("production_plan_hash") != row.get("plan_hash")
            or receipt.get("trace_hash") != row.get("trace_hash")
            or receipt.get("optimizer_schedule_hash") != row.get("schedule_hash")
            or receipt.get("evaluation_count") != 10
            or receipt.get("optimizer_mutation_count") != 9
            or receipt.get("scheduler_mutation_count") != 9
            or receipt.get("custom_forward_count") != 10
            or receipt.get("custom_backward_count") != 10
            or receipt.get("forward_graph_replay_count") != 10
            or receipt.get("selected_graph_replay_count") != 10
            or receipt.get("host_policy_cut_count") != 10
            or receipt.get("autograd_function_count") != 0
            or receipt.get("executor_registry_count") != 0
            or receipt.get("fallback_count") != 0
            or receipt.get("eager_candidate_count") != 0
            or receipt.get("native_shadow_count") != 0
            or receipt.get("saved_dense_a_count") != 0
            or receipt.get("saved_autograd_history") is not False
            or receipt.get("performance_claimed") is not False
            or not isinstance(direct_receipt, dict)
            or direct_receipt.get("performance_claimed") is not False
        ):
            raise ValueError("S3 execution receipt semantics differ")
        memory = row.get("memory")
        prepare = row.get("prepare_ns")
        if not isinstance(memory, dict) or not isinstance(prepare, dict):
            raise TypeError("S3 memory/prepare payload differs")
        allocated = memory.get("candidate_peak_dynamic_allocated")
        reserved = memory.get("candidate_peak_dynamic_reserved")
        if (
            not isinstance(allocated, int)
            or allocated < 0
            or not isinstance(reserved, int)
            or reserved < 0
        ):
            raise ValueError("S3 dynamic memory payload differs")
        dynamic_allocated.append(allocated)
        dynamic_reserved.append(reserved)
        prepare_ns.append(float(prepare["P"]))

    p_over_n_geomean = _geomean(n_over_p)
    p_over_n_worst = min(n_over_p)
    p_over_d_geomean = _geomean(d_over_p)
    validates_3x = (
        p_over_n_geomean >= float(protocol_value["p_over_n_geomean_min"])
        and p_over_n_worst >= float(protocol_value["p_over_n_worst_min"])
        and p_over_d_geomean >= float(protocol_value["p_over_d_geomean_min"])
    )
    reduced = p_over_n_geomean >= float(
        protocol_value["p_over_n_reduced_min"]
    ) and p_over_n_worst >= float(protocol_value["p_over_n_nogo_worst"])
    status = (
        "validated-s3-3x-local-optimizer"
        if validates_3x
        else (
            "validated-reduced-s3-local-optimizer"
            if reduced
            else "validated-no-go-s3-local-optimizer"
        )
    )
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "status": status,
        "run_count": len(records),
        "orders": list(ORDERS),
        "plan_hash": json.loads(next(iter(identities[0]))),
        "trace_hash": json.loads(next(iter(identities[1]))),
        "schedule_hash": json.loads(next(iter(identities[2]))),
        "p_over_n_geomean": p_over_n_geomean,
        "p_over_n_worst": p_over_n_worst,
        "p_over_d_geomean": p_over_d_geomean,
        "pair_speedups_p_over_n": n_over_p,
        "pair_speedups_p_over_d": d_over_p,
        "max_step_abs_diff": max_differences,
        "lower_sign_exact": lower_sign_exact,
        "gradient_sign_exact": gradient_sign_exact,
        "candidate_prepare_ns_geomean": _geomean(prepare_ns),
        "candidate_peak_dynamic_allocated_bytes_max": max(dynamic_allocated),
        "candidate_peak_dynamic_reserved_bytes_max": max(dynamic_reserved),
        "validated_s3_3x": validates_3x,
        "validated_reduced_s3": reduced,
        "same_solver_gate_open": validates_3x,
        "same_solver_claimed": False,
        "complete_query_claimed": False,
        "tenx_claimed": False,
        "performance_claimed": False,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"S3 JSON object differs: {path}")
    return value


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("S3 JSONL row differs")
    return rows


def _write_manifest(artifact: Path) -> dict[str, object]:
    files = {
        str(path.relative_to(artifact)): file_sha256(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "files": files,
    }
    (artifact / "manifest.json").write_text(
        canonical(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def replay(artifact: Path, *, require_validated_3x: bool = True) -> dict[str, object]:
    manifest = load_json(artifact / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("S3 manifest schema differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("S3 manifest inventory differs")
    for name, digest in files.items():
        path = artifact / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"S3 manifest digest differs: {name}")
    protocol_value = load_json(artifact / "protocol.json")
    if (
        protocol_value.get("schema_version") != PROTOCOL_SCHEMA
        or protocol_value.get("orders") != list(ORDERS)
        or protocol_value.get("fresh_process_count") != 6
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
        raise ValueError("S3 protocol differs")
    revision = protocol_value.get("source_revision")
    if not isinstance(revision, str) or protocol_value.get(
        "code_revision"
    ) != code_revision(revision):
        raise ValueError("S3 code revision differs")
    summary = validate_records(
        load_records(artifact / "raw/workers.jsonl"), protocol_value
    )
    if summary != load_json(artifact / "summary.json"):
        raise ValueError("S3 summary semantic replay differs")
    if require_validated_3x and not summary["validated_s3_3x"]:
        raise ValueError("S3 formal 3x gate differs")
    return {
        "status": "replay-passed",
        "summary_hash": hashlib.sha256(canonical(summary).encode()).hexdigest(),
        "validated_s3_3x": summary["validated_s3_3x"],
        "performance_claimed": False,
    }


def generate(artifact: Path, source_capture: Path, model: Path) -> dict[str, object]:
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(
            "S3 artifact must be generated raw-first into an empty path"
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
        with tempfile.TemporaryDirectory(prefix="boundflow-s3-worker-") as directory:
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
