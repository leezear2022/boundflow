#!/usr/bin/env python3
"""Package or replay five paired root-terminal TIR measurements."""

# pylint: disable=too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

SCHEMA_VERSION = "boundflow.root-crown-terminal-five-pair/v1"
LOWER_ATOL = 3.0e-6
PAIR_COUNT = 5
CODE_PATHS = (
    "boundflow/backends/tvm/root_crown_terminal_linear.py",
    "boundflow/runtime/root_crown_terminal_capture.py",
    "boundflow/runtime/root_crown_terminal_live.py",
    "boundflow/runtime/root_crown_terminal_tir.py",
    "scripts/package_root_crown_terminal_five_pair.py",
    "scripts/probe_root_crown_terminal_capture.py",
    "scripts/probe_root_crown_terminal_tir.py",
    "scripts/run_root_crown_terminal_live_worker.py",
    "tests/test_root_crown_terminal_five_pair_artifact.py",
    "tests/test_root_crown_terminal_tir.py",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _geomean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("five-pair ratio differs")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"five-pair {name} differs")
    return cast(Mapping[str, Any], value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"five-pair {name} differs")
    return value


def _project_worker(raw: Mapping[str, Any]) -> dict[str, object]:
    run = _as_mapping(raw.get("run"), "run")
    diagnostics = _as_mapping(raw.get("diagnostics"), "diagnostics")
    root_timings = _as_mapping(
        diagnostics.get("root_incomplete_timings"), "root timings"
    )
    projection: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "pair_index": int(run["block_index"]),
        "sequence_position": int(run["sequence_position"]),
        "mode": str(raw["root_terminal_mode"]),
        "run_id": str(run["run_id"]),
        "configuration": str(run["configuration"]),
        "metrics": run["metrics"],
        "semantics": run["semantics"],
        "execution": run["execution"],
        "protocol_identity": str(run["protocol_identity"]),
        "source_identity": str(run["source_identity"]),
        "activation": raw["activation"],
        "s4_exact_call_receipts": raw["s4_exact_call_receipts"],
        "root_timing_aggregates": root_timings["aggregates"],
        "root_terminal_compile_ns": int(raw["root_terminal_compile_ns"]),
        "root_terminal_compile_excluded_from_query": bool(
            raw["root_terminal_compile_excluded_from_query"]
        ),
        "root_terminal_receipt": raw["root_terminal_receipt"],
        "performance_claimed": bool(raw["performance_claimed"]),
    }
    if b"/home/" in _canonical_bytes(projection) or b"/tmp/" in _canonical_bytes(
        projection
    ):
        raise ValueError("five-pair projection leaks a local path")
    projection["worker_hash"] = _sha256_bytes(_canonical_bytes(projection))
    return projection


def _read_raw_workers(raw_dir: Path) -> list[dict[str, object]]:
    raw_paths = sorted(raw_dir.glob("*.json"))
    workers = []
    for path in raw_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "root_terminal_mode" in raw:
            workers.append(_project_worker(cast(Mapping[str, Any], raw)))
    if len(workers) != PAIR_COUNT * 2:
        raise ValueError("five-pair worker count differs")
    workers.sort(
        key=lambda worker: (
            _integer(worker["pair_index"], "pair index"),
            _integer(worker["sequence_position"], "sequence position"),
        )
    )
    return workers


def _read_artifact_workers(artifact: Path) -> list[dict[str, object]]:
    workers = []
    for line in (artifact / "workers.jsonl").read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("five-pair worker row differs")
        worker = cast(dict[str, object], value)
        claimed = worker.pop("worker_hash", None)
        actual = _sha256_bytes(_canonical_bytes(worker))
        worker["worker_hash"] = claimed
        if claimed != actual:
            raise ValueError("five-pair worker hash differs")
        workers.append(worker)
    if len(workers) != PAIR_COUNT * 2:
        raise ValueError("five-pair worker count differs")
    return workers


def _metric(worker: Mapping[str, object], name: str) -> int:
    if name == "root_incomplete_wall_ns":
        aggregates = _as_mapping(worker["root_timing_aggregates"], "aggregates")
        root = _as_mapping(aggregates.get("root_incomplete"), "root aggregate")
        return int(root["inclusive_ns"])
    if name == "optimized_bounds_transaction_wall_ns":
        aggregates = _as_mapping(worker["root_timing_aggregates"], "aggregates")
        optimized = _as_mapping(
            aggregates.get("optimized_bounds_transaction"), "optimized aggregate"
        )
        return int(optimized["inclusive_ns"])
    metrics = _as_mapping(worker["metrics"], "metrics")
    return int(metrics[name])


def _validate_candidate_receipt(worker: Mapping[str, object]) -> None:
    receipt = _as_mapping(worker.get("root_terminal_receipt"), "candidate receipt")
    expected = {
        "outer_call_count": 1,
        "relu_replacement_count": 5,
        "linear_replacement_count": 5,
        "forward_launch_count": 5,
        "backward_launch_count": 4,
        "fallback_count": 0,
        "performance_claimed": False,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"five-pair candidate receipt differs: {name}")
    if receipt.get("dlpack_pointer_count") != receipt.get("dlpack_pointer_exact_count"):
        raise ValueError("five-pair DLPack pointer receipt differs")


def _semantics_diff(
    control: Mapping[str, object], candidate: Mapping[str, object]
) -> float:
    control_semantics = dict(_as_mapping(control["semantics"], "control semantics"))
    candidate_semantics = dict(
        _as_mapping(candidate["semantics"], "candidate semantics")
    )
    control_lower = [float(value) for value in control_semantics.pop("lower_values")]
    candidate_lower = [
        float(value) for value in candidate_semantics.pop("lower_values")
    ]
    if control_semantics != candidate_semantics or len(control_lower) != len(
        candidate_lower
    ):
        raise ValueError("five-pair discrete semantics differ")
    maximum = max(
        abs(left - right)
        for left, right in zip(control_lower, candidate_lower, strict=True)
    )
    if maximum > LOWER_ATOL:
        raise ValueError("five-pair lower tolerance differs")
    return maximum


def _derive_summary(workers: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_pair: dict[int, dict[str, Mapping[str, object]]] = {}
    expected_identities: tuple[str, str] | None = None
    candidate_hashes: dict[str, set[str]] = {
        "template_hash": set(),
        "unscheduled_tir_hash": set(),
        "scheduled_tir_hash": set(),
        "device_source_hash": set(),
    }
    for worker in workers:
        pair_index = _integer(worker["pair_index"], "pair index")
        mode = str(worker["mode"])
        if pair_index not in range(PAIR_COUNT) or mode not in {"control", "candidate"}:
            raise ValueError("five-pair identity differs")
        if bool(worker["performance_claimed"]):
            raise ValueError("five-pair claim boundary differs")
        if not bool(worker["root_terminal_compile_excluded_from_query"]):
            raise ValueError("five-pair compile boundary differs")
        identity = (str(worker["protocol_identity"]), str(worker["source_identity"]))
        if expected_identities is None:
            expected_identities = identity
        elif identity != expected_identities:
            raise ValueError("five-pair protocol/source identity differs")
        pair = by_pair.setdefault(pair_index, {})
        if mode in pair:
            raise ValueError("five-pair duplicate worker differs")
        pair[mode] = worker
        if mode == "control":
            if worker.get("root_terminal_receipt") is not None:
                raise ValueError("five-pair control receipt differs")
        else:
            _validate_candidate_receipt(worker)
            receipt = _as_mapping(worker["root_terminal_receipt"], "receipt")
            for name, values in candidate_hashes.items():
                values.add(str(receipt[name]))
    if set(by_pair) != set(range(PAIR_COUNT)):
        raise ValueError("five-pair coverage differs")
    for name, values in candidate_hashes.items():
        if len(values) != 1:
            raise ValueError(f"five-pair compiler identity differs: {name}")

    metric_names = (
        "query_wall_ns",
        "root_incomplete_wall_ns",
        "optimized_bounds_transaction_wall_ns",
    )
    ratios: dict[str, list[float]] = {name: [] for name in metric_names}
    pair_rows = []
    max_lower_diff = 0.0
    for pair_index in range(PAIR_COUNT):
        pair = by_pair[pair_index]
        if set(pair) != {"control", "candidate"}:
            raise ValueError("five-pair mode coverage differs")
        expected_order = (
            ("control", "candidate")
            if pair_index % 2 == 0
            else (
                "candidate",
                "control",
            )
        )
        observed_order = tuple(
            str(worker["mode"])
            for worker in sorted(
                pair.values(),
                key=lambda worker: _integer(
                    worker["sequence_position"], "sequence position"
                ),
            )
        )
        if observed_order != expected_order:
            raise ValueError("five-pair alternating order differs")
        control = pair["control"]
        candidate = pair["candidate"]
        lower_diff = _semantics_diff(control, candidate)
        max_lower_diff = max(max_lower_diff, lower_diff)
        pair_ratios = {}
        for metric_name in metric_names:
            ratio = _metric(control, metric_name) / _metric(candidate, metric_name)
            ratios[metric_name].append(ratio)
            pair_ratios[metric_name] = ratio
        pair_rows.append(
            {
                "pair_index": pair_index,
                "order": list(expected_order),
                "speedups": pair_ratios,
                "lower_max_abs_diff": lower_diff,
            }
        )
    aggregates = {
        name: {
            "geomean": _geomean(values),
            "worst": min(values),
            "best": max(values),
            "pairs": values,
        }
        for name, values in ratios.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_count": PAIR_COUNT,
        "fresh_process_count": PAIR_COUNT * 2,
        "pairs": pair_rows,
        "speedups": aggregates,
        "max_lower_abs_diff": max_lower_diff,
        "lower_atol": LOWER_ATOL,
        "discrete_semantics_exact": True,
        "candidate_compiler_hashes": {
            name: next(iter(values)) for name, values in candidate_hashes.items()
        },
        "compile_excluded_from_query": True,
        "performance_claimed": False,
        "decision": "mechanism-correct-no-stable-query-speedup",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _pack(args: argparse.Namespace) -> None:
    workers = _read_raw_workers(args.raw_dir)
    summary = _derive_summary(workers)
    args.artifact.mkdir(parents=True, exist_ok=False)
    workers_payload = b"".join(_canonical_bytes(worker) + b"\n" for worker in workers)
    (args.artifact / "workers.jsonl").write_bytes(workers_payload)
    _write_json(args.artifact / "summary.json", summary)
    if args.isolated_probe is not None:
        isolated = json.loads(args.isolated_probe.read_text(encoding="utf-8"))
        if b"/home/" in _canonical_bytes(isolated) or b"/tmp/" in _canonical_bytes(
            isolated
        ):
            raise ValueError("isolated probe leaks a local path")
        _write_json(args.artifact / "isolated_probe.json", isolated)
    speedups = cast(Mapping[str, Any], summary["speedups"])
    readme = (
        "# Root CROWN terminal TIR five-pair artifact\n\n"
        "This artifact contains a local-path-free projection of ten fresh "
        "same-solver processes. It binds the final semantics, wrapper timing, "
        "root timing, compiled-module identities, and activation receipts.\n\n"
        f"- query geomean: `{speedups['query_wall_ns']['geomean']:.12f}x`\n"
        f"- root geomean: "
        f"`{speedups['root_incomplete_wall_ns']['geomean']:.12f}x`\n"
        f"- optimizer-transaction geomean: "
        f"`{speedups['optimized_bounds_transaction_wall_ns']['geomean']:.12f}x`\n"
        f"- maximum lower difference: `{summary['max_lower_abs_diff']:.12g}`\n"
        "- decision: mechanism correct; no stable query speedup claim\n\n"
        "Replay with:\n\n"
        "```bash\n"
        "python scripts/package_root_crown_terminal_five_pair.py replay "
        "--artifact artifacts/root-crown-terminal-tir/resnet2b-prop0-v1\n"
        "```\n"
    )
    (args.artifact / "README.md").write_text(readme, encoding="utf-8")
    repository = args.repository.resolve()
    code_revision = {name: _sha256_file(repository / name) for name in CODE_PATHS}
    data_files = sorted(path.name for path in args.artifact.iterdir() if path.is_file())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_parent_revision": args.source_parent_revision,
        "code_revision": code_revision,
        "files": {name: _sha256_file(args.artifact / name) for name in data_files},
        "model_relative_path": "benchmarks/cifar10_resnet/onnx/resnet_2b.onnx",
        "model_sha256": "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d",
        "property_relative_path": (
            "benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/"
            "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
        ),
        "property_sha256": (
            "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
        ),
        "raw_projection": True,
        "raw_projection_excludes_local_environment_and_event_stream": True,
        "performance_claimed": False,
    }
    _write_json(args.artifact / "manifest.json", manifest)
    _replay(args.artifact, repository, check_code=True)


def _replay(artifact: Path, repository: Path, *, check_code: bool) -> None:
    manifest = _as_mapping(
        json.loads((artifact / "manifest.json").read_text(encoding="utf-8")),
        "manifest",
    )
    files = _as_mapping(manifest["files"], "manifest files")
    for name, expected in files.items():
        if _sha256_file(artifact / str(name)) != expected:
            raise ValueError(f"five-pair artifact file differs: {name}")
    if check_code:
        code_revision = _as_mapping(manifest["code_revision"], "code revision")
        for name, expected in code_revision.items():
            if _sha256_file(repository / str(name)) != expected:
                raise ValueError(f"five-pair code revision differs: {name}")
    workers = _read_artifact_workers(artifact)
    derived = _derive_summary(workers)
    stored = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
    if _canonical_bytes(derived) != _canonical_bytes(stored):
        raise ValueError("five-pair derived summary differs")
    print(json.dumps(derived, sort_keys=True, separators=(",", ":")))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack")
    pack.add_argument("--raw-dir", type=Path, required=True)
    pack.add_argument("--artifact", type=Path, required=True)
    pack.add_argument("--repository", type=Path, default=Path.cwd())
    pack.add_argument("--source-parent-revision", required=True)
    pack.add_argument("--isolated-probe", type=Path)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact", type=Path, required=True)
    replay.add_argument("--repository", type=Path, default=Path.cwd())
    replay.add_argument("--skip-code-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected pack or replay operation."""
    args = _parse_args()
    if args.command == "pack":
        _pack(args)
    else:
        _replay(
            args.artifact,
            args.repository.resolve(),
            check_code=not args.skip_code_check,
        )


if __name__ == "__main__":
    main()
