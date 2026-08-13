#!/usr/bin/env python3
"""Generate or replay the RVIR-v4 V4-3E five-fresh artifact."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from boundflow.runtime.rvir_v4_whole_core_truth import (
    compare_rvir_v4_live_return_truth,
    validate_rvir_v4_whole_core_truth,
)
from scripts import run_rvir_v4_five_fresh_worker as fresh_worker
from scripts import run_rvir_v4_live_return_artifact as live_artifact_runner
from scripts import run_rvir_v4_live_return_capture as live_runner
from scripts import run_rvir_v4_production_state_capture as original_runner

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-five-fresh-artifact/v1"
SEQUENCE = (
    "original",
    "candidate",
    "candidate",
    "original",
    "candidate",
    "original",
    "original",
    "candidate",
    "original",
    "candidate",
)
PAIR_INDICES = ((0, 1), (3, 2), (5, 4), (6, 7), (8, 9))
RUN_FILES = tuple(f"run_{index:02d}_{mode}.pt" for index, mode in enumerate(SEQUENCE))
STDOUT_FILES = tuple(
    f"run_{index:02d}_{mode}.stdout.txt" for index, mode in enumerate(SEQUENCE)
)
ARTIFACT_FILES = (
    RUN_FILES
    + STDOUT_FILES
    + (
        "summary.json",
        "replay_stdout.txt",
        "README.md",
    )
)
REPLAY_CONTRACT = {
    "sequence": list(SEQUENCE),
    "pair_indices": [list(value) for value in PAIR_INDICES],
    "fresh_process_per_run": True,
    "cold_isolated_property": True,
    "atol": 2e-4,
    "rtol": 2e-4,
    "sign_exact": True,
    "discrete_structure_exact": True,
    "performance_claimed": False,
}
CODE_PATHS = (
    "boundflow/runtime/rvir_v4_atomic_copy_out.py",
    "boundflow/runtime/rvir_v4_live_return.py",
    "boundflow/runtime/rvir_v4_whole_core_truth.py",
    "scripts/run_rvir_v4_five_fresh_worker.py",
    "scripts/run_rvir_v4_five_fresh_artifact.py",
    "scripts/probe_rvir_v4_five_fresh_artifact_tamper.py",
)


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
        raise TypeError(f"RVIR-v4 five-fresh JSON root differs: {path}")
    return value


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"RVIR-v4 five-fresh torch root differs: {path}")
    return value


def _git_value(*args: str) -> str:
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


def _code_paths_clean() -> bool:
    return not _git_value("status", "--porcelain=v1", "--", *CODE_PATHS)


def _verify_code_provenance(manifest: Mapping[str, Any]) -> None:
    source_head = manifest.get("source_git_head")
    revision = manifest.get("code_revision")
    if not isinstance(source_head, str) or not isinstance(revision, Mapping):
        raise ValueError("RVIR-v4 five-fresh code provenance differs")
    if _git_value("rev-parse", "HEAD") == source_head:
        observed = _code_revision()
    else:
        observed = {
            path: hashlib.sha256(
                subprocess.run(
                    ("git", "show", f"{source_head}:{path}"),
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).stdout
            ).hexdigest()
            for path in CODE_PATHS
        }
    if dict(revision) != observed:
        raise ValueError("RVIR-v4 five-fresh code revision differs")


def _core_post(
    payload: Mapping[str, Any], mode: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    core_key = "whole_core_truths" if mode == "original" else "whole_core_results"
    post_key = "whole_post_truths" if mode == "original" else "whole_post_results"
    cores = payload.get(core_key)
    posts = payload.get(post_key)
    if (
        not isinstance(cores, list)
        or len(cores) != 1
        or not isinstance(cores[0], Mapping)
        or not isinstance(posts, list)
        or len(posts) != 1
        or not isinstance(posts[0], Mapping)
    ):
        raise ValueError("RVIR-v4 five-fresh core/post inventory differs")
    return cast(Mapping[str, Any], cores[0]), cast(Mapping[str, Any], posts[0])


def _one_tensor_record(mapping: object, *, label: str) -> torch.Tensor:
    if not isinstance(mapping, Mapping) or len(mapping) != 1:
        raise ValueError(f"RVIR-v4 five-fresh {label} inventory differs")
    record = next(iter(mapping.values()))
    if not isinstance(record, Mapping) or not torch.is_tensor(record.get("value")):
        raise TypeError(f"RVIR-v4 five-fresh {label} record differs")
    return cast(torch.Tensor, record["value"])


def _validate_queue_event(
    payload: Mapping[str, Any], *, mode: str
) -> Mapping[str, Any]:
    events = payload.get("queue_events")
    if (
        payload.get("five_fresh_worker_schema") != fresh_worker.WORKER_SCHEMA
        or payload.get("five_fresh_mode") != mode
        or not isinstance(events, list)
        or len(events) != 1
        or not isinstance(events[0], Mapping)
    ):
        raise ValueError("RVIR-v4 five-fresh queue event inventory differs")
    event = cast(Mapping[str, Any], events[0])
    required = {
        "schema_version",
        "before_domain_count",
        "input_domain_count",
        "accepted_domain_count",
        "pruned_domain_count",
        "after_domain_count",
        "final_name",
        "lower_sha256",
        "upper_sha256",
        "thresholds_sha256",
        "history_count",
        "depths",
        "performance_claimed",
    }
    core, post = _core_post(payload, mode)
    fields = cast(Mapping[str, Any], core["fields"])
    lower = _one_tensor_record(post["lower_bounds"], label="post lower")
    upper = _one_tensor_record(post["upper_bounds"], label="post upper")
    thresholds_record = cast(Mapping[str, Any], fields["thresholds"])
    depths_record = cast(Mapping[str, Any], fields["depths"])
    thresholds = thresholds_record.get("value")
    depths = depths_record.get("value")
    if (
        set(event) != required
        or event.get("schema_version") != fresh_worker.WORKER_SCHEMA
        or event.get("before_domain_count") != 0
        or event.get("input_domain_count") != 6
        or event.get("accepted_domain_count") != 6
        or event.get("pruned_domain_count") != 0
        or event.get("after_domain_count") != 6
        or event.get("final_name") != "/49"
        or event.get("history_count") != 6
        or event.get("depths") != [1, 1, 1, 1, 1, 1]
        or event.get("performance_claimed") is not False
        or not torch.is_tensor(thresholds)
        or not torch.is_tensor(depths)
        or event.get("lower_sha256") != production_tensor_sha256(lower)
        or event.get("upper_sha256") != production_tensor_sha256(upper)
        or event.get("thresholds_sha256")
        != production_tensor_sha256(cast(torch.Tensor, thresholds))
        or event.get("depths") != [int(value) for value in depths.tolist()]
    ):
        raise ValueError("RVIR-v4 five-fresh queue accounting differs")
    return event


def _validate_run(payload: Mapping[str, Any], *, mode: str) -> dict[str, object]:
    solver = payload.get("solver_result")
    if (
        not isinstance(solver, Mapping)
        or solver.get("status") != "verified"
        or solver.get("success") is not True
        or solver.get("visited_domains") != [6]
        or payload.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 five-fresh solver result differs")
    event = _validate_queue_event(payload, mode=mode)
    core, post = _core_post(payload, mode)
    if mode == "original":
        calls = payload.get("calls")
        if (
            payload.get("schema_version")
            != original_runner.WHOLE_CORE_WORKER_SCHEMA_VERSION
            or not isinstance(calls, list)
            or len(calls) != 24
        ):
            raise ValueError("RVIR-v4 five-fresh original lineage differs")
        validated = validate_rvir_v4_whole_core_truth(core, post)
        lineage = {"provider_call_count": 24, "candidate_callback_count": None}
    else:
        if payload.get("schema_version") != live_runner.WORKER_SCHEMA:
            raise ValueError("RVIR-v4 five-fresh candidate schema differs")
        validated = live_artifact_runner._structural_summary(payload)
        lineage = {"provider_call_count": 0, "candidate_callback_count": 0}
    return {
        "mode": mode,
        "solver_result": dict(solver),
        "queue_accounting": {
            key: event[key]
            for key in (
                "before_domain_count",
                "input_domain_count",
                "accepted_domain_count",
                "pruned_domain_count",
                "after_domain_count",
                "depths",
            )
        },
        "branching_decision": validated["branching_decision"],
        "n_verified": validated["n_verified"],
        "n_splits": validated["n_splits"],
        **lineage,
        "performance_claimed": False,
    }


def _pair_summary(
    *,
    pair_index: int,
    original_index: int,
    candidate_index: int,
    original: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, object]:
    original_summary = _validate_run(original, mode="original")
    candidate_summary = _validate_run(candidate, mode="candidate")
    original_core, original_post = _core_post(original, "original")
    candidate_core, candidate_post = _core_post(candidate, "candidate")
    parity = compare_rvir_v4_live_return_truth(
        original_core,
        original_post,
        candidate_core,
        candidate_post,
    )
    for key in (
        "solver_result",
        "queue_accounting",
        "branching_decision",
        "n_verified",
        "n_splits",
    ):
        if original_summary[key] != candidate_summary[key]:
            raise ValueError(f"RVIR-v4 five-fresh pair discrete parity differs: {key}")
    summary: dict[str, object] = {
        "pair_index": pair_index,
        "original_run_index": original_index,
        "candidate_run_index": candidate_index,
        "execution_order": (
            "original-candidate"
            if original_index < candidate_index
            else "candidate-original"
        ),
        "original_provider_call_count": 24,
        "candidate_provider_callback_count": 0,
        "solver_result_exact": True,
        "queue_accounting_exact": True,
        "branching_decision_exact": True,
        "domain_accounting_exact": True,
        "semantic_parity": parity,
        "pair_admitted": True,
        "performance_claimed": False,
    }
    summary["pair_hash"] = _canonical_hash(summary)
    return summary


def _summary(runs: list[Mapping[str, Any]]) -> dict[str, object]:
    if len(runs) != len(SEQUENCE):
        raise ValueError("RVIR-v4 five-fresh run count differs")
    run_summaries = [
        _validate_run(payload, mode=mode) for payload, mode in zip(runs, SEQUENCE)
    ]
    pairs = [
        _pair_summary(
            pair_index=pair_index,
            original_index=original_index,
            candidate_index=candidate_index,
            original=runs[original_index],
            candidate=runs[candidate_index],
        )
        for pair_index, (original_index, candidate_index) in enumerate(PAIR_INDICES)
    ]
    parities = [cast(Mapping[str, Any], pair["semantic_parity"]) for pair in pairs]
    summary: dict[str, object] = {
        "status": "validated-five-fresh-correctness",
        "sequence": list(SEQUENCE),
        "run_count": len(runs),
        "pair_count": len(pairs),
        "original_run_count": SEQUENCE.count("original"),
        "candidate_run_count": SEQUENCE.count("candidate"),
        "run_summaries": run_summaries,
        "pairs": pairs,
        "all_pairs_admitted": all(pair["pair_admitted"] is True for pair in pairs),
        "maximum_absolute_difference": max(
            float(parity["max_abs_diff"]) for parity in parities
        ),
        "tensor_comparison_count": sum(
            int(parity["tensor_count"]) for parity in parities
        ),
        "sign_element_comparison_count": sum(
            int(parity["sign_element_count"]) for parity in parities
        ),
        "all_sign_exact": all(parity["sign_exact"] is True for parity in parities),
        "provider_original_call_count": 5 * 24,
        "candidate_provider_core_callback_count": 0,
        "candidate_provider_compute_bounds_callback_count": 0,
        "candidate_provider_update_bounds_callback_count": 0,
        "candidate_fallback_dispatch_count": 0,
        "accepted_domain_count_per_run": 6,
        "pruned_domain_count_per_run": 0,
        "visited_domains_per_run": [6],
        "five_fresh_correctness_admitted": True,
        "whole_core_replacement_admitted": True,
        "b2_same_solver_timing_admitted": True,
        "performance_claimed": False,
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "run_count": summary["run_count"],
        "pair_count": summary["pair_count"],
        "all_pairs_admitted": summary["all_pairs_admitted"],
        "maximum_absolute_difference": summary["maximum_absolute_difference"],
        "all_sign_exact": summary["all_sign_exact"],
        "b2_same_solver_timing_admitted": summary["b2_same_solver_timing_admitted"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-3E Five-Fresh Correctness\n\n"
        "This artifact contains ten isolated GPU processes in the preregistered "
        "O,C,C,O,C,O,O,C,O,C order. Each of five pairs compares complete core/post "
        "semantics, state, branch decisions, queue admission, visited domains and "
        "termination. It admits later B2 timing but contains no timing claim.\n"
    )


def _run_worker(
    *, mode: str, benchmark: Path, abcrown: Path, python: Path, result: Path
) -> tuple[dict[str, Any], str]:
    command = (
        str(python),
        str(REPOSITORY_ROOT / "scripts/run_rvir_v4_five_fresh_worker.py"),
        "--mode",
        mode,
        "--benchmark-root",
        str(benchmark),
        "--abcrown-root",
        str(abcrown),
        "--model",
        str(benchmark / original_runner.MODEL_RELATIVE_PATH),
        "--property",
        str(benchmark / original_runner.PROPERTY_RELATIVE_PATH),
        "--result",
        str(result),
    )
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"RVIR-v4 five-fresh {mode} worker failed:\n{completed.stdout}"
        )
    return _load_torch(result), completed.stdout


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("RVIR-v4 five-fresh code paths must be clean")
    output = args.artifact_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    benchmark = args.benchmark_root.resolve()
    abcrown = args.abcrown_root.resolve()
    python = args.abcrown_python.expanduser().absolute()
    runs: list[Mapping[str, Any]] = []
    for index, mode in enumerate(SEQUENCE):
        payload, stdout = _run_worker(
            mode=mode,
            benchmark=benchmark,
            abcrown=abcrown,
            python=python,
            result=output / RUN_FILES[index],
        )
        runs.append(payload)
        (output / STDOUT_FILES[index]).write_text(stdout, encoding="utf-8")
    summary = _summary(runs)
    _write_json(output / "summary.json", summary)
    result = _replay_result(summary)
    (output / "replay_stdout.txt").write_text(
        _canonical_json(result) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(_readme(), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_git_head": _git_value("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "files": {name: _file_sha256(output / name) for name in ARTIFACT_FILES},
        "abcrown_commit": original_runner.ABCROWN_COMMIT,
        "auto_lirpa_commit": original_runner.AUTO_LIRPA_COMMIT,
        "vnncomp_commit": original_runner.VNNCOMP_COMMIT,
        "model_sha256": original_runner.MODEL_SHA256,
        "property_sha256": original_runner.PROPERTY_SHA256,
        "replay_contract": REPLAY_CONTRACT,
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    _write_json(output / "manifest.json", manifest)
    return result


def _verify_static_artifact(
    artifact: Path,
) -> tuple[list[dict[str, Any]], dict[str, object], dict[str, object]]:
    manifest = _load_json(artifact / "manifest.json")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("manifest_hash") != _canonical_hash(semantic)
        or manifest.get("replay_contract") != REPLAY_CONTRACT
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 five-fresh manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 five-fresh artifact inventory differs")
    for name in ARTIFACT_FILES:
        if files[name] != _file_sha256(artifact / name):
            raise ValueError(f"RVIR-v4 five-fresh digest differs: {name}")
    runs = [_load_torch(artifact / name) for name in RUN_FILES]
    summary = _summary(cast(list[Mapping[str, Any]], runs))
    if _load_json(artifact / "summary.json") != summary:
        raise ValueError("RVIR-v4 five-fresh semantic replay differs")
    if (
        manifest.get("summary_hash") != summary["summary_hash"]
        or manifest.get("status") != summary["status"]
    ):
        raise ValueError("RVIR-v4 five-fresh summary identity differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 five-fresh replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 five-fresh README differs")
    return runs, summary, result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--artifact-dir", type=Path, required=True)
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate or replay the five-fresh correctness artifact."""

    args = _parse_args()
    if args.command == "generate":
        result = _generate(args)
    else:
        _runs, _summary_payload, result = _verify_static_artifact(
            args.artifact_dir.resolve()
        )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
