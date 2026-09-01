#!/usr/bin/env python3
"""Probe fully resigned tampering of the RVIR-v4 V4-3A truth artifact."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,duplicate-code

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from boundflow.runtime.rvir_v4_whole_core_truth import (
    compare_rvir_v4_whole_core_truth,
    whole_core_truth_metadata,
)
from scripts import run_rvir_v4_whole_core_truth_artifact as artifact_runner

REPORT_SCHEMA_VERSION = "boundflow.rvir-v4-whole-core-truth-tamper-report/v1"
TENSOR_RECORD_KEYS = {
    "shape",
    "dtype",
    "source_device",
    "content_sha256",
    "value",
}


def _resign_tensor_tree(value: object) -> None:
    if isinstance(value, dict):
        if set(value) == TENSOR_RECORD_KEYS:
            tensor = value["value"]
            if not torch.is_tensor(tensor):
                raise TypeError("RVIR-v4 tamper tensor value differs")
            value["content_sha256"] = production_tensor_sha256(tensor)
            return
        for item in value.values():
            _resign_tensor_tree(item)
    elif isinstance(value, list):
        for item in value:
            _resign_tensor_tree(item)


def _resign_truth_record(record: dict[str, Any]) -> None:
    _resign_tensor_tree(record)
    semantic = {key: value for key, value in record.items() if key != "truth_hash"}
    record["truth_hash"] = artifact_runner._canonical_hash(
        whole_core_truth_metadata(semantic)
    )


def _resign_artifact(artifact: Path, payload: dict[str, Any]) -> None:
    cores = cast(list[dict[str, Any]], payload["whole_core_truths"])
    posts = cast(list[dict[str, Any]], payload["whole_post_truths"])
    _resign_truth_record(cores[0])
    _resign_truth_record(posts[0])
    torch.save(payload, artifact / artifact_runner.TRUTH_FILE)
    summary = artifact_runner._summary(payload)
    artifact_runner._write_json(artifact / "summary.json", summary)
    result = artifact_runner._replay_result(summary)
    (artifact / "replay_stdout.txt").write_text(
        artifact_runner._canonical_json(result) + "\n", encoding="utf-8"
    )
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    manifest["files"] = {
        name: artifact_runner._file_sha256(artifact / name)
        for name in artifact_runner.ARTIFACT_FILES
    }
    manifest["summary_hash"] = summary["summary_hash"]
    manifest["status"] = summary["status"]
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = artifact_runner._canonical_hash(semantic)
    artifact_runner._write_json(artifact / "manifest.json", manifest)


def _core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return cast(list[dict[str, Any]], payload["whole_core_truths"])[0]


def _post(payload: Mapping[str, Any]) -> dict[str, Any]:
    return cast(list[dict[str, Any]], payload["whole_post_truths"])[0]


def _mutate_l_a(payload: dict[str, Any]) -> None:
    record = _core(payload)["branch_trace"]["input"]["lAs"]["_data"]["/48"]
    record["value"] = record["value"].clone()
    record["value"].flatten()[0] += 1.0


def _mutate_intermediate(payload: dict[str, Any]) -> None:
    record = _core(payload)["working_intermediate_bounds"]["/input-28"]["lower"]
    record["value"] = record["value"].clone()
    record["value"].flatten()[0] += 1.0


def _mutate_candidate_lower(payload: dict[str, Any]) -> None:
    record = _core(payload)["branch_trace"]["candidate_child_lowers"][1]
    record["value"] = record["value"].clone()
    record["value"].flatten()[0] += 1.0


def _mutate_branch_decision(payload: dict[str, Any]) -> None:
    core = _core(payload)
    changed = deepcopy(core["branching_decision"]["decision"])
    changed[0][1] += 1
    core["branching_decision"]["decision"] = changed
    core["branch_trace"]["final_decision"]["decision"] = deepcopy(changed)
    _post(payload)["decision_info"]["packed_branching_decision"][0][0][1] += 1


def _mutate_core_accounting(payload: dict[str, Any]) -> None:
    _core(payload)["lb_final_max"] += 1.0


def _delete_l_a_field(payload: dict[str, Any]) -> None:
    del _core(payload)["branch_trace"]["input"]["lAs"]["_data"]["/48"]


ATTACKS: tuple[tuple[str, Callable[[dict[str, Any]], None], bool], ...] = (
    ("lA-numeric-full-resign", _mutate_l_a, True),
    ("intermediate-numeric-full-resign", _mutate_intermediate, True),
    ("candidate-lower-full-resign", _mutate_candidate_lower, True),
    ("branch-decision-cross-resign", _mutate_branch_decision, True),
    ("core-accounting-full-resign", _mutate_core_accounting, True),
    ("lA-field-deletion-resign-attempt", _delete_l_a_field, False),
)


def run_probe_suite(
    *,
    artifact: Path,
    benchmark_root: Path,
    abcrown_root: Path,
    abcrown_python: Path,
) -> dict[str, object]:
    """Run one fresh provider truth against six independently resigned attacks."""

    frozen, _summary, _result = artifact_runner._verify_static_artifact(artifact)
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-whole-fresh-") as raw:
        fresh = artifact_runner._run_worker(
            benchmark=benchmark_root,
            abcrown=abcrown_root,
            python=abcrown_python,
            result=Path(raw) / "fresh.pt",
        )
    frozen_core, frozen_post = artifact_runner._truth_pair(frozen)
    fresh_core, fresh_post = artifact_runner._truth_pair(fresh)
    clean_parity = compare_rvir_v4_whole_core_truth(
        frozen_core, frozen_post, fresh_core, fresh_post
    )
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-whole-tamper-") as raw:
        workspace = Path(raw)
        for name, mutate, must_fully_resign in ATTACKS:
            probe = workspace / name
            shutil.copytree(artifact, probe)
            payload = artifact_runner._load_truth(probe / artifact_runner.TRUTH_FILE)
            mutate(payload)
            outer_resigned = False
            rejection = ""
            try:
                _resign_artifact(probe, payload)
                outer_resigned = True
                resigned, _resigned_summary, _resigned_result = (
                    artifact_runner._verify_static_artifact(probe)
                )
                changed_core, changed_post = artifact_runner._truth_pair(resigned)
                compare_rvir_v4_whole_core_truth(
                    changed_core, changed_post, fresh_core, fresh_post
                )
            except (TypeError, ValueError) as error:
                rejection = str(error)
            else:
                raise AssertionError(f"tampered whole-core truth was admitted: {name}")
            if must_fully_resign and not outer_resigned:
                raise AssertionError(f"attack did not reach full resign: {name}")
            rows.append(
                {
                    "name": name,
                    "expected_full_resign": must_fully_resign,
                    "outer_resigned": outer_resigned,
                    "rejected": True,
                    "rejection": rejection,
                }
            )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_source_git_head": artifact_runner._load_json(
            artifact / "manifest.json"
        )["source_git_head"],
        "clean_live_parity": clean_parity,
        "attack_count": len(rows),
        "fully_resigned_attack_count": sum(
            int(cast(bool, row["outer_resigned"])) for row in rows
        ),
        "all_rejected": all(row["rejected"] is True for row in rows),
        "attacks": rows,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--abcrown-python", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the suite and write its canonical report."""

    args = _parse_args()
    report = run_probe_suite(
        artifact=args.artifact_dir.resolve(),
        benchmark_root=args.benchmark_root.resolve(),
        abcrown_root=args.abcrown_root.resolve(),
        abcrown_python=args.abcrown_python.expanduser().absolute(),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
