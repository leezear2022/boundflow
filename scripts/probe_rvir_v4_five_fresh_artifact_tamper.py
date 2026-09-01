#!/usr/bin/env python3
"""Probe resigned tampering of the RVIR-v4 V4-3E artifact."""

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
from typing import Any, Callable, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from boundflow.runtime.rvir_v4_whole_core_truth import whole_core_truth_metadata
from scripts import run_rvir_v4_five_fresh_artifact as artifact_runner

REPORT_SCHEMA_VERSION = "boundflow.rvir-v4-five-fresh-tamper-report/v1"
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
                raise TypeError("RVIR-v4 five-fresh tamper tensor differs")
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


def _resign_run(payload: dict[str, Any], mode: str) -> None:
    core_key = "whole_core_truths" if mode == "original" else "whole_core_results"
    post_key = "whole_post_truths" if mode == "original" else "whole_post_results"
    _resign_truth_record(cast(list[dict[str, Any]], payload[core_key])[0])
    _resign_truth_record(cast(list[dict[str, Any]], payload[post_key])[0])


def _resign_artifact(
    artifact: Path, *, run_index: int, payload: dict[str, Any]
) -> None:
    mode = artifact_runner.SEQUENCE[run_index]
    _resign_run(payload, mode)
    torch.save(payload, artifact / artifact_runner.RUN_FILES[run_index])
    summary = artifact_runner._load_json(artifact / "summary.json")
    summary.pop("summary_hash", None)
    summary["summary_hash"] = artifact_runner._canonical_hash(summary)
    artifact_runner._write_json(artifact / "summary.json", summary)
    replay = artifact_runner._replay_result(summary)
    (artifact / "replay_stdout.txt").write_text(
        artifact_runner._canonical_json(replay) + "\n", encoding="utf-8"
    )
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    manifest["files"] = {
        name: artifact_runner._file_sha256(artifact / name)
        for name in artifact_runner.ARTIFACT_FILES
    }
    manifest["summary_hash"] = summary["summary_hash"]
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = artifact_runner._canonical_hash(semantic)
    artifact_runner._write_json(artifact / "manifest.json", manifest)


def _candidate_core(payload: dict[str, Any]) -> dict[str, Any]:
    return cast(list[dict[str, Any]], payload["whole_core_results"])[0]


def _original_core(payload: dict[str, Any]) -> dict[str, Any]:
    return cast(list[dict[str, Any]], payload["whole_core_truths"])[0]


def _candidate_post(payload: dict[str, Any]) -> dict[str, Any]:
    return cast(list[dict[str, Any]], payload["whole_post_results"])[0]


def _mutate_candidate_l_a(payload: dict[str, Any]) -> None:
    record = _candidate_core(payload)["branch_trace"]["input"]["lAs"]["_data"]["/48"]
    record["value"] = record["value"].clone()
    record["value"].flatten()[0] += 1.0


def _mutate_original_l_a(payload: dict[str, Any]) -> None:
    record = _original_core(payload)["branch_trace"]["input"]["lAs"]["_data"]["/48"]
    record["value"] = record["value"].clone()
    record["value"].flatten()[0] += 1.0


def _mutate_candidate_decision(payload: dict[str, Any]) -> None:
    core = _candidate_core(payload)
    changed = deepcopy(core["branching_decision"]["decision"])
    changed[0][1] += 1
    core["branching_decision"]["decision"] = changed
    core["branch_trace"]["final_decision"]["decision"] = deepcopy(changed)
    _candidate_post(payload)["decision_info"]["packed_branching_decision"][0][0][1] += 1


def _mutate_queue_accounting(payload: dict[str, Any]) -> None:
    event = cast(list[dict[str, Any]], payload["queue_events"])[0]
    event["accepted_domain_count"] = 5
    event["pruned_domain_count"] = 1


def _mutate_candidate_callback(payload: dict[str, Any]) -> None:
    payload["provider_update_bounds_callback_count"] = 1


RUN_ATTACKS: tuple[tuple[str, int, Callable[[dict[str, Any]], None], bool], ...] = (
    ("candidate-lA-full-resign", 1, _mutate_candidate_l_a, True),
    ("original-lA-full-resign", 0, _mutate_original_l_a, True),
    ("candidate-decision-cross-resign", 2, _mutate_candidate_decision, True),
    ("queue-accounting-outer-resign", 4, _mutate_queue_accounting, False),
    ("candidate-callback-outer-resign", 7, _mutate_candidate_callback, False),
)


def _manifest_sequence_attack(artifact: Path) -> None:
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    replay_contract = cast(dict[str, Any], manifest["replay_contract"])
    sequence = list(replay_contract["sequence"])
    sequence[0], sequence[1] = sequence[1], sequence[0]
    replay_contract["sequence"] = sequence
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = artifact_runner._canonical_hash(semantic)
    artifact_runner._write_json(artifact / "manifest.json", manifest)


def run_probe_suite(*, artifact: Path) -> dict[str, object]:
    """Reject five run attacks and one independently resigned sequence attack."""

    _runs, summary, _result = artifact_runner._verify_static_artifact(artifact)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-five-tamper-") as raw:
        workspace = Path(raw)
        for name, run_index, mutate, inner_resigned in RUN_ATTACKS:
            probe = workspace / name
            shutil.copytree(artifact, probe)
            payload = artifact_runner._load_torch(
                probe / artifact_runner.RUN_FILES[run_index]
            )
            mutate(payload)
            _resign_artifact(probe, run_index=run_index, payload=payload)
            rejection = ""
            try:
                artifact_runner._verify_static_artifact(probe)
            except (TypeError, ValueError) as error:
                rejection = str(error)
            else:
                raise AssertionError(f"tampered five-fresh artifact admitted: {name}")
            rows.append(
                {
                    "name": name,
                    "inner_truth_resigned": inner_resigned,
                    "outer_artifact_resigned": True,
                    "rejected": True,
                    "rejection": rejection,
                }
            )
        name = "sequence-order-outer-resign"
        probe = workspace / name
        shutil.copytree(artifact, probe)
        _manifest_sequence_attack(probe)
        try:
            artifact_runner._verify_static_artifact(probe)
        except (TypeError, ValueError) as error:
            rejection = str(error)
        else:
            raise AssertionError("tampered five-fresh sequence admitted")
        rows.append(
            {
                "name": name,
                "inner_truth_resigned": False,
                "outer_artifact_resigned": True,
                "rejected": True,
                "rejection": rejection,
            }
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_source_git_head": artifact_runner._load_json(
            artifact / "manifest.json"
        )["source_git_head"],
        "clean_summary_hash": summary["summary_hash"],
        "attack_count": len(rows),
        "inner_truth_resigned_attack_count": sum(
            int(cast(bool, row["inner_truth_resigned"])) for row in rows
        ),
        "outer_artifact_resigned_attack_count": len(rows),
        "all_rejected": all(row["rejected"] is True for row in rows),
        "attacks": rows,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the tamper suite and write its canonical report."""

    args = _parse_args()
    report = run_probe_suite(artifact=args.artifact_dir.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
