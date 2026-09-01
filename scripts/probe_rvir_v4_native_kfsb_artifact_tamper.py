#!/usr/bin/env python3
"""Probe synchronized tampering of the RVIR-v4 V4-3C artifact."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_rvir_v4_native_kfsb_artifact as artifact_runner

REPORT_SCHEMA_VERSION = "boundflow.rvir-v4-native-kfsb-tamper-report/v1"


def _resign_manifest(artifact: Path, *changed_files: str) -> None:
    manifest_path = artifact / "manifest.json"
    manifest = artifact_runner._load_json(manifest_path)
    files = manifest["files"]
    for name in changed_files:
        files[name] = artifact_runner._file_sha256(artifact / name)
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = artifact_runner._canonical_hash(semantic)
    artifact_runner._write_json(manifest_path, manifest)


def _resign_evaluation(
    artifact: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    path = artifact / artifact_runner.EVALUATION_FILE
    payload = artifact_runner._load_torch(path)
    mutate(payload)
    typed = artifact_runner._evaluation_from_payload(payload, validate_metadata=False)
    metadata = typed.metadata()
    payload["metadata"] = metadata
    torch.save(payload, path)
    summary_path = artifact / "summary.json"
    summary = artifact_runner._load_json(summary_path)
    summary["evaluation_hash"] = metadata["evaluation_hash"]
    summary.pop("summary_hash", None)
    summary["summary_hash"] = artifact_runner._canonical_hash(summary)
    artifact_runner._write_json(summary_path, summary)
    replay = artifact_runner._replay_result(summary)
    replay_path = artifact / "replay_stdout.txt"
    replay_path.write_text(
        artifact_runner._canonical_json(replay) + "\n", encoding="utf-8"
    )
    manifest_path = artifact / "manifest.json"
    manifest = artifact_runner._load_json(manifest_path)
    manifest["evaluation_hash"] = summary["evaluation_hash"]
    manifest["summary_hash"] = summary["summary_hash"]
    for name in (
        artifact_runner.EVALUATION_FILE,
        "summary.json",
        "replay_stdout.txt",
    ):
        manifest["files"][name] = artifact_runner._file_sha256(artifact / name)
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = artifact_runner._canonical_hash(semantic)
    artifact_runner._write_json(manifest_path, manifest)


def _mutate_candidate(payload: dict[str, Any]) -> None:
    payload["candidate_splits"][0][0][1] += 1


def _mutate_child_lower(payload: dict[str, Any]) -> None:
    value = payload["candidate_child_lowers"][0].clone()
    value.flatten()[0] += 1.0
    payload["candidate_child_lowers"][0] = value


def _mutate_final_decision(payload: dict[str, Any]) -> None:
    payload["final_decision"][0][1] += 1


def _mutate_mask(payload: dict[str, Any]) -> None:
    value = payload["unstable_masks"]["/input"].clone()
    value.flatten()[0] = torch.logical_not(value.flatten()[0])
    payload["unstable_masks"]["/input"] = value
    names = list(payload["unstable_masks"])
    payload["unstable_counts"][names.index("/input")] = int(value.sum().item())


def _mutate_score(payload: dict[str, Any]) -> None:
    value = payload["alpha_score_topk_values"].clone()
    value.flatten()[0] += 1.0
    payload["alpha_score_topk_values"] = value


def _mutate_reduced(payload: dict[str, Any]) -> None:
    value = payload["reduced_candidate_values"].clone()
    value.flatten()[0] += 1.0
    payload["reduced_candidate_values"] = value


def _mutate_topology(artifact: Path) -> None:
    path = artifact / "topology.json"
    topology = artifact_runner._load_json(path)
    topology["rows"][0]["provider_preactivation"] = "/tampered"
    topology["topology_hash"] = artifact_runner._canonical_hash(topology["rows"])
    artifact_runner._write_json(path, topology)
    _resign_manifest(artifact, "topology.json")


def _mutate_truth_source(artifact: Path) -> None:
    path = artifact / artifact_runner.SOURCE_TRUTH_FILE
    truth = artifact_runner._load_torch(path)
    value = truth["whole_core_truths"][0]["branch_trace"]["candidate_child_lowers"][0][
        "value"
    ].clone()
    value.flatten()[0] += 1.0
    truth["whole_core_truths"][0]["branch_trace"]["candidate_child_lowers"][0][
        "value"
    ] = value
    torch.save(truth, path)
    _resign_manifest(artifact, artifact_runner.SOURCE_TRUTH_FILE)


def run_probe_suite(*, artifact: Path, model: Path) -> dict[str, object]:
    """Run six full evaluation resigns and two outer resigns."""

    artifact_runner._replay(artifact, model)
    rows: list[dict[str, object]] = []
    attacks: tuple[tuple[str, Callable[[Path], None]], ...] = (
        (
            "candidate-split-full-resign",
            lambda path: _resign_evaluation(path, _mutate_candidate),
        ),
        (
            "child-lower-full-resign",
            lambda path: _resign_evaluation(path, _mutate_child_lower),
        ),
        (
            "final-decision-full-resign",
            lambda path: _resign_evaluation(path, _mutate_final_decision),
        ),
        (
            "unstable-mask-full-resign",
            lambda path: _resign_evaluation(path, _mutate_mask),
        ),
        (
            "topk-score-full-resign",
            lambda path: _resign_evaluation(path, _mutate_score),
        ),
        (
            "candidate-reduction-full-resign",
            lambda path: _resign_evaluation(path, _mutate_reduced),
        ),
        ("topology-outer-resign", _mutate_topology),
        ("truth-source-outer-resign", _mutate_truth_source),
    )
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-kfsb-tamper-") as raw:
        workspace = Path(raw)
        for name, mutate in attacks:
            probe = workspace / name
            shutil.copytree(artifact, probe)
            mutate(probe)
            try:
                artifact_runner._replay(probe, model)
            except (TypeError, ValueError) as error:
                rows.append(
                    {
                        "name": name,
                        "outer_resigned": True,
                        "rejected": True,
                        "rejection": str(error),
                    }
                )
            else:
                raise AssertionError(f"tampered native KFSB artifact admitted: {name}")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_source_git_head": artifact_runner._load_json(
            artifact / "manifest.json"
        )["source_git_head"],
        "attack_count": len(rows),
        "fully_resigned_evaluation_attack_count": 6,
        "all_rejected": all(row["rejected"] is True for row in rows),
        "attacks": rows,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run probes and write the formal report."""

    args = _parse_args()
    report = run_probe_suite(
        artifact=args.artifact_dir.resolve(), model=args.model.resolve()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
