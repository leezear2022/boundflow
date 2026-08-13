#!/usr/bin/env python3
"""Probe fully resigned tampering of the RVIR-v4 V4-3D live artifact."""

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
from boundflow.runtime.rvir_v4_whole_core_truth import whole_core_truth_metadata
from scripts import run_rvir_v4_live_return_artifact as artifact_runner

REPORT_SCHEMA_VERSION = "boundflow.rvir-v4-live-return-tamper-report/v1"
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
                raise TypeError("RVIR-v4 live tamper tensor value differs")
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


def _core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return cast(list[dict[str, Any]], payload["whole_core_results"])[0]


def _post(payload: Mapping[str, Any]) -> dict[str, Any]:
    return cast(list[dict[str, Any]], payload["whole_post_results"])[0]


def _resign_receipts(payload: dict[str, Any]) -> None:
    assembly = cast(list[dict[str, Any]], payload["assembly_metadata"])[0]
    assembly_semantic = {
        key: value for key, value in assembly.items() if key != "assembly_hash"
    }
    assembly["assembly_hash"] = artifact_runner._canonical_hash(assembly_semantic)
    receipt = cast(list[dict[str, Any]], payload["commit_receipts"])[0]
    receipt["assembly_hash"] = assembly["assembly_hash"]
    commit_keys = {
        "live_copy_out_hash",
        "committed_path_count",
        "changed_path_count",
        "committed_paths",
        "host_packet_candidate_hash",
        "atomic_live_and_host_commit",
        "provider_callback_count",
        "fallback_dispatch_count",
        "performance_claimed",
    }
    receipt["commit_hash"] = artifact_runner._canonical_hash(
        {key: receipt[key] for key in commit_keys}
    )
    receipt_semantic = {
        key: value for key, value in receipt.items() if key != "live_return_commit_hash"
    }
    receipt["live_return_commit_hash"] = artifact_runner._canonical_hash(
        receipt_semantic
    )


def _forge_recorded_summary(artifact: Path, payload: dict[str, Any]) -> None:
    summary = artifact_runner._load_json(artifact / "summary.json")
    core = _core(payload)
    post = _post(payload)
    assembly = cast(list[dict[str, Any]], payload["assembly_metadata"])[0]
    receipt = cast(list[dict[str, Any]], payload["commit_receipts"])[0]
    summary.update(
        {
            "core_truth_hash": core["truth_hash"],
            "post_truth_hash": post["truth_hash"],
            "branching_decision": core["branching_decision"]["decision"],
            "assembly_hash": assembly["assembly_hash"],
            "commit_hash": receipt["live_return_commit_hash"],
            "atomic_live_and_host_commit": receipt["atomic_live_and_host_commit"],
            "provider_core_callback_count": payload["provider_core_callback_count"],
            "provider_compute_bounds_callback_count": payload[
                "provider_compute_bounds_callback_count"
            ],
            "provider_update_bounds_callback_count": payload[
                "provider_update_bounds_callback_count"
            ],
            "fallback_dispatch_count": payload["fallback_dispatch_count"],
        }
    )
    summary.pop("summary_hash", None)
    summary["summary_hash"] = artifact_runner._canonical_hash(summary)
    artifact_runner._write_json(artifact / "summary.json", summary)
    replay = artifact_runner._replay_result(summary)
    (artifact / "replay_stdout.txt").write_text(
        artifact_runner._canonical_json(replay) + "\n", encoding="utf-8"
    )


def _resign_artifact(artifact: Path, payload: dict[str, Any]) -> None:
    _resign_truth_record(_core(payload))
    _resign_truth_record(_post(payload))
    _resign_receipts(payload)
    torch.save(payload, artifact / artifact_runner.LIVE_RESULT_FILE)
    _forge_recorded_summary(artifact, payload)
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    manifest["files"] = {
        name: artifact_runner._file_sha256(artifact / name)
        for name in artifact_runner.ARTIFACT_FILES
    }
    summary = artifact_runner._load_json(artifact / "summary.json")
    manifest["summary_hash"] = summary["summary_hash"]
    manifest["status"] = summary["status"]
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = artifact_runner._canonical_hash(semantic)
    artifact_runner._write_json(artifact / "manifest.json", manifest)


def _mutate_l_a(payload: dict[str, Any]) -> None:
    record = _core(payload)["branch_trace"]["input"]["lAs"]["_data"]["/48"]
    record["value"] = record["value"].clone()
    record["value"].flatten()[0] += 1.0


def _mutate_intermediate(payload: dict[str, Any]) -> None:
    record = _core(payload)["working_intermediate_bounds"]["/input-28"]["lower"]
    record["value"] = record["value"].clone()
    record["value"].flatten()[0] -= 1.0


def _mutate_candidate_lower(payload: dict[str, Any]) -> None:
    record = _core(payload)["branch_trace"]["candidate_child_lowers"][1]
    record["value"] = record["value"].clone()
    record["value"].flatten()[0] += 1.0


def _mutate_alpha(payload: dict[str, Any]) -> None:
    record = _core(payload)["branch_trace"]["input"]["alphas"]["_data"]["/48"]["/49"]
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


def _mutate_provider_callback(payload: dict[str, Any]) -> None:
    payload["provider_update_bounds_callback_count"] = 1


def _mutate_atomic_flag(payload: dict[str, Any]) -> None:
    cast(list[dict[str, Any]], payload["commit_receipts"])[0][
        "atomic_live_and_host_commit"
    ] = False


ATTACKS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("lA-numeric-full-resign", _mutate_l_a),
    ("intermediate-numeric-full-resign", _mutate_intermediate),
    ("candidate-lower-full-resign", _mutate_candidate_lower),
    ("alpha-state-full-resign", _mutate_alpha),
    ("branch-decision-cross-resign", _mutate_branch_decision),
    ("core-accounting-full-resign", _mutate_core_accounting),
    ("provider-callback-full-resign", _mutate_provider_callback),
    ("atomic-flag-full-resign", _mutate_atomic_flag),
)


def run_probe_suite(
    *, artifact: Path, benchmark_root: Path, abcrown_root: Path, abcrown_python: Path
) -> dict[str, object]:
    """Run one clean fresh candidate and eight completely resigned attacks."""

    _frozen, truth, summary, _result = artifact_runner._verify_static_artifact(artifact)
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-live-fresh-") as raw:
        fresh = artifact_runner._run_worker(
            benchmark=benchmark_root,
            abcrown=abcrown_root,
            python=abcrown_python,
            result=Path(raw) / "fresh.pt",
        )
    clean_fresh_summary = artifact_runner._summary(fresh, truth)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-live-tamper-") as raw:
        workspace = Path(raw)
        for name, mutate in ATTACKS:
            probe = workspace / name
            shutil.copytree(artifact, probe)
            payload = artifact_runner._load_torch(
                probe / artifact_runner.LIVE_RESULT_FILE
            )
            mutate(payload)
            _resign_artifact(probe, payload)
            rejection = ""
            try:
                artifact_runner._verify_static_artifact(probe)
            except (TypeError, ValueError) as error:
                rejection = str(error)
            else:
                raise AssertionError(f"tampered live-return artifact admitted: {name}")
            rows.append(
                {
                    "name": name,
                    "fully_resigned": True,
                    "rejected": True,
                    "rejection": rejection,
                }
            )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_source_git_head": artifact_runner._load_json(
            artifact / "manifest.json"
        )["source_git_head"],
        "clean_frozen_summary_hash": summary["summary_hash"],
        "clean_fresh_semantic_parity": clean_fresh_summary["semantic_parity"],
        "attack_count": len(rows),
        "fully_resigned_attack_count": len(rows),
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
    """Run the tamper suite and write its canonical report."""

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
