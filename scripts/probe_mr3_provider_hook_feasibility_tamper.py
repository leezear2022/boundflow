#!/usr/bin/env python3
"""Run fully re-signed semantic attacks against an MR3-0 artifact."""

# pylint: disable=missing-function-docstring,import-error,wrong-import-position

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash  # noqa: E402
from scripts.run_mr3_provider_hook_feasibility import replay  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("MR3-0 tamper JSON root differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resign_run(run: dict[str, Any]) -> None:
    run["outer_result_hash"] = canonical_hash(run.get("outer_result_state"))
    run["inner_result_hashes"] = [
        canonical_hash(state) for state in run.get("inner_result_states", [])
    ]
    run["final_target_alpha_hash"] = canonical_hash(run.get("final_target_alpha_state"))
    run["final_module_state_hash"] = canonical_hash(run.get("final_module_state"))
    unsigned = dict(run)
    unsigned.pop("worker_hash", None)
    run["worker_hash"] = canonical_hash(unsigned)


def _attack_source(raw: dict[str, Any]) -> None:
    raw["runs"][0]["source"]["abcrown_commit"] = "0" * 40


def _attack_run_order(raw: dict[str, Any]) -> None:
    raw["runs"][0]["position"] = 1


def _attack_outer_count(raw: dict[str, Any]) -> None:
    raw["runs"][1]["outer_beta_exact_call_count"] = 2


def _attack_beta_numel(raw: dict[str, Any]) -> None:
    raw["runs"][1]["hook"]["evaluations"][0]["target_beta_numel"] = 1


def _attack_beta_object_count(raw: dict[str, Any]) -> None:
    raw["runs"][1]["hook"]["evaluations"][0]["target_beta_tensor_count"] = 0


def _attack_adjacency(raw: dict[str, Any]) -> None:
    raw["runs"][1]["hook"]["evaluations"][4]["conv_input_lower_a"]["content_sha256"] = (
        "f" * 64
    )


def _attack_alpha_shape(raw: dict[str, Any]) -> None:
    raw["runs"][1]["hook"]["evaluations"][2]["compressed_alpha"]["shape"][-1] = 85


def _attack_stream(raw: dict[str, Any]) -> None:
    raw["runs"][1]["hook"]["stream_after"] += 1


def _attack_numeric_result(raw: dict[str, Any]) -> None:
    raw["runs"][0]["outer_result_state"][0]["values"][0] += 0.01


def _attack_numeric_alpha(raw: dict[str, Any]) -> None:
    raw["runs"][0]["final_target_alpha_state"]["values"][0] += 0.01


def _attack_replacement_count(raw: dict[str, Any]) -> None:
    raw["runs"][1]["hook"]["counters"]["replacement_count"] = 1


def _attack_missing_hook(raw: dict[str, Any]) -> None:
    raw["runs"][1]["hook"] = None


ATTACKS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("source_commit", _attack_source),
    ("run_order", _attack_run_order),
    ("outer_count", _attack_outer_count),
    ("beta_numel", _attack_beta_numel),
    ("beta_object_count", _attack_beta_object_count),
    ("relu_conv_adjacency", _attack_adjacency),
    ("alpha_shape", _attack_alpha_shape),
    ("cuda_stream", _attack_stream),
    ("outer_numeric_result", _attack_numeric_result),
    ("target_alpha_numeric", _attack_numeric_alpha),
    ("replacement_count", _attack_replacement_count),
    ("missing_probe_hook", _attack_missing_hook),
)


def _resign_artifact(artifact: Path, raw: dict[str, Any]) -> None:
    for run in raw["runs"]:
        _resign_run(run)
    _write(artifact / "raw.json", raw)
    manifest = _load(artifact / "manifest.json")
    manifest["files"]["raw.json"] = _sha256(artifact / "raw.json")
    unsigned = dict(manifest)
    unsigned.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_hash(unsigned)
    _write(artifact / "manifest.json", manifest)


def run_attacks(artifact: Path) -> dict[str, object]:
    original_raw = _load(artifact / "raw.json")
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-mr3-hook-tamper-") as workspace:
        root = Path(workspace)
        for ordinal, (name, attack) in enumerate(ATTACKS):
            candidate = root / f"attack_{ordinal:02d}"
            shutil.copytree(artifact, candidate)
            raw = deepcopy(original_raw)
            attack(raw)
            _resign_artifact(candidate, raw)
            rejected = False
            error = ""
            try:
                replay(candidate)
            except (ValueError, TypeError, KeyError) as caught:
                rejected = True
                error = str(caught)
            results.append({"attack": name, "rejected": rejected, "error": error})
    result: dict[str, object] = {
        "status": "validated",
        "attack_count": len(results),
        "rejected_count": sum(bool(row["rejected"]) for row in results),
        "all_rejected": all(bool(row["rejected"]) for row in results),
        "results": results,
        "performance_claimed": False,
    }
    result["result_hash"] = canonical_hash(result)
    if not result["all_rejected"]:
        raise RuntimeError("MR3-0 fully re-signed tamper attack escaped")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_attacks(args.artifact_dir.resolve())
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
        if args.output.resolve().parent == args.artifact_dir.resolve():
            manifest_path = args.artifact_dir.resolve() / "manifest.json"
            manifest = _load(manifest_path)
            manifest["files"][args.output.name] = _sha256(args.output.resolve())
            unsigned = dict(manifest)
            unsigned.pop("manifest_hash", None)
            manifest["manifest_hash"] = canonical_hash(unsigned)
            _write(manifest_path, manifest)
    print(encoded)


if __name__ == "__main__":
    main()
