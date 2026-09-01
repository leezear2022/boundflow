#!/usr/bin/env python3
"""Fully re-sign semantic tampering against the R3-1b2 artifact replay."""

# pylint: disable=wrong-import-position,missing-function-docstring,protected-access

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_r3_compiled_p_alpha_vjp_artifact as artifact


def _load_record(path: Path) -> dict[str, Any]:
    value = json.loads((path / "raw.jsonl").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("R3-1b2 tamper raw root differs")
    return value


def _unsafe_resign(path: Path, record: dict[str, Any]) -> None:
    for name in ("lower_metric", "gradient_metric"):
        metric = record[name]
        candidate = tuple(float(value) for value in metric["candidate"])
        native = tuple(float(value) for value in metric["native"])
        metric["candidate_sha256"] = artifact._float32_hash(candidate)
        metric["native_sha256"] = artifact._float32_hash(native)
        metric["max_abs_diff"] = max(
            abs(left - right) for left, right in zip(candidate, native)
        )
        metric["sign_exact"] = all(
            (left > 0) - (left < 0) == (right > 0) - (right < 0)
            for left, right in zip(candidate, native)
        )
        metric["candidate_nonzero"] = sum(value != 0 for value in candidate)
        metric["native_nonzero"] = sum(value != 0 for value in native)
        metric["allclose"] = metric["max_abs_diff"] <= 2e-4
        metric["finite"] = True
    (path / "raw.jsonl").write_text(
        artifact._canonical(record) + "\n", encoding="utf-8"
    )
    lower = record["lower_metric"]
    gradient = record["gradient_metric"]
    receipt = record["execution_receipt"]
    summary = artifact._load_json(path / "summary.json")
    summary.update(
        trace_hash=record["trace_hash"],
        production_plan_hash=record["production_plan_hash"],
        b1_module_hash=receipt["b1_module_hash"],
        b2_module_hash=receipt["b2_module_hash"],
        b2_device_source_hash=receipt["b2_device_source_hash"],
        candidate_lower_sha256=lower["candidate_sha256"],
        candidate_gradient_sha256=gradient["candidate_sha256"],
        native_gradient_sha256=gradient["native_sha256"],
        lower_max_abs_diff=lower["max_abs_diff"],
        gradient_max_abs_diff=gradient["max_abs_diff"],
        lower_sign_exact=lower["sign_exact"],
        gradient_sign_exact=gradient["sign_exact"],
        gradient_nonzero=gradient["candidate_nonzero"],
        b1_forward_launch_count=receipt["b1_forward_launch_count"],
        b1_backward_launch_count=receipt["b1_backward_launch_count"],
        b2_launch_count=receipt["b2_launch_count"],
        coefficient_scratch_count=receipt["coefficient_scratch_count"],
        sign_bitmap_count=receipt["sign_bitmap_count"],
        sign_bitmap_bytes=receipt["sign_bitmap_bytes"],
        saved_dense_a_count=receipt["saved_dense_a_count"],
        warm_dynamic_allocated_bytes=receipt["warm_dynamic_allocated_bytes"],
        dlpack_pointer_count=receipt["dlpack_pointer_count"],
        dlpack_pointer_exact_count=receipt["dlpack_pointer_exact_count"],
        compiled_vjp=receipt["compiled_vjp"],
        custom_vjp=record["custom_vjp"],
    )
    summary.pop("summary_hash", None)
    summary["summary_hash"] = artifact._hash(summary)
    artifact._write_json(path / "summary.json", summary)
    (path / "replay_stdout.txt").write_text(
        artifact._canonical(artifact._result(summary)) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": artifact.ARTIFACT_SCHEMA,
        "files": {
            name: artifact._file_hash(path / name) for name in artifact.SIGNED_FILES
        },
    }
    manifest["manifest_hash"] = artifact._hash(manifest)
    artifact._write_json(path / "manifest.json", manifest)


def _mutations() -> tuple[tuple[str, Callable[[dict[str, Any]], None]], ...]:
    return (
        (
            "candidate-lower",
            lambda record: record["lower_metric"]["candidate"].__setitem__(0, 0.0),
        ),
        (
            "candidate-gradient",
            lambda record: record["gradient_metric"]["candidate"].__setitem__(0, 1.0),
        ),
        (
            "native-gradient",
            lambda record: record["gradient_metric"]["native"].__setitem__(0, 1.0),
        ),
        (
            "module-hash",
            lambda record: record["execution_receipt"].__setitem__(
                "b2_module_hash", "0" * 64
            ),
        ),
        (
            "launch-count",
            lambda record: record["execution_receipt"].__setitem__(
                "b2_launch_count", 9
            ),
        ),
        (
            "scratch-count",
            lambda record: record["execution_receipt"].__setitem__(
                "coefficient_scratch_count", 3
            ),
        ),
        (
            "sign-bytes",
            lambda record: record["execution_receipt"].__setitem__(
                "sign_bitmap_bytes", 43009
            ),
        ),
        (
            "saved-dense-a",
            lambda record: record["execution_receipt"].__setitem__(
                "saved_dense_a_count", 1
            ),
        ),
        (
            "warm-allocation",
            lambda record: record["execution_receipt"].__setitem__(
                "warm_dynamic_allocated_bytes", 4
            ),
        ),
        (
            "compiled-vjp",
            lambda record: record["execution_receipt"].__setitem__(
                "compiled_vjp", False
            ),
        ),
        (
            "custom-vjp",
            lambda record: record.__setitem__("custom_vjp", False),
        ),
        (
            "dlpack-exact",
            lambda record: record["execution_receipt"].__setitem__(
                "dlpack_pointer_exact_count", 78
            ),
        ),
    )


def probe(source: Path) -> dict[str, object]:
    rejected: list[str] = []
    with tempfile.TemporaryDirectory(prefix="r31b2-tamper-") as temporary:
        root = Path(temporary)
        original = _load_record(source)
        for name, mutate in _mutations():
            target = root / name
            shutil.copytree(source, target)
            changed = copy.deepcopy(original)
            mutate(changed)
            try:
                _unsafe_resign(target, changed)
                artifact.replay(target)
            except (ValueError, TypeError, KeyError):
                rejected.append(name)
            else:
                raise RuntimeError(f"R3-1b2 tamper was accepted: {name}")
    return {
        "status": "tamper-passed",
        "attack_count": len(_mutations()),
        "rejected_count": len(rejected),
        "rejected": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    print(artifact._canonical(probe(args.artifact)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
