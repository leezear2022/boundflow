#!/usr/bin/env python3
"""Probe coordinated all-run rewrites against B4-B1 numerical replay."""

# pylint: disable=protected-access,missing-function-docstring,import-error
# pylint: disable=too-many-locals

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Callable, cast, Mapping

import torch

from boundflow.runtime.fsg4_b4b1_reference_capture import (
    ProductionDifferentiableReferenceCaptureV1,
    production_differentiable_reference_capture_from_payload_v1,
    production_differentiable_reference_capture_to_payload_v1,
)
from boundflow.runtime.fsg4_b4b_production_region_capture import CapturedCudaTensorV1
from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_fsg4_b4b1_pytorch_reference_artifact as reference_artifact
from scripts import run_fsg4_b4b1_reference_five_fresh_artifact as source_artifact

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "boundflow.fsg4-b4b1-pytorch-reference-integrity/v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_snapshot(
    snapshot: CapturedCudaTensorV1, value: torch.Tensor
) -> CapturedCudaTensorV1:
    raw = value.detach().contiguous()
    return replace(
        snapshot,
        value=raw,
        content_sha256=production_tensor_sha256(raw),
    )


def _incoming_bias_rewrite(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> ProductionDifferentiableReferenceCaptureV1:
    return replace(
        capture,
        incoming_lower_bias=_replace_snapshot(
            capture.incoming_lower_bias, capture.incoming_lower_bias.value + 0.125
        ),
    )


def _output_adjoint_rewrite(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> ProductionDifferentiableReferenceCaptureV1:
    changed = capture.output_lower_a_gradient.value.clone()
    changed.reshape(-1)[0] += 0.25
    return replace(
        capture,
        output_lower_a_gradient=_replace_snapshot(
            capture.output_lower_a_gradient, changed
        ),
    )


CASES: tuple[
    tuple[
        str,
        Callable[
            [ProductionDifferentiableReferenceCaptureV1],
            ProductionDifferentiableReferenceCaptureV1,
        ],
    ],
    ...,
] = (
    ("all-run-incoming-bias-fully-resigned", _incoming_bias_rewrite),
    ("all-run-output-adjoint-fully-resigned", _output_adjoint_rewrite),
)


def _resign_source_artifact(
    source: Path,
    target: Path,
    mutation: Callable[
        [ProductionDifferentiableReferenceCaptureV1],
        ProductionDifferentiableReferenceCaptureV1,
    ],
) -> None:
    shutil.copytree(source, target)
    runs: list[dict[str, object]] = []
    for filename in source_artifact.RUN_FILES:
        payload = source_artifact._load_torch(target / filename)
        raw_captures = cast(list[object], payload["captures"])
        captures = [
            production_differentiable_reference_capture_from_payload_v1(
                cast(Mapping[str, object], raw)
            )
            for raw in raw_captures
        ]
        mutated = [mutation(capture) for capture in captures]
        for capture in mutated:
            capture.validate()
        payload["captures"] = [
            production_differentiable_reference_capture_to_payload_v1(capture)
            for capture in mutated
        ]
        torch.save(payload, target / filename)
        runs.append(cast(dict[str, object], payload))
    protocol = source_artifact._load_json(target / "protocol.json")
    summary = source_artifact._summary(cast(list[Mapping[str, object]], runs), protocol)
    source_artifact._write_json(target / "summary.json", summary)
    result = source_artifact._result(summary)
    (target / "replay_stdout.txt").write_text(
        source_artifact._canonical_json(result) + "\n", encoding="utf-8"
    )
    (target / "README.md").write_text(source_artifact._readme(), encoding="utf-8")
    manifest = source_artifact._load_json(target / "manifest.json")
    manifest["summary_hash"] = summary["summary_hash"]
    manifest["files"] = source_artifact._all_files(target)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = source_artifact._canonical_hash(manifest)
    source_artifact._write_json(target / "manifest.json", manifest)
    source_artifact._verify_static_artifact(target)


def _probe(source: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="fsg4-b4b1-reference-integrity-") as root:
        temporary = Path(root)
        for name, mutation in CASES:
            mutated = temporary / name
            _resign_source_artifact(source, mutated, mutation)
            protocol = reference_artifact._protocol(mutated)
            rejected = False
            error = ""
            try:
                reference_artifact._records_from_source(mutated, protocol)
            except ValueError as exception:
                error = str(exception)
                rejected = "numerical semantics differ" in error
            rows.append(
                {
                    "case": name,
                    "all_runs_rewritten": True,
                    "inner_capture_hashes_resigned": True,
                    "source_summary_resigned": True,
                    "source_manifest_resigned": True,
                    "derived_protocol_resigned": True,
                    "rejected_by_numerical_reference": rejected,
                    "error": error,
                }
            )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "source_git_head": reference_artifact._git("rev-parse", "HEAD"),
        "reference_code_revision": reference_artifact._code_revision(),
        "source_artifact_manifest_hash": source_artifact._load_json(
            source / "manifest.json"
        )["manifest_hash"],
        "probe_code_sha256": _file_sha256(Path(__file__)),
        "case_count": len(rows),
        "rejected_count": sum(
            row["rejected_by_numerical_reference"] is True for row in rows
        ),
        "rows": rows,
        "performance_claimed": False,
        "tir_admitted": False,
    }
    report["report_hash"] = reference_artifact._canonical_hash(report)
    if report["case_count"] != report["rejected_count"]:
        raise ValueError("FSG4/B4-B1 coordinated rewrite escaped reference replay")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = _probe(args.capture_artifact.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    reference_artifact._write_json(args.report, report)
    print(reference_artifact._canonical_json(report), flush=True)


if __name__ == "__main__":
    main()
