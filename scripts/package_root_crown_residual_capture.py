#!/usr/bin/env python3
"""Package or replay the root CROWN residual full-VJP capture."""

# pylint: disable=too-many-locals,too-many-boolean-expressions,too-many-branches

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, cast

import torch

SCHEMA_VERSION = "boundflow.root-crown-residual-capture-artifact/v1"
CODE_PATHS = (
    "boundflow/runtime/root_crown_residual_capture.py",
    "scripts/package_root_crown_residual_capture.py",
    "scripts/probe_root_crown_residual_capture.py",
    "tests/test_root_crown_residual_capture_artifact.py",
)
EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "incoming_lower_a": (3, 1, 16, 8, 8),
    "entry_lower": (1, 16, 8, 8),
    "entry_upper": (1, 16, 8, 8),
    "entry_raw_alpha": (2, 3, 1, 178),
    "main_conv_weight": (16, 16, 3, 3),
    "main_conv_bias": (16,),
    "inner_lower": (1, 16, 8, 8),
    "inner_upper": (1, 16, 8, 8),
    "inner_raw_alpha": (2, 3, 1, 86),
    "inner_conv_weight": (16, 16, 3, 3),
    "inner_conv_bias": (16,),
    "output_lower_a": (3, 1, 16, 8, 8),
    "output_bias": (3, 1),
    "output_lower_a_gradient": (3, 1, 16, 8, 8),
    "output_bias_gradient": (3, 1),
    "incoming_lower_a_gradient": (3, 1, 16, 8, 8),
    "entry_lower_gradient": (1, 16, 8, 8),
    "entry_upper_gradient": (1, 16, 8, 8),
    "entry_raw_alpha_gradient": (2, 3, 1, 178),
    "inner_lower_gradient": (1, 16, 8, 8),
    "inner_upper_gradient": (1, 16, 8, 8),
    "inner_raw_alpha_gradient": (2, 3, 1, 86),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"root residual {name} differs")
    return cast(Mapping[str, Any], value)


def _validate_capture(capture_path: Path, receipt_path: Path) -> dict[str, object]:
    payload = _mapping(
        torch.load(capture_path, map_location="cpu", weights_only=True), "payload"
    )
    receipt = _mapping(json.loads(receipt_path.read_text(encoding="utf-8")), "receipt")
    evaluations = payload.get("evaluations")
    if (
        payload.get("schema_version") != "boundflow.root-crown-residual-tensors/v1"
        or payload.get("performance_claimed") is not False
        or not isinstance(evaluations, list)
        or len(evaluations) != 5
        or receipt.get("schema_version") != "boundflow.root-crown-residual-capture/v1"
        or receipt.get("forward_count") != 5
        or receipt.get("backward_count") != 4
        or receipt.get("performance_claimed") is not False
    ):
        raise ValueError("root residual artifact envelope differs")
    for ordinal, raw_evaluation in enumerate(evaluations):
        evaluation = _mapping(raw_evaluation, "evaluation")
        if evaluation.get("ordinal") != ordinal:
            raise ValueError("root residual evaluation order differs")
        for name, shape in EXPECTED_SHAPES.items():
            value = evaluation.get(name)
            if ordinal == 4 and name.endswith("_gradient"):
                if value is not None:
                    raise ValueError("root residual terminal gradient differs")
                continue
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != shape
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"root residual tensor differs: {ordinal}.{name}")
        for prefix, count in (("entry", 178), ("inner", 86)):
            indices = evaluation.get(f"{prefix}_alpha_feature_indices")
            if not isinstance(indices, tuple) or len(indices) != 3:
                raise ValueError(f"root residual {prefix} indices differ")
            for index, limit in zip(indices, (16, 8, 8), strict=True):
                if (
                    not torch.is_tensor(index)
                    or tuple(index.shape) != (count,)
                    or index.dtype != torch.int64
                    or not bool(((index >= 0) & (index < limit)).all().item())
                ):
                    raise ValueError(f"root residual {prefix} index tensor differs")
            coordinates = torch.stack(indices, dim=1)
            if int(torch.unique(coordinates, dim=0).shape[0]) != count:
                raise ValueError(f"root residual {prefix} coordinates differ")
        for name in ("entry", "inner"):
            lower = cast(torch.Tensor, evaluation[f"{name}_lower"])
            upper = cast(torch.Tensor, evaluation[f"{name}_upper"])
            alpha = cast(torch.Tensor, evaluation[f"{name}_raw_alpha"])
            if bool((lower > upper).any().item()) or bool(
                ((alpha < 0) | (alpha > 1)).any().item()
            ):
                raise ValueError(f"root residual {name} legality differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_count": 5,
        "forward_count": 5,
        "backward_count": 4,
        "tensor_field_count": len(EXPECTED_SHAPES),
        "entry_alpha_feature_count": 178,
        "inner_alpha_feature_count": 86,
        "spec_count": 3,
        "domain_count": 1,
        "full_vjp_output_count": 7,
        "performance_claimed": False,
        "next_action": "parameterize-residual-tir-spec-domain-and-full-vjp",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _pack(args: argparse.Namespace) -> None:
    summary = _validate_capture(args.capture, args.receipt)
    args.artifact.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.capture, args.artifact / "capture.pt")
    shutil.copyfile(args.receipt, args.artifact / "receipt.json")
    _write_json(args.artifact / "summary.json", summary)
    code_revision = {name: _sha256(args.repository / name) for name in CODE_PATHS}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_parent_revision": args.source_parent_revision,
        "code_revision": code_revision,
        "files": {
            name: _sha256(args.artifact / name)
            for name in ("capture.pt", "receipt.json", "summary.json")
        },
        "model_sha256": (
            "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
        ),
        "property_sha256": (
            "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
        ),
        "performance_claimed": False,
    }
    _write_json(args.artifact / "manifest.json", manifest)
    _replay(args.artifact, args.repository, check_code=True)


def _replay(artifact: Path, repository: Path, *, check_code: bool) -> None:
    manifest = _mapping(
        json.loads((artifact / "manifest.json").read_text(encoding="utf-8")),
        "manifest",
    )
    for name, expected in _mapping(manifest.get("files"), "files").items():
        if _sha256(artifact / str(name)) != expected:
            raise ValueError(f"root residual artifact file differs: {name}")
    if check_code:
        for name, expected in _mapping(
            manifest.get("code_revision"), "code revision"
        ).items():
            if _sha256(repository / str(name)) != expected:
                raise ValueError(f"root residual code revision differs: {name}")
    derived = _validate_capture(artifact / "capture.pt", artifact / "receipt.json")
    stored = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
    if derived != stored:
        raise ValueError("root residual summary differs")
    print(json.dumps(derived, sort_keys=True, separators=(",", ":")))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack")
    pack.add_argument("--capture", type=Path, required=True)
    pack.add_argument("--receipt", type=Path, required=True)
    pack.add_argument("--artifact", type=Path, required=True)
    pack.add_argument("--repository", type=Path, default=Path.cwd())
    pack.add_argument("--source-parent-revision", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact", type=Path, required=True)
    replay.add_argument("--repository", type=Path, default=Path.cwd())
    replay.add_argument("--skip-code-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected package or replay command."""

    args = _parse_args()
    repository = args.repository.resolve()
    if args.command == "pack":
        args.repository = repository
        _pack(args)
    else:
        _replay(args.artifact, repository, check_code=not args.skip_code_check)


if __name__ == "__main__":
    main()
