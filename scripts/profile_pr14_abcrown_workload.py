#!/usr/bin/env python3
"""Capture coverage profiles from an unmodified αβ-CROWN ONNX/VNNLIB run."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.runtime.abcrown_adapter import (
    ABCrownBoundQueryProfiler,
    file_sha256,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vnnlib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-name", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--skip-attack", action="store_true")
    parser.add_argument("--baseline-first", action="store_true")
    parser.add_argument(
        "--complete-verifier", choices=("auto", "bab", "bab-refine", "input_bab")
    )
    return parser.parse_args()


def _require_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    return resolved


def _git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _result_summary(result: Any) -> dict[str, Any]:
    """Keep solver status and stats while normalizing external objects."""

    return {
        "status": getattr(result, "status", None),
        "success": getattr(result, "success", None),
        "stats": _json_value(getattr(result, "stats", None)),
    }


def _visited_domains(result: Any) -> list[int]:
    """Extract per-instance BaB counts without comparing wall-clock fields."""

    stats = getattr(result, "stats", None)
    if not isinstance(stats, dict) or not isinstance(stats.get("bab"), list):
        return []
    return [
        int(row[2])
        for row in stats["bab"]
        if isinstance(row, (tuple, list)) and len(row) >= 3
    ]


def _audit_boundflow_import(model: Path) -> dict[str, Any]:
    """Run the local ONNX frontend and preserve a fail-closed reason."""

    try:
        program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    except Exception as error:  # pylint: disable=broad-exception-caught
        message = str(error)
        unsupported_prefix = "unsupported ONNX op: "
        unsupported_op = (
            message.removeprefix(unsupported_prefix)
            if message.startswith(unsupported_prefix)
            else None
        )
        reason = (
            f"onnx_frontend_unsupported_op:{unsupported_op}"
            if unsupported_op
            else f"onnx_frontend_import_failed:{type(error).__name__}"
        )
        return {
            "supported": False,
            "reason": reason,
            "error_type": type(error).__name__,
            "error": message,
            "layer_pattern": [],
        }
    return {
        "supported": True,
        "reason": None,
        "error_type": None,
        "error": None,
        "layer_pattern": [node.op_type for node in program.graph.nodes],
    }


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Run one official verifier instance and persist the coverage artifacts."""

    args = _parse_args()
    abcrown_root = args.abcrown_root.expanduser().resolve()
    complete_verifier = abcrown_root / "complete_verifier"
    auto_lirpa = abcrown_root / "auto_LiRPA"
    if not (complete_verifier / "abcrown.py").is_file():
        raise FileNotFoundError(f"invalid αβ-CROWN checkout: {abcrown_root}")
    if not (auto_lirpa / "auto_LiRPA" / "__init__.py").is_file():
        raise FileNotFoundError(
            "αβ-CROWN auto_LiRPA submodule is absent; clone/update it recursively"
        )
    model = _require_file(args.model, "ONNX model")
    vnnlib = _require_file(args.vnnlib, "VNNLIB property")
    config_path = None if args.config is None else _require_file(args.config, "config")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")

    sys.path.insert(0, str(auto_lirpa))
    # Import the packaged API from ``<root>/abcrown``. Its initializer adds
    # ``complete_verifier`` for the upstream modules' historical flat imports.
    sys.path.insert(0, str(abcrown_root))
    abcrown = importlib.import_module("abcrown")
    auto_lirpa_module = importlib.import_module("auto_LiRPA")
    abcrown_solver = abcrown.ABCrownSolver
    config_builder = abcrown.ConfigBuilder
    io_constraints = abcrown.IOConstraints
    bounded_module = auto_lirpa_module.BoundedModule

    config = (
        config_builder.from_defaults()
        if config_path is None
        else config_builder.from_yaml(str(config_path))
    )
    overrides: dict[str, object] = {}
    if args.device is not None:
        config.set("general/device", args.device)
        overrides["general/device"] = args.device
    if args.timeout is not None:
        config.set("bab/timeout", args.timeout)
        overrides["bab/timeout"] = args.timeout
    if args.complete_verifier is not None:
        config.set("general/complete_verifier", args.complete_verifier)
        overrides["general/complete_verifier"] = args.complete_verifier
    if args.skip_attack:
        config.set("attack/pgd_order", "skip")
        overrides["attack/pgd_order"] = "skip"

    model_hash = file_sha256(model)
    boundflow_import = _audit_boundflow_import(model)
    profiler = ABCrownBoundQueryProfiler(
        model_structure_hash=f"onnx:{model_hash}",
        weight_version=f"onnx:{model_hash}",
        query_prefix=args.workload_name,
        precondition_rejections=(
            () if boundflow_import["supported"] else (str(boundflow_import["reason"]),)
        ),
    )
    baseline_result = None
    if args.baseline_first:
        baseline_solver = abcrown_solver(str(model), config=config.copy())
        baseline_result = baseline_solver.verify(
            constraints=io_constraints(vnnlib_path=str(vnnlib))
        )

    constraints = io_constraints(vnnlib_path=str(vnnlib))
    solver = abcrown_solver(str(model), config=config.copy())
    with profiler.instrument(bounded_module):
        result = solver.verify(constraints=constraints)

    profiler.write_artifacts(output_dir)
    manifest = {
        "schema_version": "boundflow.pr14-abcrown-workload/v1",
        "workload_name": args.workload_name,
        "abcrown_root": str(abcrown_root),
        "abcrown_commit": _git_revision(abcrown_root),
        "model": str(model),
        "model_sha256": model_hash,
        "vnnlib": str(vnnlib),
        "vnnlib_sha256": file_sha256(vnnlib),
        "config": None if config_path is None else str(config_path),
        "config_sha256": None if config_path is None else file_sha256(config_path),
        "config_overrides": overrides,
        "boundflow_import": boundflow_import,
        "command": sys.argv,
        "result": _result_summary(result),
        "baseline_result": (
            None if baseline_result is None else _result_summary(baseline_result)
        ),
        "baseline_comparison": (
            None
            if baseline_result is None
            else {
                "status_match": getattr(baseline_result, "status", None)
                == getattr(result, "status", None),
                "visited_domains_match": _visited_domains(baseline_result)
                == _visited_domains(result),
                "baseline_visited_domains": _visited_domains(baseline_result),
                "profiled_visited_domains": _visited_domains(result),
            }
        ),
        "query_count": len(profiler.queries),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["result"]["status"],
                "query_count": manifest["query_count"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
