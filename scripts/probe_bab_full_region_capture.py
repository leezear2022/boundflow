#!/usr/bin/env python3
"""Capture one real activation-BaB full CROWN transaction on CUDA."""

# pylint: disable=import-error,import-outside-toplevel,wrong-import-position
# pylint: disable=too-many-locals

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.bab_full_region_capture import (  # noqa: E402
    BAB_OPTIMIZER_ITERATIONS,
    BabInputCaptureV1,
    BabProjectionCaptureV1,
    BabResidualCaptureV1,
    BabTerminalCaptureV1,
)

SCHEMA_VERSION = "boundflow.activation-bab-full-region-tensors/v1"
OPTIMIZER_ITERATIONS = BAB_OPTIMIZER_ITERATIONS


def _visited_domains(result: Any) -> list[int]:
    value = getattr(result, "visited_domains", ())
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import (  # type: ignore[import-not-found]
        ABCrownSolver,
        ConfigBuilder,
        IOConstraints,
    )
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped]

    if not torch.cuda.is_available():
        raise RuntimeError("activation-BaB full-region capture requires CUDA")
    terminal = BabTerminalCaptureV1()
    residual = BabResidualCaptureV1()
    projection = BabProjectionCaptureV1()
    input_domain = BabInputCaptureV1()
    captures: dict[str, Any] = {
        "terminal": terminal,
        "residual": residual,
        "projection": projection,
        "input_domain": input_domain,
    }
    with tempfile.TemporaryDirectory(prefix="boundflow-bab-full-region-") as raw:
        isolated_property = Path(raw) / args.property.name
        shutil.copy2(args.property, isolated_property)
        config = (
            ConfigBuilder.from_defaults()
            .set("general/device", "cuda")
            .set("general/seed", 100)
            .set("general/reset_seed_after_precompile", True)
            .set("general/complete_verifier", "bab")
            .set("attack/pgd_order", "skip")
            .set("bab/timeout", 60)
            .set("bab/max_iterations", 1)
            .set("solver/batch_size", 64)
            .set("solver/auto_enlarge_batch_size", False)
            .set("solver/alpha-crown/iteration", 5)
            .set("solver/beta-crown/iteration", OPTIMIZER_ITERATIONS)
        )
        with ExitStack() as stack:
            for capture in captures.values():
                stack.enter_context(capture.install(BoundedModule))
            result = ABCrownSolver(str(args.model), config=config).verify(
                constraints=IOConstraints(vnnlib_path=str(isolated_property))
            )
    for capture in captures.values():
        capture.validate()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "optimizer_iterations": OPTIMIZER_ITERATIONS,
        "optimizer_mutations": OPTIMIZER_ITERATIONS - 1,
        "segments": {
            name: {
                "receipt": capture.shape_receipt(),
                "evaluations": [
                    evaluation.tensor_payload() for evaluation in capture.evaluations
                ],
                "beta_evidence": (
                    [item.tensor_payload() for item in terminal.beta_evidence]
                    if name == "terminal"
                    else []
                ),
            }
            for name, capture in captures.items()
        },
        "solver": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": _visited_domains(result),
        },
        "active_beta_captured": True,
        "candidate_executed": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    args.tensor_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.tensor_out)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "optimizer_iterations": OPTIMIZER_ITERATIONS,
        "optimizer_mutations": OPTIMIZER_ITERATIONS - 1,
        "segment_receipts": {
            name: capture.shape_receipt() for name, capture in captures.items()
        },
        "solver": payload["solver"],
        "active_beta_captured": True,
        "candidate_executed": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--tensor-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Capture the first real activation-BaB transaction."""

    _worker(_parse_args())


if __name__ == "__main__":
    main()
