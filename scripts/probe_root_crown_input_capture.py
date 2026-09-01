#!/usr/bin/env python3
"""Capture one production root CROWN input Conv/domain transaction."""

# pylint: disable=import-error,import-outside-toplevel,wrong-import-position

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.root_crown_input_capture import (  # noqa: E402
    RootCrownInputCaptureV1,
)


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch
    from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints
    from auto_LiRPA import BoundedModule

    if not torch.cuda.is_available():
        raise RuntimeError("root CROWN input capture requires CUDA")
    capture = RootCrownInputCaptureV1()
    with tempfile.TemporaryDirectory(prefix="boundflow-root-input-") as raw:
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
            .set("solver/beta-crown/iteration", 10)
        )
        with capture.install(BoundedModule):
            result = ABCrownSolver(str(args.model), config=config).verify(
                constraints=IOConstraints(vnnlib_path=str(isolated_property))
            )
    capture.validate()
    args.tensor_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "boundflow.root-crown-input-tensors/v1",
            "evaluations": [item.tensor_payload() for item in capture.evaluations],
            "performance_claimed": False,
        },
        args.tensor_out,
    )
    receipt = capture.shape_receipt()
    receipt["solver_status"] = str(result.status)
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
    """Run one fresh production capture."""

    _worker(_parse_args())


if __name__ == "__main__":
    main()
