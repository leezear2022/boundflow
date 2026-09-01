#!/usr/bin/env python3
"""Run one fresh control or compiled root-residual solver process."""

# pylint: disable=import-error,wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-locals,protected-access,missing-function-docstring
# pylint: disable=duplicate-code
# mypy: disable-error-code=import-untyped

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.root_crown_residual import (  # noqa: E402
    RootCrownResidualTemplateV1,
)
from boundflow.runtime.root_crown_residual_live import (  # noqa: E402
    RootCrownResidualLiveBridgeV1,
)
from scripts import run_asplos27_s4_same_solver_worker as s4_worker  # noqa: E402

DEFAULT_CAPTURE = (
    REPOSITORY_ROOT
    / "artifacts/root-crown-residual-capture/resnet2b-prop0-v1/capture.pt"
)


def _coordinates(values: Sequence[Any]) -> tuple[tuple[int, int, int], ...]:
    import torch

    if len(values) != 3 or not all(torch.is_tensor(value) for value in values):
        raise ValueError("root residual worker coordinate payload differs")
    tensors = [value for value in values if torch.is_tensor(value)]
    lengths = {int(value.numel()) for value in tensors}
    if len(lengths) != 1:
        raise ValueError("root residual worker coordinate length differs")
    return cast(
        tuple[tuple[int, int, int], ...],
        tuple(
            tuple(int(tensors[axis][ordinal]) for axis in range(3))
            for ordinal in range(int(tensors[0].numel()))
        ),
    )


def _template(capture_path: Path) -> RootCrownResidualTemplateV1:
    import torch

    payload = torch.load(capture_path, map_location="cpu", weights_only=True)
    evaluations = payload.get("evaluations")
    if (
        payload.get("schema_version") != "boundflow.root-crown-residual-tensors/v1"
        or not isinstance(evaluations, list)
        or len(evaluations) != 5
    ):
        raise ValueError("root residual worker capture differs")
    first = evaluations[0]
    incoming = first["incoming_lower_a"]
    major, minor = torch.cuda.get_device_capability()
    return RootCrownResidualTemplateV1(
        spec_count=int(incoming.shape[0]),
        domain_count=int(incoming.shape[1]),
        channels=int(incoming.shape[2]),
        height=int(incoming.shape[3]),
        width=int(incoming.shape[4]),
        entry_alpha_coordinates=_coordinates(first["entry_alpha_feature_indices"]),
        inner_alpha_coordinates=_coordinates(first["inner_alpha_feature_indices"]),
        compute_capability=f"sm_{major}{minor}",
        thread_extent=128,
    )


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    from auto_LiRPA import BoundedModule  # type: ignore[import-not-found,import-untyped]

    bridge = None
    compile_ns = 0
    if args.mode == "candidate":
        started = time.perf_counter_ns()
        bridge = RootCrownResidualLiveBridgeV1(_template(args.capture))
        bridge.executor.prepare()
        compile_ns = time.perf_counter_ns() - started
    with tempfile.TemporaryDirectory(prefix="boundflow-root-residual-live-") as raw:
        base_result = Path(raw) / "base.json"
        namespace = argparse.Namespace(
            configuration="S4-PREP",
            mode="control",
            run_id=args.run_id,
            block_index=args.block_index,
            sequence_position=args.sequence_position,
            benchmark_root=args.benchmark_root,
            abcrown_root=args.abcrown_root,
            model=args.model,
            property=args.property,
            result=base_result,
            attribute_root_incomplete=True,
        )
        if bridge is None:
            s4_worker._worker(namespace)
        else:
            with bridge.install(BoundedModule):
                s4_worker._worker(namespace)
        base = json.loads(base_result.read_text(encoding="utf-8"))
    base["root_residual_mode"] = args.mode
    base["root_residual_compile_ns"] = compile_ns
    base["root_residual_compile_excluded_from_query"] = True
    base["root_residual_receipt"] = None if bridge is None else bridge.receipt()
    base["performance_claimed"] = False
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(base, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics = base["run"]["metrics"]
    root_timings = base["diagnostics"]["root_incomplete_timings"]
    root_wall_ns = root_timings["aggregates"]["root_incomplete"]["inclusive_ns"]
    print(
        json.dumps(
            {
                "mode": args.mode,
                "query_wall_ns": metrics["query_wall_ns"],
                "root_incomplete_wall_ns": root_wall_ns,
                "performance_claimed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "candidate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--sequence-position", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    _worker(_parse_args())


if __name__ == "__main__":
    main()
