#!/usr/bin/env python3
"""Run one fresh S4-PREP control or compiled root-terminal candidate."""

# pylint: disable=import-error,wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-locals,protected-access
# mypy: disable-error-code=import-untyped

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.root_crown_terminal_linear import (  # noqa: E402
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_terminal_live import (  # noqa: E402
    RootCrownTerminalLiveBridgeV1,
)
from scripts import run_asplos27_s4_same_solver_worker as s4_worker  # noqa: E402

FEATURE_INDICES = (
    0,
    1,
    3,
    4,
    6,
    13,
    17,
    20,
    24,
    27,
    29,
    30,
    31,
    32,
    42,
    45,
    46,
    58,
    64,
    65,
    75,
    78,
    86,
    88,
    89,
    90,
    93,
)


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch
    from auto_LiRPA import BoundedModule

    bridge = None
    compile_ns = 0
    if args.mode == "candidate":
        major, minor = torch.cuda.get_device_capability()
        template = RootCrownTerminalLinearTemplateV1(
            spec_count=3,
            domain_count=1,
            current_features=100,
            previous_features=1024,
            alpha_feature_indices=FEATURE_INDICES,
            compute_capability=f"sm_{major}{minor}",
            thread_extent=128,
        )
        started = time.perf_counter_ns()
        bridge = RootCrownTerminalLiveBridgeV1(
            template, capture_debug=args.debug_tensors is not None
        )
        compile_ns = time.perf_counter_ns() - started
    with tempfile.TemporaryDirectory(prefix="boundflow-root-live-") as raw:
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
    base["root_terminal_mode"] = args.mode
    base["root_terminal_compile_ns"] = compile_ns
    base["root_terminal_compile_excluded_from_query"] = True
    base["root_terminal_receipt"] = None if bridge is None else bridge.receipt()
    base["performance_claimed"] = False
    if bridge is not None and args.debug_tensors is not None:
        args.debug_tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bridge.debug_payload(), args.debug_tensors)
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
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--debug-tensors", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run one fresh control or candidate solver process."""
    _worker(_parse_args())


if __name__ == "__main__":
    main()
