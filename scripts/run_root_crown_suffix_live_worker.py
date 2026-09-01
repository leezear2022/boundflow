#!/usr/bin/env python3
"""Run one fresh control or cumulative root-suffix candidate process."""

# pylint: disable=import-error,wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-locals,protected-access,duplicate-code
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
from boundflow.runtime.root_crown_suffix_live import (  # noqa: E402
    RootCrownSuffixLiveBridgeV1,
)
from scripts import run_asplos27_s4_same_solver_worker as s4_worker  # noqa: E402
from scripts.run_root_crown_residual_live_worker import (  # noqa: E402
    DEFAULT_CAPTURE,
    _template as residual_template,
)
from scripts.run_root_crown_terminal_live_worker import (  # noqa: E402
    FEATURE_INDICES,
)


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    from auto_LiRPA import BoundedModule

    bridge = None
    prepare_ns = 0
    if args.mode == "candidate":
        residual = residual_template(args.capture)
        terminal = RootCrownTerminalLinearTemplateV1(
            spec_count=residual.spec_count,
            domain_count=residual.domain_count,
            current_features=100,
            previous_features=(residual.channels * residual.height * residual.width),
            alpha_feature_indices=FEATURE_INDICES,
            compute_capability=residual.compute_capability,
            thread_extent=128,
        )
        started = time.perf_counter_ns()
        bridge = RootCrownSuffixLiveBridgeV1(terminal, residual)
        bridge.executor.prepare()
        prepare_ns = time.perf_counter_ns() - started
    with tempfile.TemporaryDirectory(prefix="boundflow-root-suffix-live-") as raw:
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
    base["root_suffix_mode"] = args.mode
    base["root_suffix_prepare_ns"] = prepare_ns
    base["root_suffix_prepare_excluded_from_query"] = True
    base["root_suffix_receipt"] = None if bridge is None else bridge.receipt()
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
    """Run one fresh solver process."""

    _worker(_parse_args())


if __name__ == "__main__":
    main()
