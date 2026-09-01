#!/usr/bin/env python3
"""Run fresh control or terminal+residual+projection root CROWN execution."""

# pylint: disable=import-error,wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-locals,too-many-statements,protected-access,duplicate-code
# mypy: disable-error-code=import-untyped

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.root_crown_projection import (  # noqa: E402
    RootCrownProjectionTemplateV1,
)
from boundflow.runtime.root_crown_full_pipeline_tir import (  # noqa: E402
    RootCrownFullPipelineTIRExecutorV1,
)
from boundflow.runtime.root_crown_input_domain_live import (  # noqa: E402
    RootCrownInputDomainLiveBridgeV1,
)
from boundflow.backends.tvm.root_crown_terminal_linear import (  # noqa: E402
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_projection_live import (  # noqa: E402
    RootCrownProjectionLiveBridgeV1,
)
from boundflow.runtime.root_crown_expanded_suffix_tir import (  # noqa: E402
    RootCrownExpandedSuffixTIRExecutorV1,
)
from boundflow.runtime.root_crown_suffix_live import (  # noqa: E402
    RootCrownSuffixLiveBridgeV1,
)
from scripts import run_asplos27_s4_same_solver_worker as s4_worker  # noqa: E402
from scripts.probe_root_crown_residual_tir import _coordinates  # noqa: E402
from scripts.probe_root_crown_input_domain_tir import (  # noqa: E402
    _template as input_template,
)
from scripts.run_root_crown_residual_live_worker import (  # noqa: E402
    DEFAULT_CAPTURE as DEFAULT_RESIDUAL_CAPTURE,
    _template as residual_template,
)
from scripts.run_root_crown_terminal_live_worker import FEATURE_INDICES  # noqa: E402

DEFAULT_PROJECTION_CAPTURE = REPOSITORY_ROOT / (
    "artifacts/root-crown-projection-capture/resnet2b-prop0-v1/capture.pt"
)
DEFAULT_INPUT_CAPTURE = REPOSITORY_ROOT / (
    "artifacts/root-crown-input-capture/resnet2b-prop0-v1/capture.pt"
)


def _projection_template(path: Path) -> RootCrownProjectionTemplateV1:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    evaluations = payload.get("evaluations")
    if (
        payload.get("schema_version") != "boundflow.root-crown-projection-tensors/v1"
        or not isinstance(evaluations, list)
        or len(evaluations) != 5
    ):
        raise ValueError("root CROWN projection live capture differs")
    first = cast(dict[str, Any], evaluations[0])
    incoming = cast(torch.Tensor, first["incoming_lower_a"])
    output = cast(torch.Tensor, first["output_lower_a"])
    major, minor = torch.cuda.get_device_capability()
    return RootCrownProjectionTemplateV1(
        spec_count=int(incoming.shape[0]),
        domain_count=int(incoming.shape[1]),
        output_channels=int(incoming.shape[2]),
        output_height=int(incoming.shape[3]),
        output_width=int(incoming.shape[4]),
        input_channels=int(output.shape[2]),
        input_height=int(output.shape[3]),
        input_width=int(output.shape[4]),
        entry_alpha_coordinates=_coordinates(first["entry_alpha_feature_indices"]),
        inner_alpha_coordinates=_coordinates(first["inner_alpha_feature_indices"]),
        compute_capability=f"sm_{major}{minor}",
        thread_extent=128,
    )


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    from auto_LiRPA import BoundedModule

    suffix = None
    projection = None
    expanded = None
    input_domain = None
    prepare_ns = 0
    if args.mode != "control":
        residual = residual_template(args.residual_capture)
        terminal = RootCrownTerminalLinearTemplateV1(
            spec_count=residual.spec_count,
            domain_count=residual.domain_count,
            current_features=100,
            previous_features=(residual.channels * residual.height * residual.width),
            alpha_feature_indices=FEATURE_INDICES,
            compute_capability=residual.compute_capability,
            thread_extent=128,
        )
        projection_template = _projection_template(args.projection_capture)
        started = time.perf_counter_ns()
        if args.mode == "candidate-full":
            import torch

            input_payload = torch.load(
                args.input_capture, map_location="cpu", weights_only=True
            )
            input_evaluations = cast(
                list[dict[str, Any]], input_payload.get("evaluations")
            )
            if (
                input_payload.get("schema_version")
                != "boundflow.root-crown-input-tensors/v1"
                or len(input_evaluations) != 5
            ):
                raise ValueError("root CROWN input-domain live capture differs")
            full = RootCrownFullPipelineTIRExecutorV1(
                terminal,
                residual,
                projection_template,
                input_template(input_evaluations[0]),
            )
            expanded = full
            suffix = RootCrownSuffixLiveBridgeV1(terminal, residual, full)
            projection = RootCrownProjectionLiveBridgeV1(projection_template, full)
            input_domain = RootCrownInputDomainLiveBridgeV1(full.input_template, full)
            full.prepare()
        elif args.mode == "candidate-single":
            expanded = RootCrownExpandedSuffixTIRExecutorV1(
                terminal, residual, projection_template
            )
            suffix = RootCrownSuffixLiveBridgeV1(terminal, residual, expanded)
            projection = RootCrownProjectionLiveBridgeV1(projection_template, expanded)
            expanded.prepare()
        else:
            suffix = RootCrownSuffixLiveBridgeV1(terminal, residual)
            projection = RootCrownProjectionLiveBridgeV1(projection_template)
            suffix.executor.prepare()
            projection.executor.prepare()
        prepare_ns = time.perf_counter_ns() - started
    with tempfile.TemporaryDirectory(prefix="boundflow-root-expanded-live-") as raw:
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
        if suffix is None or projection is None:
            s4_worker._worker(namespace)
        else:
            with ExitStack() as stack:
                stack.enter_context(suffix.install(BoundedModule))
                stack.enter_context(projection.install(BoundedModule))
                if input_domain is not None:
                    stack.enter_context(input_domain.install(BoundedModule))
                s4_worker._worker(namespace)
        base = json.loads(base_result.read_text(encoding="utf-8"))
    base["root_expanded_mode"] = args.mode
    base["root_expanded_prepare_ns"] = prepare_ns
    base["root_expanded_prepare_excluded_from_query"] = True
    base["root_suffix_receipt"] = None if suffix is None else suffix.receipt()
    base["root_projection_receipt"] = (
        None if projection is None else projection.receipt()
    )
    base["root_input_domain_receipt"] = (
        None if input_domain is None else input_domain.receipt()
    )
    base["root_expanded_receipt"] = (
        None
        if expanded is None
        else {
            "schema_version": "boundflow.root-crown-expanded-live/v1",
            "prepare_count": expanded.prepare_count,
            "residual_stage_count": expanded.residual_stage_count,
            "consume_count": expanded.consume_count,
            "fallback_count": expanded.fallback_count,
            "cumulative_autograd_owner_count": 1,
            "custom_autograd_invocation_count": expanded.consume_count,
            "performance_claimed": False,
        }
    )
    owner_count = 0 if suffix is None else (1 if expanded is not None else 2)
    base["cumulative_autograd_owner_count"] = owner_count
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
                "autograd_owner_count": owner_count,
                "performance_claimed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("control", "candidate", "candidate-single", "candidate-full"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--sequence-position", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument(
        "--residual-capture", type=Path, default=DEFAULT_RESIDUAL_CAPTURE
    )
    parser.add_argument(
        "--projection-capture", type=Path, default=DEFAULT_PROJECTION_CAPTURE
    )
    parser.add_argument("--input-capture", type=Path, default=DEFAULT_INPUT_CAPTURE)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one fresh solver process."""

    _worker(_parse_args())


if __name__ == "__main__":
    main()
