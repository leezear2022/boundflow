#!/usr/bin/env python3
"""Run symmetric GC control or cumulative root-CROWN plus BAB4 execution."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-locals,too-many-statements,protected-access

from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import wraps
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.root_crown_terminal_linear import (  # noqa: E402
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_full_pipeline_tir import (  # noqa: E402
    RootCrownFullPipelineTIRExecutorV1,
)
from boundflow.runtime.root_crown_input_domain_live import (  # noqa: E402
    RootCrownInputDomainLiveBridgeV1,
)
from boundflow.runtime.root_crown_projection_live import (  # noqa: E402
    RootCrownProjectionLiveBridgeV1,
)
from boundflow.runtime.root_crown_suffix_live import (  # noqa: E402
    RootCrownSuffixLiveBridgeV1,
)
from scripts import run_asplos27_s4_same_solver_worker as s4_worker  # noqa: E402
from scripts import run_fsg4_b3_counter_diagnostic as diagnostic  # noqa: E402
from scripts.probe_root_crown_input_domain_tir import (  # noqa: E402
    _template as input_template,
)
from scripts.run_root_crown_expanded_live_worker import (  # noqa: E402
    DEFAULT_INPUT_CAPTURE,
    DEFAULT_PROJECTION_CAPTURE,
    DEFAULT_RESIDUAL_CAPTURE,
    _projection_template,
)
from scripts.run_root_crown_residual_live_worker import (  # noqa: E402
    _template as residual_template,
)
from scripts.run_root_crown_terminal_live_worker import FEATURE_INDICES  # noqa: E402

SCHEMA = "boundflow.bab4-root-gc-worker/v1"
CONTROL = "B4-A-GC"
CANDIDATE = "BAB4-GC-ROOT"


class _RootSegmentAttributionV1:
    """Measure host submission and same-stream CUDA spans outside formal timing."""

    def __init__(self) -> None:
        self._records: dict[str, list[tuple[int, Any, Any]]] = {}

    def _instrument_method(self, owner: Any, name: str, label: str) -> None:
        import torch

        original = cast(Callable[..., Any], getattr(owner, name))

        @wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            device = torch.cuda.current_device()
            stream = torch.cuda.current_stream(device)
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            started_ns = time.perf_counter_ns()
            start.record(stream)
            try:
                return original(*args, **kwargs)
            finally:
                finish.record(stream)
                host_ns = time.perf_counter_ns() - started_ns
                self._records.setdefault(label, []).append((host_ns, start, finish))

        setattr(owner, name, measured)

    def install(
        self,
        executor: RootCrownFullPipelineTIRExecutorV1,
        suffix_bridge: RootCrownSuffixLiveBridgeV1,
        projection_bridge: RootCrownProjectionLiveBridgeV1,
        input_bridge: RootCrownInputDomainLiveBridgeV1,
    ) -> None:
        """Instrument cumulative boundaries, TIR modules, and admission setup."""

        for name in ("stage_terminal", "stage_residual", "stage_projection", "consume"):
            self._instrument_method(executor, name, f"pipeline.{name}")
        self._instrument_method(executor, "backward", "pipeline.backward")
        for label, executor_owner in (
            ("terminal", executor.expanded.suffix.terminal),
            ("residual", executor.expanded.suffix.residual),
            ("projection", executor.expanded.projection),
            ("input_domain", executor.input_domain),
        ):
            self._instrument_method(executor_owner, "forward", f"{label}.forward")
            self._instrument_method(executor_owner, "backward", f"{label}.backward")
        for label, bridge_owner in (
            ("suffix", suffix_bridge),
            ("projection", projection_bridge),
            ("input_domain", input_bridge),
        ):
            self._instrument_method(
                bridge_owner, "_admit_static", f"{label}.admit_static"
            )
            self._instrument_method(
                bridge_owner, "_set_relu_state", f"{label}.set_relu_state"
            )

    def receipt(self) -> dict[str, object]:
        """Synchronize after the query and serialize all diagnostic intervals."""

        import torch

        torch.cuda.synchronize()
        segments: dict[str, object] = {}
        event_pair_count = 0
        for label, records in sorted(self._records.items()):
            host_values = [record[0] for record in records]
            cuda_values = [
                round(float(record[1].elapsed_time(record[2])) * 1_000_000)
                for record in records
            ]
            event_pair_count += len(records)
            segments[label] = {
                "count": len(records),
                "host_total_ns": sum(host_values),
                "host_median_ns": round(statistics.median(host_values)),
                "cuda_total_ns": sum(cuda_values),
                "cuda_median_ns": round(statistics.median(cuda_values)),
            }
        return {
            "schema_version": "boundflow.root-segment-attribution/v1",
            "diagnostic_only": True,
            "included_in_performance_claim": False,
            "event_pair_count": event_pair_count,
            "segments": segments,
        }


def _prepare_root_pipeline(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, int]:
    import torch

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
    projection = _projection_template(args.projection_capture)
    input_payload = torch.load(
        args.input_capture, map_location="cpu", weights_only=True
    )
    evaluations = cast(list[dict[str, Any]], input_payload.get("evaluations"))
    if (
        input_payload.get("schema_version") != "boundflow.root-crown-input-tensors/v1"
        or len(evaluations) != 5
    ):
        raise ValueError("BAB4 cumulative root input capture differs")
    started_ns = time.perf_counter_ns()
    executor = RootCrownFullPipelineTIRExecutorV1(
        terminal,
        residual,
        projection,
        input_template(evaluations[0]),
    )
    suffix_bridge = RootCrownSuffixLiveBridgeV1(terminal, residual, executor)
    projection_bridge = RootCrownProjectionLiveBridgeV1(projection, executor)
    input_bridge = RootCrownInputDomainLiveBridgeV1(executor.input_template, executor)
    executor.prepare()
    prepare_ns = time.perf_counter_ns() - started_ns
    return executor, suffix_bridge, projection_bridge, input_bridge, prepare_ns


def _worker(args: argparse.Namespace) -> None:
    if args.configuration not in (CONTROL, CANDIDATE):
        raise ValueError("BAB4 cumulative root configuration differs")
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    from auto_LiRPA import BoundedModule
    from boundflow.runtime import prepared_root_optimizer_warmup as warmup_module

    candidate = args.configuration == CANDIDATE
    warm_suffix = warm_projection = warm_input_domain = None
    suffix = projection = input_domain = None
    warm_executor: RootCrownFullPipelineTIRExecutorV1 | None = None
    root_warm_pipeline_prepare_ns = 0
    root_query_pipeline_prepare_ns = 0
    root_exact_warmup_reset_ns = 0
    root_segment_observer: _RootSegmentAttributionV1 | None = None
    if candidate:
        (
            warm_executor,
            warm_suffix,
            warm_projection,
            warm_input_domain,
            root_warm_pipeline_prepare_ns,
        ) = _prepare_root_pipeline(args)

    with tempfile.TemporaryDirectory(prefix="boundflow-bab4-root-gc-") as raw:
        base_result = Path(raw) / "worker.json"
        base_args = argparse.Namespace(
            configuration="BAB4-GC" if candidate else "B4-A-GC",
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
            attribute_complete_prelude=True,
        )
        root_query_install_count = 0
        root_warmup_receipts: dict[str, object] | None = None
        with ExitStack() as stack:
            if candidate:
                assert warm_suffix is not None
                assert warm_projection is not None
                assert warm_input_domain is not None
                original_warmup = warmup_module.prepare_root_optimizer_warmup_v1

                def warm_then_install(*warm_args: Any, **warm_kwargs: Any) -> Any:
                    nonlocal root_query_install_count, root_warmup_receipts
                    nonlocal root_query_pipeline_prepare_ns
                    nonlocal root_exact_warmup_reset_ns
                    nonlocal suffix, projection, input_domain
                    nonlocal root_segment_observer
                    if root_query_install_count != 0:
                        raise RuntimeError("BAB4 cumulative root install count differs")
                    with ExitStack() as warm_stack:
                        warm_stack.enter_context(warm_suffix.install(BoundedModule))
                        warm_stack.enter_context(warm_projection.install(BoundedModule))
                        warm_stack.enter_context(
                            warm_input_domain.install(BoundedModule)
                        )
                        receipt = original_warmup(*warm_args, **warm_kwargs)
                        root_warmup_receipts = {
                            "suffix": warm_suffix.receipt(),
                            "projection": warm_projection.receipt(),
                            "input_domain": warm_input_domain.receipt(),
                        }
                    if warm_executor is None:
                        raise RuntimeError("BAB4 cumulative root warm executor absent")
                    reset_started_ns = time.perf_counter_ns()
                    warm_suffix.reset_after_exact_warmup_v1()
                    warm_projection.reset_after_exact_warmup_v1()
                    warm_input_domain.reset_after_exact_warmup_v1()
                    warm_executor.reset_after_exact_warmup_v1()
                    root_exact_warmup_reset_ns = (
                        time.perf_counter_ns() - reset_started_ns
                    )
                    suffix = warm_suffix
                    projection = warm_projection
                    input_domain = warm_input_domain
                    if args.attribute_root_segments:
                        root_segment_observer = _RootSegmentAttributionV1()
                        root_segment_observer.install(
                            warm_executor, suffix, projection, input_domain
                        )
                    stack.enter_context(suffix.install(BoundedModule))
                    stack.enter_context(projection.install(BoundedModule))
                    stack.enter_context(input_domain.install(BoundedModule))
                    root_query_install_count += 1
                    return receipt

                stack.enter_context(
                    diagnostic._patch_attribute(
                        warmup_module,
                        "prepare_root_optimizer_warmup_v1",
                        warm_then_install,
                    )
                )
            s4_worker._worker(base_args)
            if candidate:
                assert suffix is not None
                assert projection is not None
                assert input_domain is not None
                root_receipts = {
                    "suffix": suffix.receipt(),
                    "projection": projection.receipt(),
                    "input_domain": input_domain.receipt(),
                }
            else:
                root_receipts = None
        base = json.loads(base_result.read_text(encoding="utf-8"))

    root_segment_attribution = (
        root_segment_observer.receipt() if root_segment_observer is not None else None
    )

    if root_query_install_count != int(candidate):
        raise ValueError("BAB4 cumulative root query installation differs")
    if base.get("performance_claimed") is not False:
        raise ValueError("BAB4 cumulative root base claim differs")
    payload = dict(base)
    payload.update(
        {
            "schema_version": SCHEMA,
            "configuration": args.configuration,
            "base_configuration": "BAB4-GC" if candidate else "B4-A-GC",
            "root_warm_pipeline_prepare_ns": root_warm_pipeline_prepare_ns,
            "root_query_pipeline_prepare_ns": root_query_pipeline_prepare_ns,
            "root_exact_warmup_reset_ns": root_exact_warmup_reset_ns,
            "root_total_prepare_ns": (
                root_warm_pipeline_prepare_ns + root_query_pipeline_prepare_ns
            ),
            "root_pipeline_prepare_excluded_from_query": True,
            "root_prepared_runtime_reused_after_exact_warmup": bool(candidate),
            "root_query_install_count": root_query_install_count,
            "root_warmup_receipts": root_warmup_receipts,
            "root_receipts": root_receipts,
            "root_segment_attribution": root_segment_attribution,
            "root_segment_attribution_enabled": bool(
                candidate and args.attribute_root_segments
            ),
            "performance_claimed": False,
        }
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=(CONTROL, CANDIDATE), required=True)
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
    parser.add_argument("--attribute-root-segments", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one cumulative same-solver worker."""

    args = _parse_args()
    for name in (
        "benchmark_root",
        "abcrown_root",
        "model",
        "property",
        "residual_capture",
        "projection_capture",
        "input_capture",
        "result",
    ):
        setattr(args, name, getattr(args, name).resolve())
    _worker(args)


if __name__ == "__main__":
    main()
