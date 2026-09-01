#!/usr/bin/env python3
"""Run one B4-A or S4 optimizer inside the same RVIR host-solver path."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring
# pylint: disable=too-many-arguments,import-outside-toplevel,duplicate-code

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import MethodType
from typing import Any, Iterator, Mapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.asplos27_s4_exact_call_bridge import (
    compile_s4_exact_call_assets_v1,
    execute_s4_exact_call_handoff_v1,
    prepare_s4_exact_call_region_from_template_v1,
)
from boundflow.runtime.asplos27_s4_exact_call_plan_template import (
    load_s4_exact_call_plan_template_v1,
)
from boundflow.runtime.asplos27_s4_optimizer_driver import execute_s4_optimizer_v1
from boundflow.runtime.bab_four_segment_exact_call_bridge import (
    execute_bab_four_segment_exact_call_handoff_v1,
)
from boundflow.runtime.bab_four_segment_optimizer import (
    PreparedBabFourSegmentOptimizerV1,
)
from boundflow.runtime.rvir_v4_live_return import live_targets_from_pre_result_v4
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from scripts import run_fsg4_b3_counter_diagnostic as diagnostic
from scripts import run_fsg4_b4a_same_solver_worker as b4a_worker
from scripts import run_fsg3_same_solver_timing as fsg3_timing
from scripts import run_rvir_v4_live_return_capture as live_runner
from scripts import run_rvir_v4_production_state_capture as capture_runner
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.asplos27-s4-same-solver-worker/v1"
CONFIGURATIONS = (
    "B4-A",
    "B4-A-PREP",
    "S4",
    "S4-PREP",
    "S4-ROOT-WARM",
    "BAB4",
)
PLAN_TEMPLATE = (
    REPOSITORY_ROOT
    / "artifacts/asplos27-s4-exact-call-plan/resnet2b-prop0-v1/plan_template.json"
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _warm_s4_runtime(
    executor: Any,
    *,
    assets: Any,
    stream: Any,
) -> tuple[dict[str, object], Any]:
    """Build and warm the region from an AOT template, without query state."""

    if not PLAN_TEMPLATE.is_file():
        raise FileNotFoundError("S4 same-solver AOT plan template is absent")
    cache = executor.prepared_core_cache
    template_hash = executor.prepared_core_template_hash
    template = cache._templates.get(template_hash)
    if template is None:
        raise ValueError("S4 same-solver prepared template is absent")
    plan_template = load_s4_exact_call_plan_template_v1(
        PLAN_TEMPLATE, core_template=template, topology=TOPOLOGY
    )
    warm_region = prepare_s4_exact_call_region_from_template_v1(
        core_template=template,
        plan_template=plan_template,
        exact_call_id="asplos27-s4-aot-static-warmup",
        topology=TOPOLOGY,
        stream=stream,
        assets=assets,
    )
    warm_run = execute_s4_optimizer_v1(warm_region.evaluator)
    warm_payload: dict[str, object] = {
        "schema_version": "boundflow.asplos27-s4-aot-static-warmup/v1",
        "plan_template_hash": plan_template.stable_hash(),
        "physical_plan_hash": warm_region.plan.stable_hash(),
        "evaluation_count": warm_run.evaluation_count,
        "mutation_count": warm_run.optimizer_mutation_count,
        "source_capture_runtime_dependency": False,
        "fallback_count": warm_run.fallback_count,
        "performance_claimed": False,
    }
    warm_payload["receipt_hash"] = _canonical_hash(warm_payload)
    warm_region.evaluator.close()
    warm_region.buffers.close()
    prepared = prepare_s4_exact_call_region_from_template_v1(
        core_template=template,
        plan_template=plan_template,
        exact_call_id="asplos27-s4-live-core-000001",
        topology=TOPOLOGY,
        stream=stream,
        assets=assets,
    )
    return warm_payload, prepared


def _warm_four_segment_runtime(
    optimizer: PreparedBabFourSegmentOptimizerV1,
    prepared_region: Any,
) -> dict[str, object]:
    """Warm module loading, DLPack views, and arenas with legal dummy state."""

    import torch

    with torch.no_grad():
        for spec, tensor in zip(
            prepared_region.plan.tensor_specs, prepared_region.executor.tensors
        ):
            if not spec.name.startswith("param/"):
                tensor.zero_()
                if spec.name == "input/upper" or spec.name.endswith("/upper"):
                    tensor.fill_(1.0)
    dummy: dict[str, torch.Tensor] = {}
    for layout in prepared_region.plan.relu_layouts:
        dummy[layout.alpha_path] = torch.full(
            (2, 1, prepared_region.plan.domain_count, len(layout.alpha_flat_indices)),
            0.5,
            device=prepared_region.stream.device,
            dtype=torch.float32,
        )
        dummy[layout.beta_path] = torch.zeros(
            (prepared_region.plan.domain_count, len(layout.beta_locations[0])),
            device=prepared_region.stream.device,
            dtype=torch.float32,
        )
    optimizer.rebind(dummy)
    started = time.perf_counter_ns()
    run = optimizer.run(prepared_region.stream)
    warm_ns = time.perf_counter_ns() - started
    owner = optimizer.owner
    owner.last_trace = None
    owner.forward_count = 0
    owner.backward_count = 0
    for executor in (
        owner.terminal_executor,
        owner.residual_executor,
        owner.projection_executor,
        owner.input_executor,
    ):
        if executor is None:
            raise ValueError("activation-BaB warmup compiled segment is absent")
        executor.forward_launch_count = 0
        executor.backward_launch_count = 0
        executor.fallback_count = 0
    return {
        "schema_version": "boundflow.bab-four-segment-static-warmup/v1",
        "warmup_ns": warm_ns,
        "evaluation_count": run.evaluation_count,
        "mutation_count": run.mutation_count,
        "source_capture_runtime_dependency": False,
        "fallback_count": run.fallback_count,
        "performance_claimed": False,
    }


@contextmanager
def _s4_candidate_executor(
    receipt_sink: list[dict[str, object]],
    *,
    four_segment: bool = False,
) -> Iterator[None]:
    """Replace only B4-A's optimizer/handoff producer with compiled S4."""

    import torch

    from boundflow.runtime import fsg4_b4a_terminal_lower_adjoint_handoff as b4a

    base_prepared = diagnostic._use_prepared_executor

    @contextmanager
    def use_s4(_configuration: str) -> Iterator[None]:
        with base_prepared("B4-A"):
            prepared_factory = live_runner._LiveExecutor

            def s4_factory(**kwargs: Any) -> Any:
                program = kwargs.get("precompiled_program")
                if program is None:
                    raise ValueError(
                        "S4 same-solver execution requires compiled program"
                    )
                prepare_started = time.perf_counter_ns()
                executor = prepared_factory(**kwargs)
                device = torch.device("cuda", torch.cuda.current_device())
                assets = compile_s4_exact_call_assets_v1(device=device)
                stream = torch.cuda.Stream(device=device)
                warmup_receipt, prepared_region = _warm_s4_runtime(
                    executor, assets=assets, stream=stream
                )
                four_optimizer = (
                    PreparedBabFourSegmentOptimizerV1(prepared_region)
                    if four_segment
                    else None
                )
                four_warmup = (
                    _warm_four_segment_runtime(four_optimizer, prepared_region)
                    if four_optimizer is not None
                    else None
                )
                static_prepare_ns = time.perf_counter_ns() - prepare_started
                base_execute = executor.execute

                def execute_with_s4(
                    _self: Any,
                    net: Any,
                    pre_result: Any,
                    call_kwargs: Mapping[str, Any],
                ) -> Any:
                    snapshots: list[Any] = []
                    original_snapshot = capture_runner._build_core_pre_snapshot

                    def capture_snapshot(*args: Any, **inner_kwargs: Any) -> Any:
                        snapshot = original_snapshot(*args, **inner_kwargs)
                        snapshots.append(snapshot)
                        return snapshot

                    def s4_handoff(*args: Any, **inner_kwargs: Any) -> Any:
                        if len(snapshots) != 1 or receipt_sink:
                            raise ValueError(
                                "S4 same-solver exact-call cardinality differs"
                            )
                        module = args[0]
                        input_spec = args[1]
                        linear_spec_c = cast(
                            torch.Tensor, inner_kwargs["linear_spec_C"]
                        )
                        snapshot = snapshots[0]
                        signature_mapping = initialize_rvir_v4_native_pre_state(
                            snapshot, TOPOLOGY
                        )
                        mapping = signature_mapping.to(
                            device=linear_spec_c.device, dtype=linear_spec_c.dtype
                        )
                        live_sources = live_targets_from_pre_result_v4(
                            pre_result, TOPOLOGY
                        )
                        execution: Any
                        if four_optimizer is None:
                            execution = execute_s4_exact_call_handoff_v1(
                                program=program,
                                module=module,
                                snapshot=snapshot,
                                mapping=mapping,
                                live_sources=live_sources,
                                exact_call_id="asplos27-s4-live-core-000001",
                                input_spec=input_spec,
                                linear_spec_C=linear_spec_c,
                                relu_pre=inner_kwargs["relu_pre"],
                                initial_state=inner_kwargs["initial_state"],
                                mutation_policy=inner_kwargs["mutation_policy"],
                                schedule=inner_kwargs["schedule"],
                                topology=inner_kwargs["topology"],
                                stream=stream,
                                assets=assets,
                                prevalidated_plan=inner_kwargs.get("prevalidated_plan"),
                                prepared_region=prepared_region,
                                signature_mapping=signature_mapping,
                            )
                        else:
                            prevalidated_plan = inner_kwargs.get("prevalidated_plan")
                            if prevalidated_plan is None:
                                raise ValueError(
                                    "activation-BaB exact-call plan is absent"
                                )
                            execution = execute_bab_four_segment_exact_call_handoff_v1(
                                module=module,
                                live_sources=live_sources,
                                exact_call_id="asplos27-s4-live-core-000001",
                                input_spec=input_spec,
                                linear_spec_C=linear_spec_c,
                                relu_pre=inner_kwargs["relu_pre"],
                                initial_state=inner_kwargs["initial_state"],
                                mutation_policy=inner_kwargs["mutation_policy"],
                                schedule=inner_kwargs["schedule"],
                                topology=inner_kwargs["topology"],
                                stream=stream,
                                prevalidated_plan=prevalidated_plan,
                                prepared_region=prepared_region,
                                optimizer=four_optimizer,
                            )
                        payload = execution.receipt.to_dict()
                        payload["static_prepare_ns"] = static_prepare_ns
                        payload["static_prepare_excluded_from_query"] = True
                        payload["source_capture_runtime_dependency"] = False
                        payload["plan_template_relative_path"] = str(
                            PLAN_TEMPLATE.relative_to(REPOSITORY_ROOT)
                        )
                        payload["static_warmup_receipt_hash"] = warmup_receipt[
                            "receipt_hash"
                        ]
                        if four_warmup is not None:
                            payload["four_segment_static_warmup"] = four_warmup
                        receipt_sink.append(payload)
                        return execution.handoff_result

                    with (
                        diagnostic._patch_attribute(
                            capture_runner,
                            "_build_core_pre_snapshot",
                            capture_snapshot,
                        ),
                        diagnostic._patch_attribute(
                            b4a,
                            "execute_terminal_optimizer_with_lower_adjoint_handoff_v1",
                            s4_handoff,
                        ),
                    ):
                        result = base_execute(net, pre_result, call_kwargs)
                    if len(snapshots) != 1 or len(receipt_sink) != 1:
                        raise ValueError(
                            "S4 same-solver execution did not activate exactly once"
                        )
                    return result

                setattr(executor, "execute", MethodType(execute_with_s4, executor))
                return executor

            with diagnostic._patch_attribute(live_runner, "_LiveExecutor", s4_factory):
                yield

    with diagnostic._patch_attribute(diagnostic, "_use_prepared_executor", use_s4):
        yield


def _base_namespace(args: argparse.Namespace, result: Path) -> argparse.Namespace:
    return argparse.Namespace(
        configuration="B4-A",
        mode=args.mode,
        run_id=args.run_id,
        block_index=args.block_index,
        sequence_position=args.sequence_position,
        benchmark_root=args.benchmark_root,
        abcrown_root=args.abcrown_root,
        model=args.model,
        property=args.property,
        result=result,
        prepare_static_request=args.configuration
        in {"B4-A-PREP", "S4-PREP", "S4-ROOT-WARM", "BAB4"},
        prepare_root_optimizer_warmup=args.configuration == "S4-ROOT-WARM",
        attribute_root_incomplete=bool(
            getattr(args, "attribute_root_incomplete", False)
        ),
    )


def _worker(args: argparse.Namespace) -> None:
    if args.configuration not in CONFIGURATIONS or args.mode != "control":
        raise ValueError("S4 initial same-solver worker admits control mode only")
    receipts: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-s4-same-solver-") as raw:
        base_result = Path(raw) / "b4a-worker.json"
        if args.configuration in {"S4", "S4-PREP", "S4-ROOT-WARM", "BAB4"}:
            with (
                diagnostic._patch_attribute(
                    fsg3_timing, "POST_PREPARE_ENVIRONMENT_WINDOW", True
                ),
                _s4_candidate_executor(
                    receipts, four_segment=args.configuration == "BAB4"
                ),
            ):
                b4a_worker._worker(_base_namespace(args, base_result))
        else:
            with diagnostic._patch_attribute(
                fsg3_timing, "POST_PREPARE_ENVIRONMENT_WINDOW", True
            ):
                b4a_worker._worker(_base_namespace(args, base_result))
        base = json.loads(base_result.read_text(encoding="utf-8"))
    if (
        not isinstance(base, Mapping)
        or base.get("schema_version") != b4a_worker.WORKER_SCHEMA
        or base.get("configuration") != "B4-A"
        or base.get("performance_claimed") is not False
        or len(receipts)
        != int(args.configuration in {"S4", "S4-PREP", "S4-ROOT-WARM", "BAB4"})
    ):
        raise ValueError("S4 inherited same-solver envelope differs")
    run = base.get("run")
    if not isinstance(run, Mapping):
        raise TypeError("S4 inherited same-solver run differs")
    envelope = {
        "schema_version": WORKER_SCHEMA,
        "configuration": args.configuration,
        "mode": args.mode,
        "run": dict(run),
        "activation": base.get("activation"),
        "diagnostics": base.get("diagnostics"),
        "s4_exact_call_receipts": receipts,
        "performance_claimed": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics = run.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("S4 inherited same-solver metrics differ")
    print(
        json.dumps(
            {
                "configuration": args.configuration,
                "core_wall_ns": metrics["core_wall_ns"],
                "query_wall_ns": metrics["query_wall_ns"],
                "s4_exact_call_count": len(receipts),
                "performance_claimed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--mode", choices=("control",), default="control")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--sequence-position", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for name in ("benchmark_root", "abcrown_root", "model", "property", "result"):
        setattr(args, name, getattr(args, name).resolve())
    _worker(args)


if __name__ == "__main__":
    main()
