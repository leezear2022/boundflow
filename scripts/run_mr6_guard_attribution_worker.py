#!/usr/bin/env python3
"""Run one provider/full/structural-only MR6 attribution worker."""

# pylint: disable=protected-access,import-error,wrong-import-position
# pylint: disable=missing-function-docstring,too-many-locals,duplicate-code
# pylint: disable=too-few-public-methods,super-init-not-called
# mypy: disable-error-code=import-untyped

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import (  # noqa: E402
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    VNNCOMP_COMMIT,
    canonical_hash,
)
from boundflow.runtime import mr5_generalized_crown_conv as runtime_module  # noqa: E402
from boundflow.runtime import (
    mr5_multi_conv_production_bridge as bridge_module,
)  # noqa: E402
from scripts import run_mr5_multi_conv_timing_worker as mr5_worker  # noqa: E402
from scripts.run_mr3_provider_hook_feasibility import (  # noqa: E402
    MODEL_SHA256,
    PROPERTY_SHA256,
    _git,
    _sha256,
)

WORKER_SCHEMA = "boundflow.mr6-hot-path-guard-attribution-worker/v1"
MODES = ("provider", "full", "diagnostic")


def _validate_structural(signature: Any, tensors: Any) -> None:
    signature.validate()
    expected = {
        "incoming": signature.incoming_shape,
        "lower": signature.relaxation_shape,
        "upper": signature.relaxation_shape,
        "alpha": signature.relaxation_shape,
        "incoming_bias": (signature.domain_count, signature.spec_count),
        "weight": signature.weight_shape,
        "operator_bias": (signature.output_channels,),
    }
    for name, shape in expected.items():
        tensor = getattr(tensors, name)
        if (
            tuple(tensor.shape) != shape
            or str(tensor.dtype) != "torch.float32"
            or tensor.device.type != "cuda"
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"MR6 structural tensor differs: {name}")
    if tensors.incoming.requires_grad != tensors.alpha.requires_grad:
        raise ValueError("MR6 structural gradient ownership differs")
    if tensors.lower.requires_grad or tensors.upper.requires_grad:
        raise ValueError("MR6 structural bound ownership differs")


class _GuardLedger:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.validation_calls = 0
        self.content_calls = 0

    def to_dict(self, *, site_evaluations: int) -> dict[str, object]:
        input_guards = self.validation_calls * 9
        content_guards = self.content_calls
        output_guards = site_evaluations * 2
        if self.mode == "diagnostic":
            executed_input = 0
            executed_content = 0
            elided_input = input_guards
            elided_content = content_guards
        else:
            executed_input = input_guards
            executed_content = content_guards
            elided_input = 0
            elided_content = 0
        value: dict[str, object] = {
            "policy": self.mode,
            "site_evaluations": site_evaluations,
            "validation_calls": self.validation_calls,
            "content_calls": self.content_calls,
            "input_value_guards_executed": executed_input,
            "handoff_content_guards_executed": executed_content,
            "output_finite_guards_executed": output_guards,
            "input_value_guards_elided": elided_input,
            "handoff_content_guards_elided": elided_content,
            "synchronizing_guards_executed": (
                executed_input + executed_content + output_guards
            ),
            "production_admitted": self.mode != "diagnostic",
            "performance_claimed": False,
        }
        value["receipt_hash"] = canonical_hash(value)
        return value


@contextmanager
def _guard_policy(ledger: _GuardLedger) -> Iterator[None]:
    original_validate = runtime_module.validate_mr5_generalized_conv_tensors
    original_content = bridge_module._content_equal

    def counted_validate(signature: Any, tensors: Any) -> None:
        ledger.validation_calls += 1
        if ledger.mode == "diagnostic":
            _validate_structural(signature, tensors)
        else:
            original_validate(signature, tensors)

    def counted_content(left: Any, right: Any) -> bool:
        ledger.content_calls += 1
        if ledger.mode == "diagnostic":
            return tuple(left.shape) == tuple(right.shape)
        return original_content(left, right)

    runtime_module.validate_mr5_generalized_conv_tensors = counted_validate
    bridge_module._content_equal = counted_content
    try:
        yield
    finally:
        runtime_module.validate_mr5_generalized_conv_tensors = original_validate
        bridge_module._content_equal = original_content


def _configured_tracker(mode: str, sink: list[Any]):
    requested_mode = mode
    base_mode = "provider" if mode == "provider" else "bridge"

    class _MR6Tracker(mr5_worker._TimingTracker):
        def __init__(self, torch_module: Any, *, mode: str) -> None:
            super().__init__(torch_module, mode=base_mode)
            self.guard_ledger = _GuardLedger(
                requested_mode if base_mode == "bridge" else "provider"
            )
            sink.append(self)

        @contextmanager
        def install(self, bounded_module: Any) -> Iterator[None]:
            if base_mode == "provider":
                with super().install(bounded_module):
                    yield
                return
            with _guard_policy(self.guard_ledger):
                with super().install(bounded_module):
                    yield

    return _MR6Tracker


def _run(args: argparse.Namespace) -> dict[str, object]:
    base_mode = "provider" if args.mode == "provider" else "bridge"
    trackers: list[Any] = []
    setattr(
        mr5_worker.legacy, "_TimingTracker", _configured_tracker(args.mode, trackers)
    )
    mr5_worker.legacy.WORKER_SCHEMA = mr5_worker.WORKER_SCHEMA
    base_path = args.result_json.with_suffix(".base.json")
    base_args = argparse.Namespace(
        benchmark_root=args.benchmark_root,
        abcrown_root=args.abcrown_root,
        model=args.model,
        property=args.property,
        mode=base_mode,
        result_json=base_path,
    )
    mr5_worker.legacy._worker(base_args)
    if len(trackers) != 1:
        raise ValueError("MR6 tracker lifecycle differs")
    tracker = trackers[0]
    base_worker = json.loads(base_path.read_text(encoding="utf-8"))
    base_path.unlink()
    site_evaluations = 0 if base_mode == "provider" else tracker.inner_count * 3
    guard_receipt = tracker.guard_ledger.to_dict(site_evaluations=site_evaluations)
    if args.mode == "full" and guard_receipt["synchronizing_guards_executed"] != 360:
        raise ValueError("MR6 full guard count differs")
    if (
        args.mode == "diagnostic"
        and guard_receipt["synchronizing_guards_executed"] != 60
    ):
        raise ValueError("MR6 diagnostic guard count differs")
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "mode": args.mode,
        "base_worker": base_worker,
        "guard_receipt": guard_receipt,
        "timing_recorded": True,
        "production_admitted": args.mode != "diagnostic",
        "performance_claimed": False,
    }
    payload["worker_hash"] = canonical_hash(payload)
    args.result_json.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    if (
        _git(args.abcrown_root, "rev-parse", "HEAD") != ABCROWN_COMMIT
        or _git(args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD")
        != AUTO_LIRPA_COMMIT
        or _git(args.benchmark_root, "rev-parse", "HEAD") != VNNCOMP_COMMIT
        or _sha256(args.model) != MODEL_SHA256
        or _sha256(args.property) != PROPERTY_SHA256
    ):
        raise ValueError("MR6 attribution frozen input differs")
    result = _run(args)
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "guard_receipt": result["guard_receipt"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
