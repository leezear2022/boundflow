#!/usr/bin/env python3
"""Run one fresh original or candidate RVIR-v4 correctness worker."""

# pylint: disable=wrong-import-position,protected-access,import-outside-toplevel
# pylint: disable=too-many-locals,missing-function-docstring,import-error
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_rvir_v4_live_return_capture as candidate_runner
from scripts import run_rvir_v4_production_state_capture as original_runner

WORKER_SCHEMA = "boundflow.rvir-v4-five-fresh-worker/v1"


def _one_final_bound(value: object, *, label: str) -> tuple[str, Any]:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise ValueError(f"RVIR-v4 five-fresh {label} inventory differs")
    name, tensor = next(iter(value.items()))
    if not isinstance(name, str):
        raise TypeError(f"RVIR-v4 five-fresh {label} name differs")
    return name, tensor


def _run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch
    from branching_domains import (  # type: ignore[import-not-found]
        BatchedDomainList,
    )

    rows: list[dict[str, object]] = []
    original_add = BatchedDomainList.add

    def wrapped_add(
        instance: Any,
        ret: MutableMapping[str, object],
        host: Mapping[str, object],
        *positional: object,
        **keyword: object,
    ) -> object:
        before = len(instance)
        lower_name, lower = _one_final_bound(ret.get("lower_bounds"), label="lower")
        upper_name, upper = _one_final_bound(ret.get("upper_bounds"), label="upper")
        history = host.get("history")
        depths = host.get("depths")
        thresholds = host.get("thresholds")
        is_target_post_add = (
            lower_name == upper_name
            and torch.is_tensor(lower)
            and torch.is_tensor(upper)
            and torch.is_tensor(depths)
            and torch.is_tensor(thresholds)
            and isinstance(history, list)
            and tuple(depths.shape) == (6,)
            and tuple(thresholds.shape) == (6, 1)
            and int(lower.shape[0]) == 6
        )
        if not is_target_post_add:
            return original_add(instance, ret, host, *positional, **keyword)
        if (
            lower_name != upper_name
            or not torch.is_tensor(lower)
            or not torch.is_tensor(upper)
            or not torch.is_tensor(depths)
            or not torch.is_tensor(thresholds)
            or not isinstance(history, list)
        ):
            raise ValueError("RVIR-v4 five-fresh queue input differs")
        input_count = int(lower.shape[0])
        result = original_add(instance, ret, host, *positional, **keyword)
        after = len(instance)
        accepted = after - before
        row: dict[str, object] = {
            "schema_version": WORKER_SCHEMA,
            "before_domain_count": before,
            "input_domain_count": input_count,
            "accepted_domain_count": accepted,
            "pruned_domain_count": input_count - accepted,
            "after_domain_count": after,
            "final_name": lower_name,
            "lower_sha256": production_tensor_sha256(lower),
            "upper_sha256": production_tensor_sha256(upper),
            "thresholds_sha256": production_tensor_sha256(thresholds),
            "history_count": len(history),
            "depths": [int(value) for value in depths.tolist()],
            "performance_claimed": False,
        }
        rows.append(row)
        return result

    BatchedDomainList.add = wrapped_add
    worker_args = argparse.Namespace(
        benchmark_root=args.benchmark_root,
        abcrown_root=args.abcrown_root,
        model=args.model,
        property=args.property,
        result=args.result,
        optimizer_step_trace=False,
        whole_core_truth=True,
    )
    try:
        if args.mode == "original":
            original_runner._worker(worker_args)
        else:
            candidate_runner._worker(worker_args)
    finally:
        BatchedDomainList.add = original_add
    payload = torch.load(args.result, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or len(rows) != 1:
        raise ValueError("RVIR-v4 five-fresh worker result differs")
    payload["five_fresh_mode"] = args.mode
    payload["queue_events"] = rows
    payload["five_fresh_worker_schema"] = WORKER_SCHEMA
    payload["performance_claimed"] = False
    torch.save(payload, args.result)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "queue_event": rows[0],
                "result": payload["solver_result"],
                "performance_claimed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("original", "candidate"), required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    args.abcrown_root = args.abcrown_root.resolve()
    args.model = args.model.resolve()
    args.property = args.property.resolve()
    args.result = args.result.resolve()
    _run(args)


if __name__ == "__main__":
    main()
