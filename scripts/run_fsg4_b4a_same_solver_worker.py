#!/usr/bin/env python3
"""Run one B3 or B4-A same-solver correctness/timing worker."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring
# pylint: disable=too-many-boolean-expressions
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Mapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import canonical_hash, FSG3Mode
from boundflow.runtime.fsg4_b3_explicit_counters import EXPECTED_B3C_FIXED_COUNTERS
from boundflow.runtime.fsg4_b3_same_solver_timing import fsg4_b3_timing_run_from_dict
from scripts import run_fsg4_b3_counter_diagnostic as diagnostic
from scripts import run_fsg4_b3_same_solver_timing as b3_worker

WORKER_SCHEMA = "boundflow.fsg4-b4a-same-solver-worker/v1"
CONFIGURATIONS = ("B3", "B4-A")
CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *diagnostic.B3C_CODE_PATHS,
            "boundflow/runtime/fsg4_b4a_terminal_lower_adjoint_handoff.py",
            "scripts/run_fsg4_b4a_same_solver_worker.py",
            "scripts/run_fsg3_same_solver_timing.py",
        )
    )
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_revision() -> dict[str, str]:
    return {path: _file_sha256(REPOSITORY_ROOT / path) for path in CODE_PATHS}


def _one_mapping(value: object, label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], Mapping)
    ):
        raise ValueError(f"FSG4/B4-A {label} cardinality differs")
    return cast(Mapping[str, Any], value[0])


def _empty_list(value: object, label: str) -> None:
    if value != []:
        raise ValueError(f"FSG4/B4-A B3 {label} must be empty")


def _activation(
    configuration: str,
    diagnostics: Mapping[str, Any],
    *,
    actual_profile_counts: Mapping[str, int] | None,
) -> dict[str, object]:
    exports = _one_mapping(
        diagnostics.get("native_backward_export_metadata"), "native export"
    )
    handoffs = diagnostics.get("terminal_lower_adjoint_handoff_metadata")
    assemblies = diagnostics.get("terminal_export_assembly_metadata")
    if configuration == "B3":
        _empty_list(handoffs, "handoff metadata")
        _empty_list(assemblies, "assembly metadata")
        activation: dict[str, object] = {
            "terminal_lower_adjoint_handoff_count": 0,
            "terminal_export_crown_rerun_count": 1,
            "native_backward_export_hash": exports["export_hash"],
            "provider_callback_count": 0,
            "fallback_dispatch_count": 0,
            "performance_claimed": False,
        }
    else:
        handoff = _one_mapping(handoffs, "handoff metadata")
        handoff_payload = handoff.get("handoff")
        assembly = _one_mapping(assemblies, "assembly metadata")
        if (
            not isinstance(handoff_payload, Mapping)
            or handoff.get("optimizer_evaluation_count") != 10
            or handoff.get("optimizer_update_count") != 9
            or handoff.get("terminal_lower_adjoint_handoff_count") != 1
            or handoff.get("terminal_export_crown_rerun_count") != 0
            or handoff_payload.get("terminal_lower_adjoint_handoff_count") != 1
            or handoff_payload.get("provider_core_callback_count") != 0
            or handoff_payload.get("provider_compute_bounds_callback_count") != 0
            or handoff_payload.get("provider_update_bounds_callback_count") != 0
            or handoff_payload.get("fallback_dispatch_count") != 0
            or assembly.get("terminal_lower_adjoint_handoff_count") != 1
            or assembly.get("terminal_export_crown_rerun_count") != 0
            or assembly.get("provider_core_callback_count") != 0
            or assembly.get("provider_compute_bounds_callback_count") != 0
            or assembly.get("provider_update_bounds_callback_count") != 0
            or assembly.get("fallback_dispatch_count") != 0
            or assembly.get("handoff_hash") != handoff.get("runtime_handoff_hash")
            or handoff_payload.get("runtime_handoff_hash")
            != handoff.get("runtime_handoff_hash")
            or assembly.get("export_schema_version") != exports.get("schema_version")
        ):
            raise ValueError("FSG4/B4-A direct activation receipt differs")
        lineages = handoff_payload.get("lineages")
        lower_adjoints = handoff_payload.get("lower_adjoints")
        if (
            not isinstance(lineages, Mapping)
            or not isinstance(lower_adjoints, Mapping)
            or len(lineages) != 6
            or set(lineages) != set(lower_adjoints)
            or any(
                not isinstance(lineage, Mapping)
                or lineage.get("shape_source")
                != "correlation-parent-boundflow-operator"
                or lineage.get("kernel_shape_inferred") is not False
                or not isinstance(lineage.get("producer_op_ordinal"), int)
                or not lineage.get("producer_op_name")
                for lineage in lineages.values()
            )
        ):
            raise ValueError("FSG4/B4-A handoff lineage receipt differs")
        activation = {
            "terminal_lower_adjoint_handoff_count": 1,
            "terminal_export_crown_rerun_count": 0,
            "native_backward_export_hash": exports["export_hash"],
            "handoff_hash": handoff_payload["handoff_hash"],
            "assembly_hash": assembly["assembly_hash"],
            "lineage_count": 6,
            "lineage_hashes": {
                name: cast(Mapping[str, Any], value)["lineage_hash"]
                for name, value in sorted(lineages.items())
            },
            "provider_callback_count": 0,
            "fallback_dispatch_count": 0,
            "performance_claimed": False,
        }
    if actual_profile_counts is not None:
        if any(
            actual_profile_counts.get(name) != value
            for name, value in EXPECTED_B3C_FIXED_COUNTERS.items()
        ):
            raise ValueError("FSG4/B4-A profile counter receipt differs")
        activation.update(
            {
                "profile_counter_counts": dict(sorted(actual_profile_counts.items())),
                "profile_counter_counts_hash": canonical_hash(
                    dict(sorted(actual_profile_counts.items()))
                ),
                "forward_trace_build_count": actual_profile_counts[
                    "forward_trace_build_count"
                ],
            }
        )
    activation["activation_hash"] = canonical_hash(activation)
    return activation


@contextmanager
def _candidate_executor() -> Iterator[None]:
    original = diagnostic._use_prepared_executor

    @contextmanager
    def use_b4a(_configuration: str) -> Iterator[None]:
        with original("B4-A"):
            yield

    with diagnostic._patch_attribute(diagnostic, "_use_prepared_executor", use_b4a):
        yield


def _actual_profile_counts(
    run_payload: Mapping[str, Any], mode: FSG3Mode
) -> dict[str, int] | None:
    activation = run_payload.get("activation")
    if not isinstance(activation, Mapping):
        raise TypeError("FSG4/B4-A inherited activation differs")
    counts = activation.get("detailed_counts")
    if mode == FSG3Mode.CONTROL:
        if counts is not None:
            raise ValueError("FSG4/B4-A control counter receipt differs")
        return None
    if not isinstance(counts, Mapping) or any(
        not isinstance(name, str)
        or not isinstance(value, int)
        or isinstance(value, bool)
        for name, value in counts.items()
    ):
        raise TypeError("FSG4/B4-A profile counter payload differs")
    return {str(name): int(value) for name, value in counts.items()}


def _b3_namespace(args: argparse.Namespace, result: Path) -> argparse.Namespace:
    return argparse.Namespace(
        configuration="B3",
        mode=args.mode,
        run_id=args.run_id,
        block_index=args.block_index,
        sequence_position=args.sequence_position,
        benchmark_root=args.benchmark_root,
        abcrown_root=args.abcrown_root,
        model=args.model,
        property=args.property,
        result=result,
        prepare_static_request=bool(getattr(args, "prepare_static_request", False)),
        attribute_root_incomplete=bool(
            getattr(args, "attribute_root_incomplete", False)
        ),
        attribute_complete_prelude=bool(
            getattr(args, "attribute_complete_prelude", False)
        ),
        prepare_root_optimizer_warmup=bool(
            getattr(args, "prepare_root_optimizer_warmup", False)
        ),
    )


def _worker(args: argparse.Namespace) -> None:
    configuration = str(args.configuration)
    mode = FSG3Mode(args.mode)
    if configuration not in CONFIGURATIONS:
        raise ValueError("FSG4/B4-A worker configuration differs")
    with tempfile.TemporaryDirectory(prefix="boundflow-fsg4-b4a-base-worker-") as raw:
        base_result = Path(raw) / "b3-worker.json"
        if configuration == "B3":
            b3_worker._worker(_b3_namespace(args, base_result))
        else:
            with _candidate_executor():
                b3_worker._worker(_b3_namespace(args, base_result))
        base_envelope = json.loads(base_result.read_text(encoding="utf-8"))
    if (
        not isinstance(base_envelope, Mapping)
        or base_envelope.get("schema_version") != b3_worker.WORKER_ENVELOPE_SCHEMA
        or base_envelope.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B4-A inherited B3 worker envelope differs")
    run_payload = base_envelope.get("run")
    diagnostics = base_envelope.get("diagnostics")
    if not isinstance(run_payload, Mapping) or not isinstance(diagnostics, Mapping):
        raise TypeError("FSG4/B4-A inherited worker payload differs")
    actual_profile_counts = _actual_profile_counts(run_payload, mode)
    run = fsg4_b3_timing_run_from_dict(cast(Mapping[str, object], run_payload))
    if run.configuration.value != "B3" or run.mode != mode:
        raise ValueError("FSG4/B4-A inherited measurement identity differs")
    activation = _activation(
        configuration,
        diagnostics,
        actual_profile_counts=actual_profile_counts,
    )
    protocol: dict[str, object] = {
        "source_git_head": diagnostic._git("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "base_b3_protocol_identity": run.protocol_identity,
        "configuration": configuration,
        "feature": (
            "b3-terminal-export-crown-rerun"
            if configuration == "B3"
            else "b4a-terminal-lower-adjoint-handoff"
        ),
        "same_solver": True,
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    envelope = {
        "schema_version": WORKER_SCHEMA,
        "configuration": configuration,
        "mode": mode.value,
        "run": run.to_dict(),
        "activation": activation,
        "protocol": protocol,
        "diagnostics": diagnostics,
        "performance_claimed": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "configuration": configuration,
                "mode": mode.value,
                "core_wall_ns": run.metrics.core_wall_ns,
                "query_wall_ns": run.metrics.query_wall_ns,
                "activation_hash": activation["activation_hash"],
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
    parser.add_argument("--mode", choices=("control", "profile"), required=True)
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
    args.benchmark_root = args.benchmark_root.resolve()
    args.abcrown_root = args.abcrown_root.resolve()
    args.model = args.model.resolve()
    args.property = args.property.resolve()
    args.result = args.result.resolve()
    _worker(args)


if __name__ == "__main__":
    main()
