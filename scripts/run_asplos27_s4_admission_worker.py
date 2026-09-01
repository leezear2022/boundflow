#!/usr/bin/env python3
"""Capture one fresh real-provider S4-0 mutable-state admission receipt."""

# pylint: disable=wrong-import-position,import-error,import-outside-toplevel
# pylint: disable=too-many-locals,too-many-statements,too-many-arguments
# pylint: disable=protected-access,missing-function-docstring,no-member
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=line-too-long,unidiomatic-typecheck

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_rvir_v4_production_state_capture as capture_runner
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.asplos27-s4-admission-worker/v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class _AdmissionCaptured(RuntimeError):
    """Private sentinel used to stop the solver before provider core execution."""


class _AdmissionObserver:
    def __init__(
        self,
        *,
        model: Path,
        run_ordinal: int,
        torch_module: Any,
        arguments_module: Any,
        program: Any,
        module: Any,
    ) -> None:
        self.model = model
        self.run_ordinal = run_ordinal
        self.torch = torch_module
        self.arguments = arguments_module
        self.program = program
        self.module = module
        self.intercept_count = 0
        self.provider_compute_bounds_callback_count = 0
        self.provider_update_bounds_callback_count = 0
        self.payload: dict[str, object] | None = None

    def _profile_provider_callback(self, frame, event, _value) -> None:  # type: ignore[no-untyped-def]
        if event != "call":
            return
        filename = frame.f_code.co_filename.replace("\\", "/")
        name = frame.f_code.co_name
        if name == "compute_bounds" and "/auto_LiRPA/" in filename:
            self.provider_compute_bounds_callback_count += 1
        if name == "update_bounds" and filename.endswith(
            "/complete_verifier/beta_CROWN_solver.py"
        ):
            self.provider_update_bounds_callback_count += 1

    def capture(self, net: Any, pre_result: Any, kwargs: Mapping[str, Any]) -> None:
        from boundflow.runtime.asplos27_s4_mutable_state_admission import (
            extract_s4_live_mutable_sources_v1,
            prepare_s4_mutable_state_admission_v1,
        )
        from boundflow.runtime.r3_structured_owner_custom_backward import (
            compile_r31_full_region_plan_v1,
        )
        from boundflow.runtime.rvir_v4_pre_state_initializer import (
            initialize_rvir_v4_native_pre_state,
        )

        if self.intercept_count:
            raise RuntimeError("S4-0 admission provider core intercept repeats")
        self.intercept_count += 1
        if (
            kwargs.get("fix_interm_bounds") is not True
            or kwargs.get("enable_decision_precompute") is not True
            or kwargs.get("enable_clip_domains") is not False
            or kwargs.get("precompute_bfs_flag") is not False
            or kwargs.get("is_multitree_bab") is not False
            or type(kwargs.get("branching_heuristic")).__name__ != "KfsbBranching"
        ):
            raise ValueError("S4-0 admission production flags differ")
        policy = capture_runner._optimizer_policy(self.arguments, kwargs)
        snapshot = capture_runner._build_core_pre_snapshot(
            pre_result,
            net=net,
            core_id=0,
            policy=policy,
        )
        mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
        plan = compile_r31_full_region_plan_v1(self.module, snapshot, mapping, TOPOLOGY)
        alpha_data = object.__getattribute__(
            object.__getattribute__(pre_result, "alphas_by_layer"), "_data"
        )
        beta_data = object.__getattribute__(
            object.__getattribute__(pre_result, "betas_by_layer"), "_data"
        )
        provider_structure = {
            "alpha_data": (
                f"{type(alpha_data).__module__}.{type(alpha_data).__qualname__}"
            ),
            "beta_data": (
                f"{type(beta_data).__module__}.{type(beta_data).__qualname__}"
            ),
            "slots": [
                {
                    "alpha_nested": (
                        f"{type(alpha_data[link.provider_activation]).__module__}."
                        f"{type(alpha_data[link.provider_activation]).__qualname__}"
                    ),
                    "alpha_tensor": (
                        f"{type(alpha_data[link.provider_activation][link.provider_start_node]).__module__}."
                        f"{type(alpha_data[link.provider_activation][link.provider_start_node]).__qualname__}"
                    ),
                    "beta_collection": (
                        f"{type(beta_data[link.provider_preactivation]).__module__}."
                        f"{type(beta_data[link.provider_preactivation]).__qualname__}"
                    ),
                    "beta_entry": (
                        f"{type(beta_data[link.provider_preactivation][0]).__module__}."
                        f"{type(beta_data[link.provider_preactivation][0]).__qualname__}"
                    ),
                    "beta_tensor": (
                        f"{type(object.__getattribute__(beta_data[link.provider_preactivation][0], 'val')).__module__}."
                        f"{type(object.__getattribute__(beta_data[link.provider_preactivation][0], 'val')).__qualname__}"
                    ),
                }
                for link in TOPOLOGY
            ],
        }
        previous_profile = sys.getprofile()
        sys.setprofile(self._profile_provider_callback)
        try:
            live_sources = extract_s4_live_mutable_sources_v1(pre_result, TOPOLOGY)
            exact_call_id = f"asplos27-s4-formal:{self.run_ordinal:03d}"
            prepared = prepare_s4_mutable_state_admission_v1(
                snapshot,
                TOPOLOGY,
                plan,
                live_sources,
                exact_call_id=exact_call_id,
            )
        finally:
            sys.setprofile(previous_profile)
        receipt = prepared.receipt
        receipt.validate()
        lease = prepared._live_lease
        if lease is None:
            raise RuntimeError("S4-0 admission lease was not published")
        retained_before_close = len(lease._source_rows)
        provider_projection = [
            {
                "semantic_path": row.semantic_path,
                "python_type": (
                    f"{type(row.tensor).__module__}.{type(row.tensor).__qualname__}"
                ),
                "shape": list(row.shape),
                "dtype": row.dtype,
                "device": row.device,
                "stride": list(row.stride),
                "storage_offset": row.storage_offset,
                "version": row.version,
                "requires_grad": row.requires_grad,
                "is_leaf": row.is_leaf,
                "content_hash": row.content_hash,
            }
            for row in lease._source_rows
        ]
        lease_state_before_close = lease._state
        prepared.close()
        retained_after_close = len(lease._source_rows)
        lease_state_after_close = lease._state
        self.payload = {
            "schema_version": WORKER_SCHEMA,
            "run_ordinal": self.run_ordinal,
            "provider": {
                "pre_result_type": (
                    f"{type(pre_result).__module__}.{type(pre_result).__qualname__}"
                ),
                "alpha_wrapper_type": (
                    f"{type(pre_result.alphas_by_layer).__module__}."
                    f"{type(pre_result.alphas_by_layer).__qualname__}"
                ),
                "beta_wrapper_type": (
                    f"{type(pre_result.betas_by_layer).__module__}."
                    f"{type(pre_result.betas_by_layer).__qualname__}"
                ),
                "structure": provider_structure,
                "live_projection": provider_projection,
            },
            "receipt": receipt.to_dict(),
            "lease": {
                "state_before_close": lease_state_before_close,
                "retained_tensor_count_before_close": retained_before_close,
                "state_after_close": lease_state_after_close,
                "retained_tensor_count_after_close": retained_after_close,
                "single_transfer_observed": False,
                "buffer_prepare_count": 0,
            },
            "counters": {
                "provider_core_intercept_count": self.intercept_count,
                "provider_core_execute_count": 0,
                "provider_compute_bounds_callback_count": (
                    self.provider_compute_bounds_callback_count
                ),
                "provider_update_bounds_callback_count": (
                    self.provider_update_bounds_callback_count
                ),
                "candidate_kernel_launch_count": (
                    receipt.candidate_kernel_launch_count
                ),
                "candidate_cuda_allocation_count": (
                    receipt.candidate_cuda_allocation_count
                ),
                "fallback_count": 0,
                "retry_count": 0,
                "mutation_count": 0,
            },
            "performance_claimed": False,
        }
        self.payload["worker_payload_hash"] = _canonical_hash(self.payload)

    @contextmanager
    def instrument(self, stage_solve: Any) -> Iterator[None]:
        original = stage_solve.update_bounds_core

        def replacement(*args: Any, **kwargs: Any) -> Any:
            net = kwargs.get("net", args[0] if args else None)
            pre_result = kwargs.get("pre_result", args[1] if len(args) > 1 else None)
            if net is None or pre_result is None:
                raise TypeError("S4-0 admission core call schema differs")
            self.capture(net, pre_result, kwargs)
            raise _AdmissionCaptured("S4-0 admission capture complete")

        stage_solve.update_bounds_core = replacement
        try:
            yield
        finally:
            stage_solve.update_bounds_core = original


def run(args: argparse.Namespace) -> dict[str, object]:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints  # type: ignore[import-not-found]
    import arguments  # type: ignore[import-not-found]
    from activation_split import stage_solve  # type: ignore[import-not-found]
    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.planner import plan_interval_ibp_v0

    capture_runner._validate_inputs(
        args.benchmark_root, args.abcrown_root, Path(sys.executable)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("S4-0 formal provider worker requires CUDA")
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    observer = _AdmissionObserver(
        model=args.model,
        run_ordinal=args.run_ordinal,
        torch_module=torch,
        arguments_module=arguments,
        program=program,
        module=module,
    )
    with tempfile.TemporaryDirectory(
        prefix="boundflow-s4-admission-property-"
    ) as temporary:
        isolated_property = Path(temporary) / args.property.name
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
        try:
            with observer.instrument(stage_solve):
                solver = ABCrownSolver(str(args.model), config=config)
                solver.verify(
                    constraints=IOConstraints(vnnlib_path=str(isolated_property))
                )
        except _AdmissionCaptured:
            pass
    if observer.payload is None or observer.intercept_count != 1:
        raise RuntimeError("S4-0 formal provider admission was not captured")
    source = {
        "abcrown_commit": capture_runner.ABCROWN_COMMIT,
        "auto_lirpa_commit": capture_runner.AUTO_LIRPA_COMMIT,
        "vnncomp_commit": capture_runner.VNNCOMP_COMMIT,
        "model_relative_path": capture_runner.MODEL_RELATIVE_PATH,
        "property_relative_path": capture_runner.PROPERTY_RELATIVE_PATH,
        "model_sha256": capture_runner.file_sha256(args.model),
        "property_sha256": capture_runner.file_sha256(args.property),
    }
    protocol = {
        "device": "cuda",
        "seed": 100,
        "max_iterations": 1,
        "batch_size": 64,
        "alpha_steps": 5,
        "beta_steps": 10,
        "property_cache": "cold_isolated_copy",
        "exact_call_id_template": "asplos27-s4-formal:{run_ordinal:03d}",
        "admission_per_process": 1,
        "buffer_prepare": False,
        "candidate_execute": False,
        "mutation": False,
        "performance_claimed": False,
    }
    result = {
        "schema_version": WORKER_SCHEMA,
        "source": source,
        "source_hash": _canonical_hash(source),
        "protocol": protocol,
        "protocol_hash": _canonical_hash(protocol),
        "admission": observer.payload,
        "performance_claimed": False,
    }
    result["raw_hash"] = _canonical_hash(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--run-ordinal", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    args.abcrown_root = args.abcrown_root.resolve()
    args.model = args.model.resolve()
    args.property = args.property.resolve()
    args.result = args.result.resolve()
    result = run(args)
    admission = result["admission"]
    if type(admission) is not dict:
        raise TypeError("S4-0 worker admission payload differs")
    receipt = admission.get("receipt")
    if type(receipt) is not dict:
        raise TypeError("S4-0 worker receipt payload differs")
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        _canonical_json(
            {
                "status": "captured",
                "run_ordinal": args.run_ordinal,
                "admission_hash": receipt["admission_hash"],
                "performance_claimed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
