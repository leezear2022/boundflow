#!/usr/bin/env python3
"""Run one capture-ready RVIR-v4 V4-3D live GPU return replacement."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,no-member,import-outside-toplevel
# pylint: disable=too-many-instance-attributes,missing-function-docstring,import-error
# pylint: disable=too-many-arguments

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterator, Mapping, MutableMapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_rvir_v4_production_state_capture as capture_runner
from scripts.run_rvir_v4_pre_state_artifact import EXPECTED_IDENTITY, TOPOLOGY

WORKER_SCHEMA = "boundflow.rvir-v4-live-return-worker/v1"


def _move_tensors(value: object, *, device: Any, dtype: Any) -> object:
    import torch

    if torch.is_tensor(value):
        return value.to(
            device=device, dtype=dtype if value.is_floating_point() else None
        )
    if isinstance(value, dict):
        return {
            key: _move_tensors(item, device=device, dtype=dtype)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_tensors(item, device=device, dtype=dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device=device, dtype=dtype) for item in value)
    return value


def _one_role(snapshot: Any, role: Any) -> Any:
    values = [tensor.value for tensor in snapshot.tensors if tensor.role == role]
    if len(values) != 1:
        raise ValueError(f"RVIR-v4 live capture requires one {role.value}")
    return values[0]


def _branch_trace(
    evaluation: Any, export: Any, core_result: Any, torch_module: Any
) -> dict[str, object]:
    lower_bounds = {
        name: bounds.lower_bound
        for name, bounds in core_result.working_interm_bounds.items()
    }
    upper_bounds = {
        name: bounds.upper_bound
        for name, bounds in core_result.working_interm_bounds.items()
    }
    lower_bounds["/49"] = core_result.lb
    upper_bounds["/49"] = core_result.ub
    return {
        "input": {
            "alphas": capture_runner._truth_value(
                core_result.working_alpha, "native_branch_alphas", torch_module
            ),
            "betas": capture_runner._truth_value(
                [
                    ValueError("Only nonlinear branching needs beta from this iter")
                    for _ in range(6)
                ],
                "native_branch_betas",
                torch_module,
            ),
            "cs": capture_runner._truth_value(
                core_result.c, "native_branch_cs", torch_module
            ),
            "history": capture_runner._truth_value(
                core_result.history, "native_branch_history", torch_module
            ),
            "lAs": capture_runner._truth_value(
                {"_data": export.l_as, "is_emptied": False},
                "native_branch_lAs",
                torch_module,
            ),
            "lower_bounds": capture_runner._truth_value(
                lower_bounds, "native_branch_lower_bounds", torch_module
            ),
            "mask": capture_runner._truth_value(
                evaluation.unstable_masks, "native_branch_mask", torch_module
            ),
            "thresholds": capture_runner._truth_value(
                core_result.thresholds, "native_branch_thresholds", torch_module
            ),
            "upper_bounds": capture_runner._truth_value(
                upper_bounds, "native_branch_upper_bounds", torch_module
            ),
        },
        "candidate_splits": [
            {"decision": [list(value) for value in decisions]}
            for decisions in evaluation.candidate_splits
        ],
        "candidate_child_lowers": [
            capture_runner._truth_value(value, "native_child_lower", torch_module)
            for value in evaluation.candidate_child_lowers
        ],
        "final_decision": {
            "decision": [list(value) for value in evaluation.final_decision],
            "points": None,
            "split_depth": 1,
            "batch_size": 6,
        },
        "provider_update_bounds_call_count": 0,
    }


class _LiveExecutor:
    def __init__(
        self,
        *,
        model: Path,
        torch_module: Any,
        arguments_module: Any,
        precompiled_program: Any = None,
        precompiled_module: Any = None,
        capture_payloads: bool = True,
    ):
        if (precompiled_program is None) != (precompiled_module is None):
            raise ValueError("RVIR-v4 live precompiled program/module must be paired")
        self.model = model
        self.torch = torch_module
        self.arguments = arguments_module
        self.precompiled_program = precompiled_program
        self.precompiled_module = precompiled_module
        self.capture_payloads = capture_payloads
        self.active = False
        self.core_count = 0
        self.provider_compute_bounds_callback_count = 0
        self.provider_update_bounds_callback_count = 0
        self.fallback_dispatch_count = 0
        self.core_payloads: list[dict[str, object]] = []
        self.post_payloads: list[dict[str, object]] = []
        self.commit_receipts: list[dict[str, object]] = []
        self.assembly_metadata: list[dict[str, object]] = []
        self.pre_state_identities: list[dict[str, str]] = []
        self.last_core_result: Any = None
        self.last_post_result: Any = None

    def execute(self, net: Any, pre_result: Any, kwargs: Mapping[str, Any]) -> Any:
        import torch

        from boundflow.frontends.onnx.frontend import import_onnx
        from boundflow.planner import plan_interval_ibp_v0
        from boundflow.runtime.native_alpha_beta_optimization_state import (
            build_native_alpha_beta_scope,
        )
        from boundflow.runtime.rvir_v4_atomic_copy_out import (
            stage_rvir_v4_live_atomic_copy_out,
        )
        from boundflow.runtime.rvir_v4_live_return import (
            assemble_rvir_v4_live_core_return,
            commit_rvir_v4_live_core_return,
        )
        from boundflow.runtime.rvir_v4_native_backward_export import (
            export_rvir_v4_native_backward,
        )
        from boundflow.runtime.rvir_v4_native_kfsb import (
            evaluate_rvir_v4_native_kfsb,
        )
        from boundflow.runtime.rvir_v4_native_optimizer import (
            execute_rvir_v4_native_optimizer_trace,
        )
        from boundflow.runtime.rvir_v4_optimizer_mutation import (
            capture_production_optimizer_controls_v4,
            ProductionMutationPolicyV4,
        )
        from boundflow.runtime.rvir_v4_pre_state_initializer import (
            initialize_rvir_v4_native_pre_state,
        )
        from boundflow.runtime.rvir_v4_production_state import ProductionTensorRole
        from boundflow.runtime.task_executor import InputSpec

        if self.active or self.core_count != 0:
            raise RuntimeError("RVIR-v4 live core execution repeats")
        self.active = True
        previous_profile = sys.getprofile()

        def forbid_provider_bound(frame, event, value) -> None:  # type: ignore[no-untyped-def]
            if previous_profile is not None:
                previous_profile(frame, event, value)
            if event != "call":
                return
            filename = frame.f_code.co_filename.replace("\\", "/")
            name = frame.f_code.co_name
            if name == "compute_bounds" and "/auto_LiRPA/" in filename:
                self.provider_compute_bounds_callback_count += 1
                raise RuntimeError("RVIR-v4 live provider compute_bounds forbidden")
            if name == "update_bounds" and filename.endswith(
                "/complete_verifier/beta_CROWN_solver.py"
            ):
                self.provider_update_bounds_callback_count += 1
                raise RuntimeError("RVIR-v4 live provider update_bounds forbidden")

        sys.setprofile(forbid_provider_bound)
        try:
            if (
                kwargs.get("fix_interm_bounds") is not True
                or kwargs.get("enable_decision_precompute") is not True
                or kwargs.get("enable_clip_domains") is not False
                or kwargs.get("precompute_bfs_flag") is not False
                or kwargs.get("is_multitree_bab") is not False
                or type(kwargs.get("branching_heuristic")).__name__ != "KfsbBranching"
            ):
                raise ValueError("RVIR-v4 live fixed production flags differ")
            production_policy = capture_runner._optimizer_policy(self.arguments, kwargs)
            pre_snapshot = capture_runner._build_core_pre_snapshot(
                pre_result,
                net=net,
                core_id=self.core_count,
                policy=production_policy,
            )
            beta_args = self.arguments.Config["solver"]["beta-crown"]
            net.net.set_bound_opts(
                {
                    "optimize_bound_args": {
                        "enable_beta_crown": beta_args["beta"],
                        "fix_interm_bounds": True,
                        "stop_criterion_func": kwargs["stop_criterion_func"],
                        "multi_spec_keep_func": kwargs["multi_spec_keep_func"],
                        "iteration": beta_args["iteration"],
                    },
                    "enable_opt_interm_bounds": beta_args["enable_opt_interm_bounds"],
                }
            )
            net.set_crown_bound_opts("beta")
            controls = capture_production_optimizer_controls_v4(
                cast(Mapping[str, Any], net.net.bound_opts["optimize_bound_args"]),
                cuts_enabled=False,
            )
            mutation_policy = ProductionMutationPolicyV4(
                production=production_policy,
                controls=controls,
            )
            mutation_policy.validate()
            mapping = initialize_rvir_v4_native_pre_state(
                pre_snapshot,
                TOPOLOGY,
            )
            if mapping.identity.topology_hash != EXPECTED_IDENTITY.topology_hash:
                raise ValueError("RVIR-v4 live frozen topology identity differs")
            self.pre_state_identities.append(mapping.identity.to_dict())
            execution_device = cast(torch.Tensor, pre_result.c).device
            execution_dtype = cast(torch.Tensor, pre_result.c).dtype
            mapping = mapping.to(device=execution_device, dtype=execution_dtype)
            input_lower = _one_role(pre_snapshot, ProductionTensorRole.INPUT_LOWER).to(
                device=execution_device, dtype=execution_dtype
            )
            input_upper = _one_role(pre_snapshot, ProductionTensorRole.INPUT_UPPER).to(
                device=execution_device, dtype=execution_dtype
            )
            objective = _one_role(pre_snapshot, ProductionTensorRole.LINEAR_SPEC).to(
                device=execution_device, dtype=execution_dtype
            )
            thresholds = _one_role(
                pre_snapshot, ProductionTensorRole.DECISION_THRESHOLD
            ).to(device=execution_device, dtype=execution_dtype)
            if self.precompiled_program is None:
                program = import_onnx(
                    str(self.model), do_shape_infer=True, normalize=True
                )
                module = plan_interval_ibp_v0(program)
            else:
                program = self.precompiled_program
                module = self.precompiled_module
            module.bindings = cast(
                dict[str, Any],
                _move_tensors(
                    module.bindings,
                    device=execution_device,
                    dtype=execution_dtype,
                ),
            )
            input_spec = InputSpec.box(
                value_name=program.graph.inputs[0],
                lower=input_lower,
                upper=input_upper,
            )
            policy = mutation_policy.to_native_policy()
            scope = build_native_alpha_beta_scope(
                module,
                input_spec,
                linear_spec_C=objective,
                relu_pre=mapping.relu_pre,
                relu_split_state=mapping.splits,
                policy=policy,
            )
            initial = mapping.to_native_state(scope)
            native = execute_rvir_v4_native_optimizer_trace(
                module,
                input_spec,
                linear_spec_C=objective,
                relu_pre=mapping.relu_pre,
                initial_state=initial,
                mutation_policy=mutation_policy,
            )
            terminal = type(initial)(
                scope=initial.scope,
                split_by_relu_input=initial.split_by_relu_input,
                alpha_by_relu_input=native.steps[-1].alpha_by_relu_input,
                beta_by_relu_input=native.steps[-1].beta_by_relu_input,
            )
            export = export_rvir_v4_native_backward(
                module=module,
                input_spec=input_spec,
                linear_spec_C=objective,
                relu_pre=mapping.relu_pre,
                terminal_state=terminal,
                topology=TOPOLOGY,
            )
            evaluation = evaluate_rvir_v4_native_kfsb(
                module=module,
                input_spec=input_spec,
                linear_spec_C=objective,
                thresholds=thresholds,
                terminal_state=terminal,
                topology=TOPOLOGY,
                backward_export=export,
            )
            d = cast(MutableMapping[str, object], pre_result.d_dict)
            host_candidate = {
                "history": d["history"],
                "depths": list(cast(list[object], d["depths"])),
                "thresholds": d["thresholds"],
            }
            staged = stage_rvir_v4_live_atomic_copy_out(
                pre=pre_snapshot,
                terminal_state=terminal,
                topology=TOPOLOGY,
                terminal_lower=export.lower,
                host_packet=d,
                host_packet_candidate=host_candidate,
                candidate_snapshot_id=f"core:{self.core_count:06d}:live-candidate",
            )
            assembly = assemble_rvir_v4_live_core_return(
                pre_result=pre_result,
                pre_snapshot=pre_snapshot,
                staged_copy_out=staged,
                backward_export=export,
                kfsb_evaluation=evaluation,
                topology=TOPOLOGY,
            )
            core_result, receipt = commit_rvir_v4_live_core_return(
                assembly,
                pre_snapshot=pre_snapshot,
                host_packet=d,
            )
            self.assembly_metadata.append(assembly.metadata())
            self.commit_receipts.append(receipt)
            self.last_core_result = core_result
            if self.capture_payloads:
                self.core_payloads.append(
                    capture_runner._capture_whole_core_truth(
                        core_result,
                        torch,
                        _branch_trace(evaluation, export, core_result, torch),
                    )
                )
            self.core_count += 1
            return core_result
        finally:
            sys.setprofile(previous_profile)
            self.active = False

    @contextmanager
    def instrument(
        self,
        *,
        stage_solve: Any,
        stage_postprocess: Any,
    ) -> Iterator[None]:
        original_core = stage_solve.update_bounds_core
        original_post = stage_postprocess.update_bounds_post

        def replacement_core(*args: Any, **kwargs: Any) -> Any:
            net = kwargs.get("net", args[0] if args else None)
            pre_result = kwargs.get("pre_result", args[1] if len(args) > 1 else None)
            if net is None or pre_result is None:
                raise TypeError("RVIR-v4 live core call schema differs")
            return self.execute(net, pre_result, kwargs)

        def wrapped_post(*args: Any, **kwargs: Any) -> Any:
            result = original_post(*args, **kwargs)
            self.last_post_result = result
            if self.capture_payloads:
                self.post_payloads.append(
                    capture_runner._capture_whole_post_truth(result, self.torch)
                )
            return result

        stage_solve.update_bounds_core = replacement_core
        stage_postprocess.update_bounds_post = wrapped_post
        try:
            yield
        finally:
            stage_solve.update_bounds_core = original_core
            stage_postprocess.update_bounds_post = original_post


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import (  # type: ignore[import-not-found]
        ABCrownSolver,
        ConfigBuilder,
        IOConstraints,
    )
    import arguments  # type: ignore[import-not-found]
    from activation_split import (  # type: ignore[import-not-found]
        stage_postprocess,
        stage_solve,
    )

    capture_runner._validate_inputs(
        args.benchmark_root, args.abcrown_root, Path(sys.executable)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("RVIR-v4 live return requires CUDA")
    executor = _LiveExecutor(
        model=args.model,
        torch_module=torch,
        arguments_module=arguments,
    )
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-live-property-") as raw:
        isolated_property = Path(raw) / args.property.name
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
        with executor.instrument(
            stage_solve=stage_solve,
            stage_postprocess=stage_postprocess,
        ):
            solver = ABCrownSolver(str(args.model), config=config)
            result = solver.verify(
                constraints=IOConstraints(vnnlib_path=str(isolated_property))
            )
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "source": {
            "abcrown_commit": capture_runner.ABCROWN_COMMIT,
            "auto_lirpa_commit": capture_runner.AUTO_LIRPA_COMMIT,
            "vnncomp_commit": capture_runner.VNNCOMP_COMMIT,
            "model_sha256": capture_runner.file_sha256(args.model),
            "property_sha256": capture_runner.file_sha256(args.property),
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
            "performance_claimed": False,
        },
        "solver_result": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": capture_runner._visited_domains(result),
        },
        "whole_core_results": executor.core_payloads,
        "whole_post_results": executor.post_payloads,
        "assembly_metadata": executor.assembly_metadata,
        "commit_receipts": executor.commit_receipts,
        "pre_state_identities": executor.pre_state_identities,
        "provider_core_callback_count": 0,
        "provider_compute_bounds_callback_count": (
            executor.provider_compute_bounds_callback_count
        ),
        "provider_update_bounds_callback_count": (
            executor.provider_update_bounds_callback_count
        ),
        "fallback_dispatch_count": executor.fallback_dispatch_count,
        "performance_claimed": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.result)
    print(
        json.dumps(
            {
                "status": payload["solver_result"],
                "core_count": len(executor.core_payloads),
                "post_count": len(executor.post_payloads),
                "provider_compute_bounds_callback_count": (
                    executor.provider_compute_bounds_callback_count
                ),
                "provider_update_bounds_callback_count": (
                    executor.provider_update_bounds_callback_count
                ),
                "performance_claimed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
