#!/usr/bin/env python3
"""Run one real-provider control or MR3 P-anchor bridge exact call."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=import-outside-toplevel,protected-access,import-error
# pylint: disable=missing-function-docstring,wrong-import-position
# pylint: disable=too-many-instance-attributes,too-few-public-methods
# pylint: disable=line-too-long,too-many-boolean-expressions

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_production_p_anchor_bridge import (  # noqa: E402
    MR3ProductionPAnchorBridgeV1,
    TARGET_CONV,
    TARGET_RELU,
    TARGET_START,
)
from boundflow.runtime.mr3_provider_hook_feasibility import (  # noqa: E402
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    VNNCOMP_COMMIT,
    canonical_hash,
)
from scripts.run_mr3_provider_hook_feasibility import (  # noqa: E402
    MODEL_SHA256,
    PROPERTY_SHA256,
    _git,
    _module_state,
    _phase_from_stack,
    _result_state,
    _sha256,
    _target_alpha,
    _tensor_state,
    _visited_domains,
    _walk_tensor_values,
)

WORKER_SCHEMA = "boundflow.mr3-production-p-anchor-bridge-worker/v1"


class _BridgeTracker:
    def __init__(
        self,
        torch_module: Any,
        *,
        mode: str,
        inject_failure_evaluation: int | None = None,
    ) -> None:
        self.torch = torch_module
        self.mode = mode
        self.inject_failure_evaluation = inject_failure_evaluation
        self.stack: list[int] = []
        self.active_outer = False
        self.current_evaluation: int | None = None
        self.outer_count = 0
        self.inner_states: list[list[dict[str, object]]] = []
        self.outer_state: list[dict[str, object]] | None = None
        self.final_alpha_state: dict[str, object] | None = None
        self.final_module_state: list[dict[str, object]] | None = None
        self.bridge: MR3ProductionPAnchorBridgeV1 | None = None
        self.bridge_receipt: dict[str, object] | None = None
        self.region_states: list[dict[str, object]] = []
        self.evaluation_trajectory: list[dict[str, object]] = []
        self.mutation_trajectory: list[dict[str, object]] = []
        self.final_clip_state: dict[str, object] | None = None
        self._clip_count = 0
        self.atomic_receipt: dict[str, object] | None = None
        self._provider_relu_bias: Any | None = None
        self._owner_pre_hash: str | None = None
        self._owner_pointer_hash: str | None = None
        self._owner_pre_versions: list[int] = []

    def _record_region(
        self, result: Any, *, lower_bias_override: Any | None = None
    ) -> None:
        try:
            lower_a = result[0][0][0]
            lower_bias = (
                result[1] if lower_bias_override is None else lower_bias_override
            )
        except (IndexError, TypeError) as error:
            raise ValueError("MR3 production region result differs") from error
        if not self.torch.is_tensor(lower_a) or not self.torch.is_tensor(lower_bias):
            raise TypeError("MR3 production region tensors are absent")
        self.region_states.append(
            {
                "evaluation_ordinal": self.current_evaluation,
                "lower_a": _tensor_state(lower_a, self.torch),
                "lower_bias": _tensor_state(lower_bias, self.torch),
            }
        )

    def _owner_snapshot(self, instance: Any) -> list[tuple[Any, Any, Any]]:
        snapshot: list[tuple[Any, Any, Any]] = []
        seen: set[int] = set()
        for node in instance.nodes():
            for attribute in ("alpha", "sparse_betas", "beta", "split_beta"):
                for tensor in _walk_tensor_values(
                    getattr(node, attribute, None), self.torch
                ):
                    if id(tensor) not in seen:
                        seen.add(id(tensor))
                        snapshot.append(
                            (tensor, tensor.detach(), tensor.detach().clone())
                        )
        semantic = [_tensor_state(tensor, self.torch) for tensor, _, _ in snapshot]
        self._owner_pre_hash = canonical_hash(semantic)
        self._owner_pointer_hash = canonical_hash(
            [int(tensor.data_ptr()) for tensor, _, _ in snapshot]
        )
        self._owner_pre_versions = [int(tensor._version) for tensor, _, _ in snapshot]
        return snapshot

    def _restore_owner_snapshot(
        self, snapshot: list[tuple[Any, Any, Any]]
    ) -> dict[str, object]:
        with self.torch.no_grad():
            for tensor, original_storage, frozen in snapshot:
                tensor.data = original_storage.data
                tensor.copy_(frozen)
        if any(not self.torch.equal(tensor, frozen) for tensor, _, frozen in snapshot):
            raise RuntimeError("MR3 provider owner state rollback differs")
        post_hash = canonical_hash(
            [_tensor_state(tensor, self.torch) for tensor, _, _ in snapshot]
        )
        pointer_hash = canonical_hash(
            [int(tensor.data_ptr()) for tensor, _, _ in snapshot]
        )
        version_deltas = [
            int(tensor._version) - before
            for (tensor, _, _), before in zip(snapshot, self._owner_pre_versions)
        ]
        content_exact = post_hash == self._owner_pre_hash
        pointer_exact = pointer_hash == self._owner_pointer_hash
        versions_advanced = all(delta >= 1 for delta in version_deltas)
        if not content_exact or not pointer_exact or not versions_advanced:
            raise RuntimeError(
                "MR3 provider owner rollback receipt differs: "
                f"content_exact={content_exact},pointer_exact={pointer_exact},"
                f"versions_advanced={versions_advanced}"
            )
        return {
            "owner_tensor_count": len(snapshot),
            "owner_content_hash_before": self._owner_pre_hash,
            "owner_content_hash_after": post_hash,
            "owner_pointer_hash_before": self._owner_pointer_hash,
            "owner_pointer_hash_after": pointer_hash,
            "version_delta_min": min(version_deltas, default=0),
            "version_delta_max": max(version_deltas, default=0),
        }

    @contextmanager
    def _node_bridge(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        relu = nodes.get(TARGET_RELU)
        conv = nodes.get(TARGET_CONV)
        if (
            relu is None
            or conv is None
            or not getattr(relu, "inputs", ())
            or relu.inputs[0] is not conv
        ):
            raise ValueError("MR3 production bridge topology differs")
        if self.mode == "bridge":
            self.bridge = MR3ProductionPAnchorBridgeV1()
        original_relu = relu.bound_backward
        original_conv = conv.bound_backward
        original_clip_alpha = relu.clip_alpha
        original_adam_step = self.torch.optim.Adam.step
        target_alpha = _target_alpha(instance, self.torch)

        def adam_step_wrapped(optimizer: Any, *args: Any, **kwargs: Any) -> Any:
            owns_target = any(
                parameter is target_alpha
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            if not self.active_outer or not owns_target:
                return original_adam_step(optimizer, *args, **kwargs)
            groups = [
                group
                for group in optimizer.param_groups
                if any(parameter is target_alpha for parameter in group["params"])
            ]
            if len(groups) != 1 or target_alpha.grad is None:
                raise ValueError("MR3 target Adam ownership differs")
            lr_used = float(groups[0]["lr"])
            result = original_adam_step(optimizer, *args, **kwargs)
            state = optimizer.state.get(target_alpha)
            if (
                not isinstance(state, dict)
                or not self.torch.is_tensor(state.get("exp_avg"))
                or not self.torch.is_tensor(state.get("exp_avg_sq"))
            ):
                raise ValueError("MR3 target Adam state differs")
            self.mutation_trajectory.append(
                {
                    "mutation_ordinal": len(self.mutation_trajectory),
                    "gradient": _tensor_state(target_alpha.grad, self.torch),
                    "alpha_pre_clamp": _tensor_state(target_alpha, self.torch),
                    "exp_avg": _tensor_state(state["exp_avg"], self.torch),
                    "exp_avg_sq": _tensor_state(state["exp_avg_sq"], self.torch),
                    "lr_used": lr_used,
                    "optimizer_step": float(state["step"].item()),
                }
            )
            return result

        def clip_alpha_wrapped() -> Any:
            result = original_clip_alpha()
            state = _tensor_state(target_alpha, self.torch)
            values = state["values"]
            if not isinstance(values, list):
                raise TypeError("MR3 target alpha values are absent")
            mask = {
                "zero_count": sum(float(value) == 0.0 for value in values),
                "one_count": sum(float(value) == 1.0 for value in values),
                "interior_count": sum(0.0 < float(value) < 1.0 for value in values),
            }
            if self._clip_count < 9:
                if len(self.mutation_trajectory) != self._clip_count + 1:
                    raise ValueError("MR3 target Adam/clip order differs")
                self.mutation_trajectory[-1]["alpha_post_clamp"] = state
                self.mutation_trajectory[-1]["clamp_mask"] = mask
            elif self._clip_count == 9:
                self.final_clip_state = {"alpha": state, "clamp_mask": mask}
            else:
                raise ValueError("MR3 target clip count differs")
            self._clip_count += 1
            return result

        def relu_wrapped(*args: Any, **kwargs: Any) -> Any:
            start = str(getattr(kwargs.get("start_node"), "name", ""))
            if self.current_evaluation is None or start != TARGET_START:
                return original_relu(*args, **kwargs)
            if self.mode == "provider":
                result = original_relu(*args, **kwargs)
                if self._provider_relu_bias is not None:
                    raise ValueError("MR3 provider ReLU bias staging differs")
                self._provider_relu_bias = result[1]
                return result
            beta_tensors = []
            for owner in (relu, conv):
                for attribute in ("sparse_betas", "beta", "split_beta"):
                    beta_tensors.extend(
                        _walk_tensor_values(getattr(owner, attribute, None), self.torch)
                    )
            assert self.bridge is not None
            self.bridge.begin_evaluation(self.current_evaluation)
            return self.bridge.route_relu(
                relu,
                args,
                kwargs,
                beta_tensors=tuple(beta_tensors),
            )

        def conv_wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.current_evaluation is None:
                return original_conv(*args, **kwargs)
            if self.mode == "provider":
                result = original_conv(*args, **kwargs)
                if self._provider_relu_bias is None:
                    raise ValueError("MR3 provider fused bias staging is absent")
                self._record_region(
                    result,
                    lower_bias_override=self._provider_relu_bias + result[1],
                )
                self._provider_relu_bias = None
                return result
            assert self.bridge is not None
            result = self.bridge.route_conv(args)
            self._record_region(result)
            if self.current_evaluation == self.inject_failure_evaluation:
                raise RuntimeError("MR3 injected candidate failure")
            return result

        relu.bound_backward = relu_wrapped
        conv.bound_backward = conv_wrapped
        relu.clip_alpha = clip_alpha_wrapped
        self.torch.optim.Adam.step = adam_step_wrapped
        try:
            yield
        finally:
            relu.bound_backward = original_relu
            conv.bound_backward = original_conv
            relu.clip_alpha = original_clip_alpha
            self.torch.optim.Adam.step = original_adam_step
            if self._provider_relu_bias is not None:
                raise RuntimeError("MR3 provider fused bias staging leaked")

    @contextmanager
    def install(self, bounded_module: Any) -> Iterator[None]:
        original = bounded_module.compute_bounds

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            method = str(kwargs.get("method", "backward"))
            phase = _phase_from_stack(method)
            is_outer = (
                not self.stack
                and phase == "beta_split"
                and "optimized" in method.lower()
            )
            is_inner = (
                self.active_outer
                and len(self.stack) == 1
                and phase == "beta_split"
                and method.lower() == "backward"
            )
            if is_outer:
                if self.outer_count:
                    raise ValueError("MR3 production bridge outer call repeated")
                self.outer_count += 1
                self.active_outer = True
            owner_snapshot = (
                self._owner_snapshot(instance)
                if is_outer and self.mode == "bridge"
                else []
            )
            call_id = len(self.stack)
            self.stack.append(call_id)
            if is_inner:
                self.current_evaluation = len(self.inner_states)
            node_context = self._node_bridge(instance) if is_outer else nullcontext()
            try:
                with node_context:
                    result = original(instance, *args, **kwargs)
                if is_inner:
                    inner_state = _result_state(result, self.torch)
                    self.inner_states.append(inner_state)
                    lower_values = inner_state[0].get("values") if inner_state else None
                    if not isinstance(lower_values, list):
                        raise TypeError("MR3 inner lower values are absent")
                    self.evaluation_trajectory.append(
                        {
                            "evaluation_ordinal": len(self.evaluation_trajectory),
                            "lower": inner_state[0],
                            "aggregate_loss": -sum(
                                float(value) for value in lower_values
                            ),
                        }
                    )
                if is_outer:
                    self.outer_state = _result_state(result, self.torch)
                    self.final_alpha_state = _tensor_state(
                        _target_alpha(instance, self.torch), self.torch
                    )
                    self.final_module_state = _module_state(instance, self.torch)
                    if self.bridge is not None:
                        self.bridge_receipt = self.bridge.receipt().to_dict()
                    self.atomic_receipt = {
                        "exact_call_launch_count": 1,
                        "staged_emit_count": 1,
                        "atomic_commit_count": 1,
                        "rollback_count": 0,
                    }
                return result
            except Exception:
                if is_outer and self.mode == "bridge":
                    rollback = self._restore_owner_snapshot(owner_snapshot)
                    self.atomic_receipt = {
                        "exact_call_launch_count": 1,
                        "staged_emit_count": 0,
                        "atomic_commit_count": 0,
                        "rollback_count": 1,
                        **rollback,
                    }
                raise
            finally:
                if is_inner:
                    self.current_evaluation = None
                if self.stack.pop() != call_id:
                    raise RuntimeError("MR3 production bridge call stack differs")
                if is_outer:
                    self.active_outer = False

        bounded_module.compute_bounds = wrapped
        try:
            yield
        finally:
            bounded_module.compute_bounds = original


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints  # type: ignore[import-not-found]
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped]

    if not torch.cuda.is_available():
        raise RuntimeError("MR3 production bridge requires CUDA")
    tracker = _BridgeTracker(
        torch,
        mode=args.mode,
        inject_failure_evaluation=args.inject_failure_evaluation,
    )
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr3-bridge-property-"
    ) as workspace:
        isolated_property = Path(workspace) / args.property.name
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
        caught_failure: str | None = None
        with tracker.install(BoundedModule):
            solver = ABCrownSolver(str(args.model), config=config)
            try:
                result = solver.verify(
                    constraints=IOConstraints(vnnlib_path=str(isolated_property))
                )
            except RuntimeError as error:
                if (
                    args.mode != "bridge"
                    or args.inject_failure_evaluation is None
                    or str(error) != "MR3 injected candidate failure"
                ):
                    raise
                caught_failure = str(error)
    if caught_failure is not None:
        if (
            tracker.atomic_receipt is None
            or tracker.atomic_receipt.get("rollback_count") != 1
            or tracker.atomic_receipt.get("atomic_commit_count") != 0
        ):
            raise ValueError("MR3 injected rollback did not close")
        rollback_payload = {
            "schema_version": WORKER_SCHEMA,
            "mode": args.mode,
            "injected_failure_evaluation": args.inject_failure_evaluation,
            "caught_failure": caught_failure,
            "atomic_receipt": tracker.atomic_receipt,
            "timing_recorded": False,
            "performance_claimed": False,
        }
        rollback_payload["worker_hash"] = canonical_hash(rollback_payload)
        args.result_json.write_text(
            json.dumps(rollback_payload, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"rollback": "pass"}, sort_keys=True))
        return
    if (
        tracker.outer_count != 1
        or len(tracker.inner_states) != 10
        or tracker.outer_state is None
        or tracker.final_alpha_state is None
        or tracker.final_module_state is None
        or len(tracker.region_states) != 10
        or len(tracker.evaluation_trajectory) != 10
        or len(tracker.mutation_trajectory) != 9
        or tracker.final_clip_state is None
        or tracker._clip_count != 10
        or tracker.atomic_receipt is None
        or (args.mode == "bridge" and tracker.bridge_receipt is None)
    ):
        raise ValueError("MR3 production bridge worker did not close")
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "mode": args.mode,
        "source": {
            "abcrown_commit": _git(args.abcrown_root, "rev-parse", "HEAD"),
            "auto_lirpa_commit": _git(
                args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD"
            ),
            "vnncomp_commit": _git(args.benchmark_root, "rev-parse", "HEAD"),
            "model_sha256": _sha256(args.model),
            "property_sha256": _sha256(args.property),
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
        },
        "solver_result": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": _visited_domains(result),
        },
        "outer_result_state": tracker.outer_state,
        "inner_result_states": tracker.inner_states,
        "final_target_alpha_state": tracker.final_alpha_state,
        "final_module_state": tracker.final_module_state,
        "region_states": tracker.region_states,
        "evaluation_trajectory": tracker.evaluation_trajectory,
        "mutation_trajectory": tracker.mutation_trajectory,
        "final_clip_state": tracker.final_clip_state,
        "bridge_receipt": tracker.bridge_receipt,
        "atomic_receipt": tracker.atomic_receipt,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    payload["worker_hash"] = canonical_hash(payload)
    args.result_json.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"mode": args.mode, "status": str(result.status)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--mode", choices=("provider", "bridge"), required=True)
    parser.add_argument("--inject-failure-evaluation", type=int)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    if args.inject_failure_evaluation is not None and (
        args.mode != "bridge" or args.inject_failure_evaluation not in range(10)
    ):
        parser.error("failure injection requires bridge mode and ordinal 0..9")
    if (
        _git(args.abcrown_root, "rev-parse", "HEAD") != ABCROWN_COMMIT
        or _git(args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD")
        != AUTO_LIRPA_COMMIT
        or _git(args.benchmark_root, "rev-parse", "HEAD") != VNNCOMP_COMMIT
        or _sha256(args.model) != MODEL_SHA256
        or _sha256(args.property) != PROPERTY_SHA256
    ):
        raise ValueError("MR3 production bridge frozen input differs")
    _worker(args)


if __name__ == "__main__":
    main()
