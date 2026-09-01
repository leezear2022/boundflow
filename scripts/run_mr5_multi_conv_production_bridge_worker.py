#!/usr/bin/env python3
"""Run one native-provider or MR5 three-site production correctness worker."""

# pylint: disable=protected-access,import-error,import-outside-toplevel
# pylint: disable=missing-function-docstring,wrong-import-position
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,too-few-public-methods
# pylint: disable=line-too-long
# pylint: disable=duplicate-code
# mypy: disable-error-code=import-untyped

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterator, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import (  # noqa: E402
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    VNNCOMP_COMMIT,
    canonical_hash,
)
from boundflow.runtime.mr5_multi_conv_production_bridge import (  # noqa: E402
    MR5MultiConvProductionBridgeV1,
    MR5_SITE_NODES,
    MR5_SITE_ORDER,
    MR5_TARGET_START,
)
from scripts.run_mr3_production_p_anchor_bridge_worker import (  # noqa: E402
    _BridgeTracker,
)
from scripts.run_mr3_provider_hook_feasibility import (  # noqa: E402
    MODEL_SHA256,
    PROPERTY_SHA256,
    _git,
    _sha256,
    _tensor_state,
    _visited_domains,
    _walk_tensor_values,
)

WORKER_SCHEMA = "boundflow.mr5-multi-conv-production-bridge-worker/v1"


class _MR5BridgeTracker(_BridgeTracker):
    """Reuse MR3 outer-state/optimizer ownership while routing three sites."""

    def __init__(
        self,
        torch_module: Any,
        *,
        mode: str,
        inject_failure_evaluation: int | None = None,
    ) -> None:
        super().__init__(
            torch_module,
            mode=mode,
            inject_failure_evaluation=inject_failure_evaluation,
        )
        self.provider_relu_biases: dict[str, Any] = {}

    def _record_site(
        self, site: str, result: Any, *, lower_bias_override: Any | None = None
    ) -> None:
        try:
            lower_a = result[0][0][0]
            lower_bias = (
                result[1] if lower_bias_override is None else lower_bias_override
            )
        except (IndexError, TypeError) as error:
            raise ValueError(f"MR5 {site} production result differs") from error
        if not self.torch.is_tensor(lower_a) or not self.torch.is_tensor(lower_bias):
            raise TypeError(f"MR5 {site} production tensors are absent")
        self.region_states.append(
            {
                "site": site,
                "evaluation_ordinal": self.current_evaluation,
                "lower_a": _tensor_state(lower_a, self.torch),
                "lower_bias": _tensor_state(lower_bias, self.torch),
            }
        )

    @contextmanager
    def _node_bridge(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        site_nodes: dict[str, tuple[Any, Any]] = {}
        for site, (relu_name, conv_name) in MR5_SITE_NODES.items():
            relu = nodes.get(relu_name)
            conv = nodes.get(conv_name)
            if (
                relu is None
                or conv is None
                or not getattr(relu, "inputs", ())
                or relu.inputs[0] is not conv
            ):
                raise ValueError(f"MR5 {site} production topology differs")
            site_nodes[site] = (relu, conv)
        native_methods = {
            site: (relu.bound_backward, conv.bound_backward)
            for site, (relu, conv) in site_nodes.items()
        }

        # Parent context owns optimizer/clamp tracking and restores the original C2 hooks.
        with super()._node_bridge(instance):
            parent_wrappers = {
                site: (relu.bound_backward, conv.bound_backward)
                for site, (relu, conv) in site_nodes.items()
            }
            if self.mode == "bridge":
                self.bridge = cast(Any, MR5MultiConvProductionBridgeV1())

            def relu_wrapper(site: str, relu: Any, conv: Any, original: Any):
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    start = str(getattr(kwargs.get("start_node"), "name", ""))
                    if self.current_evaluation is None or start != MR5_TARGET_START:
                        return original(*args, **kwargs)
                    if self.mode == "provider":
                        result = original(*args, **kwargs)
                        if site in self.provider_relu_biases:
                            raise ValueError(
                                f"MR5 {site} provider ReLU staging differs"
                            )
                        self.provider_relu_biases[site] = result[1]
                        return result
                    assert isinstance(self.bridge, MR5MultiConvProductionBridgeV1)
                    if site == MR5_SITE_ORDER[0]:
                        self.bridge.begin_evaluation(self.current_evaluation)
                    beta_tensors = []
                    seen: set[int] = set()
                    for owner in (relu, conv):
                        for attribute in ("sparse_betas", "beta", "split_beta"):
                            for tensor in _walk_tensor_values(
                                getattr(owner, attribute, None), self.torch
                            ):
                                if id(tensor) not in seen:
                                    seen.add(id(tensor))
                                    beta_tensors.append(tensor)
                    return self.bridge.route_relu(
                        site,
                        relu,
                        args,
                        kwargs,
                        beta_tensors=tuple(beta_tensors),
                    )

                return wrapped

            def conv_wrapper(site: str, original: Any):
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    if self.current_evaluation is None:
                        return original(*args, **kwargs)
                    if self.mode == "provider":
                        result = original(*args, **kwargs)
                        staged = self.provider_relu_biases.pop(site, None)
                        if staged is None:
                            raise ValueError(f"MR5 {site} provider bias staging absent")
                        self._record_site(
                            site, result, lower_bias_override=staged + result[1]
                        )
                        return result
                    assert isinstance(self.bridge, MR5MultiConvProductionBridgeV1)
                    result = self.bridge.route_conv(site, args)
                    self._record_site(site, result)
                    if (
                        self.current_evaluation == self.inject_failure_evaluation
                        and site == "C1"
                    ):
                        raise RuntimeError("MR5 injected candidate failure")
                    return result

                return wrapped

            for site, (relu, conv) in site_nodes.items():
                relu.bound_backward = relu_wrapper(
                    site, relu, conv, native_methods[site][0]
                )
                conv.bound_backward = conv_wrapper(site, native_methods[site][1])
            try:
                yield
            finally:
                for site, (relu, conv) in site_nodes.items():
                    relu.bound_backward, conv.bound_backward = parent_wrappers[site]
                if self.provider_relu_biases:
                    raise RuntimeError("MR5 provider bias staging leaked")


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints  # type: ignore[import-not-found]
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped]

    if not torch.cuda.is_available():
        raise RuntimeError("MR5 production bridge requires CUDA")
    tracker = _MR5BridgeTracker(
        torch,
        mode=args.mode,
        inject_failure_evaluation=args.inject_failure_evaluation,
    )
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr5-bridge-property-"
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
                    or str(error) != "MR5 injected candidate failure"
                ):
                    raise
                caught_failure = str(error)
    if caught_failure is not None:
        if (
            tracker.atomic_receipt is None
            or tracker.atomic_receipt.get("rollback_count") != 1
            or tracker.atomic_receipt.get("atomic_commit_count") != 0
        ):
            raise ValueError("MR5 injected rollback did not close")
        payload: dict[str, object] = {
            "schema_version": WORKER_SCHEMA,
            "mode": args.mode,
            "injected_failure_evaluation": args.inject_failure_evaluation,
            "injected_failure_site": "C1",
            "caught_failure": caught_failure,
            "atomic_receipt": tracker.atomic_receipt,
            "timing_recorded": False,
            "performance_claimed": False,
        }
        payload["worker_hash"] = canonical_hash(payload)
        args.result_json.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
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
        or len(tracker.region_states) != 30
        or len(tracker.evaluation_trajectory) != 10
        or len(tracker.mutation_trajectory) != 9
        or tracker.final_clip_state is None
        or tracker._clip_count != 10
        or tracker.atomic_receipt is None
        or (args.mode == "bridge" and tracker.bridge_receipt is None)
    ):
        raise ValueError("MR5 production bridge worker did not close")
    payload = {
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
            "site_order": list(MR5_SITE_ORDER),
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
        raise ValueError("MR5 production bridge frozen input differs")
    _worker(args)


if __name__ == "__main__":
    main()
