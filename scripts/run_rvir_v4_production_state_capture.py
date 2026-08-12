#!/usr/bin/env python3
"""Generate or replay corrected RVIR-v4 αβ-CROWN production-state capture."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,import-outside-toplevel,protected-access
# pylint: disable=missing-function-docstring,line-too-long,import-error
# pylint: disable=too-many-instance-attributes
# pylint: disable=duplicate-code
# pylint: disable=unsubscriptable-object

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import FrameType
from typing import Any, cast, Iterator, Mapping, Sequence

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-production-capture-artifact/v2"
WORKER_SCHEMA_VERSION = "boundflow.rvir-v4-production-capture-worker/v2"
OPTIMIZER_WORKER_SCHEMA_VERSION = "boundflow.rvir-v4-optimizer-step-capture-worker/v1"
LEGACY_ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-production-capture-artifact/v1"
LEGACY_WORKER_SCHEMA_VERSION = "boundflow.rvir-v4-production-capture-worker/v1"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
AUTO_LIRPA_COMMIT = "5a098e8f9fb5786a428a024981d833d303921f2d"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
MODEL_RELATIVE_PATH = "benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
PROPERTY_RELATIVE_PATH = (
    "benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/"
    "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
PROPERTY_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
CAPTURE_FILE = "capture.pt"
ARTIFACT_FILES = (
    CAPTURE_FILE,
    "calls.json",
    "cores.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CODE_PATHS = (
    "boundflow/runtime/rvir_v4_production_state.py",
    "scripts/run_rvir_v4_production_state_capture.py",
)


def canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _code_revision() -> dict[str, str]:
    root = _repo_root()
    return {path: file_sha256(root / path) for path in CODE_PATHS}


def _code_paths_clean() -> bool:
    root = _repo_root()
    return not _git_value(root, "status", "--porcelain=v1", "--", *CODE_PATHS)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_code_provenance(manifest: Mapping[str, Any]) -> None:
    root = _repo_root()
    source_head = manifest.get("source_git_head")
    revisions = manifest.get("code_revision")
    if not isinstance(source_head, str) or not isinstance(revisions, Mapping):
        raise ValueError("RVIR-v4 capture source provenance differs")
    if _git_value(root, "rev-parse", "HEAD") == source_head:
        observed = _code_revision()
    else:
        observed = {}
        for path in CODE_PATHS:
            blob = subprocess.run(
                ("git", "show", f"{source_head}:{path}"),
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
            observed[path] = _sha256_bytes(blob)
    if dict(revisions) != observed:
        raise ValueError("RVIR-v4 capture source code revision differs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--benchmark-root", type=Path, required=True)
    worker.add_argument("--abcrown-root", type=Path, required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--result", type=Path, required=True)
    worker.add_argument("--optimizer-step-trace", action="store_true")
    return parser.parse_args()


def _phase_from_stack(method: str) -> tuple[str, str]:
    external = "unclassified_compute_bounds"
    frame: FrameType = sys._getframe(1)
    try:
        for _ in range(20):
            parent = frame.f_back
            if parent is None:
                break
            frame = parent
            filename = frame.f_code.co_filename.replace("\\", "/")
            function = frame.f_code.co_name
            if function == "update_bounds_core":
                return "beta_split", "activation_bab_bound"
            if "/input_split/" in filename:
                return "beta_split", "input_bab_bound"
            if "incomplete_verifier" in filename:
                return "initial_crown", "incomplete_verification"
            if "beta_CROWN_solver" in filename:
                normalized = method.lower().replace("_", "-")
                return (
                    "alpha_optimize" if "optimized" in normalized else "initial_crown",
                    "alpha_crown_initialization",
                )
    finally:
        del frame
    if "optimized" in method.lower().replace("_", "-"):
        return "alpha_optimize", external
    return "unclassified", external


def _tensor_axes(prefix: Sequence[str], rank: int) -> tuple[str, ...]:
    if rank < len(prefix):
        return tuple(f"axis_{index}" for index in range(rank))
    return tuple(prefix) + tuple(
        f"feature_axis_{index}" for index in range(rank - len(prefix))
    )


def _semantic_component(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


def _tensor_rows(
    value: object, path: str, torch_module: Any
) -> list[dict[str, object]]:
    from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

    if torch_module.is_tensor(value):
        tensor = cast(Any, value)
        return [
            {
                "path": path,
                "shape": [int(dimension) for dimension in tensor.shape],
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "content_sha256": production_tensor_sha256(tensor),
            }
        ]
    if isinstance(value, Mapping):
        rows: list[dict[str, object]] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            rows.extend(_tensor_rows(item, f"{path}.{key}", torch_module))
        return rows
    if isinstance(value, (tuple, list)):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_tensor_rows(item, f"{path}[{index}]", torch_module))
        return rows
    return []


def _optimizer_policy(arguments_module: Any, core_kwargs: Mapping[str, Any]) -> Any:
    from boundflow.runtime.rvir_v4_production_state import ProductionOptimizerPolicyV4

    beta = arguments_module.Config["solver"]["beta-crown"]
    stop = core_kwargs.get("stop_criterion_func")
    stop_id = f"{getattr(stop, '__module__', '')}.{getattr(stop, '__qualname__', type(stop).__name__)}"
    return ProductionOptimizerPolicyV4(
        iteration=int(beta["iteration"]),
        alpha_learning_rate=float(beta["lr_alpha"]),
        beta_learning_rate=float(beta["lr_beta"]),
        bound_lower=True,
        bound_upper=False,
        fix_intermediate_bounds=bool(core_kwargs.get("fix_interm_bounds", True)),
        deterministic=bool(arguments_module.Config["general"]["deterministic_opt"]),
        stop_criterion_id=stop_id,
    )


def _state_data(value: object, label: str) -> Mapping[str, object]:
    if isinstance(value, ValueError):
        raise ValueError(f"RVIR-v4 {label} is unavailable")
    raw = getattr(value, "_data", None)
    if not isinstance(raw, Mapping):
        raise TypeError(f"RVIR-v4 {label} data differs")
    return raw


def _bounded_input_bounds(value: object) -> tuple[Any, Any]:
    perturbation = getattr(value, "ptb", None)
    lower = getattr(perturbation, "x_L", None)
    upper = getattr(perturbation, "x_U", None)
    if lower is None or upper is None:
        raise ValueError("RVIR-v4 production input bounds are unavailable")
    return lower, upper


def _build_core_pre_snapshot(
    pre_result: Any,
    *,
    net: Any,
    core_id: int,
    policy: Any,
) -> Any:
    from boundflow.runtime.rvir_v4_production_state import (
        ProductionStateBuilderV4,
        ProductionStateSnapshotV4,
        ProductionTensorOwnership,
        ProductionTensorRole,
        capture_alpha_layout_state_v4,
        capture_alpha_state_v4,
        capture_sparse_beta_state_v4,
        capture_split_history_v4,
    )

    builder = ProductionStateBuilderV4()
    lower, upper = _bounded_input_bounds(pre_result.new_x)
    builder.add(
        path="query/input/lower",
        role=ProductionTensorRole.INPUT_LOWER,
        axes=_tensor_axes(("domain",), lower.ndim),
        value=lower,
        ownership=ProductionTensorOwnership.READ_ONLY,
    )
    builder.add(
        path="query/input/upper",
        role=ProductionTensorRole.INPUT_UPPER,
        axes=_tensor_axes(("domain",), upper.ndim),
        value=upper,
        ownership=ProductionTensorOwnership.READ_ONLY,
    )
    if pre_result.c is None:
        raise ValueError("RVIR-v4 production linear spec is unavailable")
    builder.add(
        path="query/linear_spec",
        role=ProductionTensorRole.LINEAR_SPEC,
        axes=_tensor_axes(("domain", "spec"), pre_result.c.ndim),
        value=pre_result.c,
        ownership=ProductionTensorOwnership.READ_ONLY,
    )
    for node_name, bounds in sorted(pre_result.interm_bounds.items()):
        if not isinstance(bounds, (tuple, list)) or len(bounds) < 2:
            raise ValueError("RVIR-v4 intermediate bounds differ")
        for polarity, value, role in (
            ("lower", bounds[0], ProductionTensorRole.INTERMEDIATE_LOWER),
            ("upper", bounds[1], ProductionTensorRole.INTERMEDIATE_UPPER),
        ):
            builder.add(
                path=f"intermediate/{_semantic_component(node_name)}/{polarity}",
                role=role,
                axes=_tensor_axes(("domain",), value.ndim),
                value=value,
                ownership=ProductionTensorOwnership.READ_ONLY,
            )
    thresholds = pre_result.d_dict["thresholds"]
    builder.add(
        path="query/decision_threshold",
        role=ProductionTensorRole.DECISION_THRESHOLD,
        axes=_tensor_axes(("domain",), thresholds.ndim),
        value=thresholds,
        ownership=ProductionTensorOwnership.READ_ONLY,
    )
    capture_alpha_state_v4(
        cast(
            Mapping[str, Mapping[str, Any]],
            _state_data(pre_result.alphas_by_layer, "alpha"),
        ),
        builder,
    )
    capture_alpha_layout_state_v4(net.net.nodes(), builder)
    capture_sparse_beta_state_v4(
        _state_data(pre_result.betas_by_layer, "beta"), builder
    )
    history = capture_split_history_v4(pre_result.d_dict["history"])
    snapshot = ProductionStateSnapshotV4(
        snapshot_id=f"core:{core_id:06d}:pre",
        tensors=builder.finish(),
        history=history,
        optimizer_policy=policy,
    )
    snapshot.validate()
    return snapshot


def _build_core_post_snapshot(
    pre_snapshot: Any,
    core_result: Any,
    *,
    core_id: int,
) -> Any:
    from boundflow.runtime.rvir_v4_production_state import (
        ProductionStateBuilderV4,
        ProductionStateSnapshotV4,
        ProductionTensorOwnership,
        capture_alpha_state_v4,
        capture_sparse_beta_state_v4,
    )

    builder = ProductionStateBuilderV4()
    capture_alpha_state_v4(
        cast(
            Mapping[str, Mapping[str, Any]],
            _state_data(core_result.working_alpha, "post alpha"),
        ),
        builder,
    )
    capture_sparse_beta_state_v4(
        _state_data(core_result.working_beta, "post beta"), builder
    )
    mutable = tuple(
        tensor
        for tensor in builder.finish()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    )
    read_only = tuple(
        tensor
        for tensor in pre_snapshot.tensors
        if tensor.ownership != ProductionTensorOwnership.MUTABLE_COPY_OUT
    )
    snapshot = ProductionStateSnapshotV4(
        snapshot_id=f"core:{core_id:06d}:post",
        tensors=tuple(sorted(read_only + mutable, key=lambda item: item.semantic_path)),
        history=pre_snapshot.history,
        optimizer_policy=pre_snapshot.optimizer_policy,
    )
    snapshot.validate()
    return snapshot


class _CaptureObserver:
    def __init__(
        self,
        torch_module: Any,
        arguments_module: Any,
        *,
        capture_optimizer_steps: bool = False,
    ) -> None:
        self.torch = torch_module
        self.arguments = arguments_module
        self.capture_optimizer_steps = capture_optimizer_steps
        self.calls: list[dict[str, Any]] = []
        self.cores: list[dict[str, Any]] = []
        self.optimizer_step_traces: list[dict[str, object]] = []
        self._call_stack: list[int] = []
        self._active_core_id: int | None = None
        self._core_policies: dict[int, Any] = {}
        self._core_mutation_policies: dict[int, Any] = {}
        self._optimizer_rows: dict[int, list[dict[str, Any]]] = {}
        self._optimizer_update_counts: dict[int, int] = {}
        self._active_adam: Any = None
        self._pending_optimizer_call_id: int | None = None

    def _capture_mutation_policy(self, instance: Any) -> None:
        from boundflow.runtime.rvir_v4_optimizer_mutation import (
            ProductionMutationPolicyV4,
            capture_production_optimizer_controls_v4,
        )

        core_id = self._active_core_id
        if core_id is None or core_id not in self._core_policies:
            raise RuntimeError("RVIR-v4 optimizer core policy is unavailable")
        bound_opts = getattr(instance, "bound_opts", None)
        optimize_bound_args = (
            None
            if not isinstance(bound_opts, Mapping)
            else bound_opts.get("optimize_bound_args")
        )
        if not isinstance(optimize_bound_args, Mapping):
            raise TypeError("RVIR-v4 live optimize-bound controls differ")
        cut_used = getattr(instance, "cut_used", False)
        if not isinstance(cut_used, bool):
            raise TypeError("RVIR-v4 live cut-used flag differs")
        controls = capture_production_optimizer_controls_v4(
            optimize_bound_args, cuts_enabled=cut_used
        )
        policy = ProductionMutationPolicyV4(
            production=self._core_policies[core_id], controls=controls
        )
        policy.validate()
        if core_id in self._core_mutation_policies:
            raise RuntimeError("RVIR-v4 optimizer mutation policy repeats")
        self._core_mutation_policies[core_id] = policy

    def _begin_optimizer_evaluation(
        self,
        *,
        call_id: int,
        parent_call_id: int,
        state_tensors: tuple[Any, ...],
    ) -> dict[str, Any]:
        core_id = self._active_core_id
        if core_id is None or core_id not in self._core_mutation_policies:
            raise RuntimeError("RVIR-v4 optimizer evaluation lacks live policy")
        if self._active_adam is None or self._pending_optimizer_call_id is not None:
            raise RuntimeError("RVIR-v4 optimizer evaluation/Adam ordering differs")
        raw_param_groups = getattr(self._active_adam, "param_groups", None)
        if not isinstance(raw_param_groups, list) or len(raw_param_groups) != 2:
            raise ValueError("RVIR-v4 optimizer parameter-group count differs")
        param_groups: list[Any] = raw_param_groups
        first_group = cast(dict[str, Any], param_groups[0])
        second_group = cast(dict[str, Any], param_groups[1])
        learning_rates = [first_group.get("lr"), second_group.get("lr")]
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in learning_rates
        ):
            raise TypeError("RVIR-v4 optimizer learning-rate fields differ")
        rows = self._optimizer_rows.setdefault(core_id, [])
        updates_before = self._optimizer_update_counts.setdefault(core_id, 0)
        row: dict[str, Any] = {
            "core_id": core_id,
            "call_id": call_id,
            "parent_call_id": parent_call_id,
            "evaluation_ordinal": len(rows),
            "updates_before": updates_before,
            "update_after": False,
            "optimizer_step_ordinal": None,
            "alpha_learning_rate": float(cast(float, learning_rates[0])),
            "beta_learning_rate": float(cast(float, learning_rates[1])),
            "state_tensors": state_tensors,
        }
        rows.append(row)
        return row

    @staticmethod
    def _result_lower(result: object, torch_module: Any) -> Any:
        if not isinstance(result, (tuple, list)) or not result:
            raise TypeError("RVIR-v4 optimizer evaluation result differs")
        lower = result[0]
        if not torch_module.is_tensor(lower):
            raise TypeError("RVIR-v4 optimizer evaluation lower differs")
        return lower

    def _finalize_optimizer_trace(self, core_id: int) -> None:
        from boundflow.runtime.rvir_v4_optimizer_mutation import (
            ProductionOptimizerStepTraceV4,
            ProductionOptimizerStepV4,
            production_optimizer_step_trace_to_payload_v4,
        )
        from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

        policy = self._core_mutation_policies.get(core_id)
        rows = self._optimizer_rows.get(core_id, [])
        if policy is None or len(rows) != policy.evaluation_count:
            raise ValueError("RVIR-v4 optimizer production trace cardinality differs")
        steps = []
        for row in rows:
            lower = row.get("lower")
            state_tensors = row.get("state_tensors")
            if not self.torch.is_tensor(lower) or not isinstance(state_tensors, tuple):
                raise TypeError("RVIR-v4 optimizer captured raw tensors differ")
            lower_tensor = cast(Any, lower)
            steps.append(
                ProductionOptimizerStepV4(
                    core_id=int(row["core_id"]),
                    call_id=int(row["call_id"]),
                    parent_call_id=int(row["parent_call_id"]),
                    evaluation_ordinal=int(row["evaluation_ordinal"]),
                    updates_before=int(row["updates_before"]),
                    update_after=bool(row["update_after"]),
                    optimizer_step_ordinal=cast(
                        int | None, row["optimizer_step_ordinal"]
                    ),
                    alpha_learning_rate=float(row["alpha_learning_rate"]),
                    beta_learning_rate=float(row["beta_learning_rate"]),
                    state_tensors=state_tensors,
                    lower=lower_tensor,
                    lower_sha256=production_tensor_sha256(lower_tensor),
                )
            )
        trace = ProductionOptimizerStepTraceV4(policy, tuple(steps))
        trace.validate()
        self.optimizer_step_traces.append(
            production_optimizer_step_trace_to_payload_v4(trace)
        )
        if self._pending_optimizer_call_id != steps[-1].call_id:
            raise RuntimeError("RVIR-v4 final evaluation/update boundary differs")
        self._pending_optimizer_call_id = None
        self._active_adam = None

    @contextmanager
    def instrument_adam(self) -> Iterator[None]:
        if not self.capture_optimizer_steps:
            yield
            return
        adam = self.torch.optim.Adam
        original_init = adam.__init__
        original_step = adam.step

        def wrapped_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            original_init(instance, *args, **kwargs)
            if self._active_core_id is not None:
                if self._active_adam is not None:
                    raise RuntimeError("RVIR-v4 optimizer Adam instance repeats")
                self._active_adam = instance

        def wrapped_step(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if self._active_core_id is None or instance is not self._active_adam:
                return original_step(instance, *args, **kwargs)
            core_id = self._active_core_id
            pending = self._pending_optimizer_call_id
            rows = self._optimizer_rows.get(core_id, [])
            if pending is None or not rows or rows[-1]["call_id"] != pending:
                raise RuntimeError("RVIR-v4 Adam step lacks preceding evaluation")
            expected = self._optimizer_update_counts.get(core_id, 0)
            if rows[-1]["updates_before"] != expected:
                raise RuntimeError("RVIR-v4 Adam step ordinal differs")
            result = original_step(instance, *args, **kwargs)
            rows[-1]["update_after"] = True
            rows[-1]["optimizer_step_ordinal"] = expected
            self._optimizer_update_counts[core_id] = expected + 1
            self._pending_optimizer_call_id = None
            return result

        adam.__init__ = wrapped_init
        adam.step = wrapped_step
        try:
            yield
        finally:
            adam.__init__ = original_init
            adam.step = original_step

    @contextmanager
    def instrument_compute(self, bounded_module: Any) -> Iterator[None]:
        from boundflow.runtime.rvir_v4_production_state import (
            ProductionTensorRole,
            capture_module_alpha_beta_state_v4,
        )

        original = bounded_module.compute_bounds

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            call_id = len(self.calls)
            method = str(kwargs.get("method", "backward"))
            phase, external_phase = _phase_from_stack(method)
            before = capture_module_alpha_beta_state_v4(
                instance.nodes(), require_beta=phase == "beta_split"
            )
            optimizer_row: dict[str, Any] | None = None
            if (
                self.capture_optimizer_steps
                and phase == "beta_split"
                and self._active_core_id is not None
            ):
                if not self._call_stack:
                    self._capture_mutation_policy(instance)
                elif len(self._call_stack) == 1:
                    optimizer_row = self._begin_optimizer_evaluation(
                        call_id=call_id,
                        parent_call_id=self._call_stack[-1],
                        state_tensors=before,
                    )
            row: dict[str, Any] = {
                "call_id": call_id,
                "parent_call_id": self._call_stack[-1] if self._call_stack else None,
                "depth": len(self._call_stack),
                "core_id": self._active_core_id,
                "phase": phase,
                "external_phase": external_phase,
                "method": method,
                "bound_lower": bool(kwargs.get("bound_lower", True)),
                "bound_upper": bool(kwargs.get("bound_upper", True)),
                "pre_state": [tensor.metadata() for tensor in before],
                "pre_alpha_count": sum(
                    tensor.role == ProductionTensorRole.ALPHA for tensor in before
                ),
                "pre_beta_value_count": sum(
                    tensor.role == ProductionTensorRole.BETA_VALUE for tensor in before
                ),
            }
            self.calls.append(row)
            self._call_stack.append(call_id)
            try:
                result = original(instance, *args, **kwargs)
                row["result_tensors"] = _tensor_rows(result, "result", self.torch)
                if optimizer_row is not None:
                    lower = self._result_lower(result, self.torch)
                    optimizer_row["lower"] = lower.detach().cpu().contiguous().clone()
                    self._pending_optimizer_call_id = call_id
                return result
            finally:
                after = capture_module_alpha_beta_state_v4(
                    instance.nodes(), require_beta=phase == "beta_split"
                )
                row["post_state"] = [tensor.metadata() for tensor in after]
                row["post_alpha_count"] = sum(
                    tensor.role == ProductionTensorRole.ALPHA for tensor in after
                )
                row["post_beta_value_count"] = sum(
                    tensor.role == ProductionTensorRole.BETA_VALUE for tensor in after
                )
                if self._call_stack.pop() != call_id:
                    raise RuntimeError("RVIR-v4 compute call lineage differs")

        bounded_module.compute_bounds = wrapped
        try:
            yield
        finally:
            bounded_module.compute_bounds = original

    @contextmanager
    def instrument_core(self, stage_solve_module: Any) -> Iterator[None]:
        from boundflow.runtime.rvir_v4_production_state import (
            diff_production_state_v4,
            production_snapshot_to_payload_v4,
            production_tensor_sha256,
        )

        original = stage_solve_module.update_bounds_core

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            core_id = len(self.cores)
            if self._active_core_id is not None:
                raise RuntimeError("RVIR-v4 production cores overlap")
            pre_result = kwargs.get("pre_result")
            if pre_result is None:
                raise ValueError("RVIR-v4 core pre_result is unavailable")
            policy = _optimizer_policy(self.arguments, kwargs)
            if self.capture_optimizer_steps:
                self._core_policies[core_id] = policy
            net = kwargs.get("net")
            if net is None:
                raise ValueError("RVIR-v4 core network is unavailable")
            pre = _build_core_pre_snapshot(
                pre_result, net=net, core_id=core_id, policy=policy
            )
            self._active_core_id = core_id
            try:
                result = original(*args, **kwargs)
                if self.capture_optimizer_steps:
                    self._finalize_optimizer_trace(core_id)
            finally:
                self._active_core_id = None
            post = _build_core_post_snapshot(pre, result, core_id=core_id)
            mutations = diff_production_state_v4(pre, post)
            decision = result.branching_decision
            row = {
                "core_id": core_id,
                "pre_snapshot": production_snapshot_to_payload_v4(pre),
                "post_snapshot": production_snapshot_to_payload_v4(post),
                "mutations": [mutation.__dict__ for mutation in mutations],
                "lower": result.lb.detach().cpu().contiguous().clone(),
                "upper": result.ub.detach().cpu().contiguous().clone(),
                "lower_sha256": production_tensor_sha256(result.lb),
                "upper_sha256": production_tensor_sha256(result.ub),
                "branching_decision": [
                    [int(layer), int(index)]
                    for layer, index in decision.branching_decision
                ],
                "branching_points": (
                    None
                    if decision.branching_points is None
                    else decision.branching_points.detach().cpu().tolist()
                ),
                "split_depth": int(decision.split_depth),
                "batch_size": int(decision.batch_size),
                "n_verified": int(result.n_verified),
                "n_splits": int(result.n_splits),
            }
            self.cores.append(row)
            return result

        stage_solve_module.update_bounds_core = wrapped
        try:
            yield
        finally:
            stage_solve_module.update_bounds_core = original


def _visited_domains(result: Any) -> list[int]:
    stats = getattr(result, "stats", None)
    if not isinstance(stats, dict) or not isinstance(stats.get("bab"), list):
        return []
    return [
        int(row[2])
        for row in stats["bab"]
        if isinstance(row, (tuple, list)) and len(row) >= 3
    ]


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(_repo_root()))
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints  # type: ignore[import-not-found]
    import arguments  # type: ignore[import-not-found]
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped]
    from activation_split import stage_solve  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise RuntimeError("RVIR-v4 production capture requires CUDA")
    if _git_value(args.abcrown_root, "rev-parse", "HEAD") != ABCROWN_COMMIT:
        raise ValueError("RVIR-v4 alpha-beta-CROWN commit differs")
    if (
        _git_value(args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD")
        != AUTO_LIRPA_COMMIT
    ):
        raise ValueError("RVIR-v4 auto_LiRPA commit differs")
    if _git_value(args.benchmark_root, "rev-parse", "HEAD") != VNNCOMP_COMMIT:
        raise ValueError("RVIR-v4 VNN-COMP commit differs")
    observer = _CaptureObserver(
        torch, arguments, capture_optimizer_steps=args.optimizer_step_trace
    )
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-property-") as workspace:
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
        with (
            observer.instrument_compute(BoundedModule),
            observer.instrument_adam(),
            observer.instrument_core(stage_solve),
        ):
            solver = ABCrownSolver(str(args.model), config=config)
            result = solver.verify(
                constraints=IOConstraints(vnnlib_path=str(isolated_property))
            )
    payload: dict[str, Any] = {
        "schema_version": (
            OPTIMIZER_WORKER_SCHEMA_VERSION
            if args.optimizer_step_trace
            else WORKER_SCHEMA_VERSION
        ),
        "source": {
            "abcrown_commit": ABCROWN_COMMIT,
            "auto_lirpa_commit": AUTO_LIRPA_COMMIT,
            "vnncomp_commit": VNNCOMP_COMMIT,
            "model_relative_path": MODEL_RELATIVE_PATH,
            "property_relative_path": PROPERTY_RELATIVE_PATH,
            "model_sha256": file_sha256(args.model),
            "property_sha256": file_sha256(args.property),
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
            "property_cache": "cold_isolated_copy",
            "performance_claimed": False,
        },
        "solver_result": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": _visited_domains(result),
        },
        "calls": observer.calls,
        "cores": observer.cores,
        "performance_claimed": False,
    }
    if args.optimizer_step_trace:
        payload["optimizer_step_traces"] = observer.optimizer_step_traces
    args.result.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.result)
    print(
        canonical_json(
            {
                "status": payload["solver_result"]["status"],
                "call_count": len(observer.calls),
                "core_count": len(observer.cores),
                "optimizer_step_trace_count": len(observer.optimizer_step_traces),
            }
        )
    )


def _load_capture(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("RVIR-v4 capture root differs")
    return value


def _projections(
    capture: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    import torch

    from boundflow.runtime.rvir_v4_production_state import (
        ProductionTensorRole,
        diff_production_state_v4,
        production_snapshot_from_payload_v4,
        production_tensor_sha256,
    )

    calls_raw = capture.get("calls")
    cores_raw = capture.get("cores")
    if not isinstance(calls_raw, list) or not isinstance(cores_raw, list):
        raise TypeError("RVIR-v4 capture call/core rows differ")
    calls = cast(list[dict[str, Any]], calls_raw)
    cores = cast(list[dict[str, Any]], cores_raw)
    capture_schema = capture.get("schema_version")
    if capture_schema not in {WORKER_SCHEMA_VERSION, LEGACY_WORKER_SCHEMA_VERSION}:
        raise ValueError("RVIR-v4 capture worker schema differs")
    layout_required = capture_schema == WORKER_SCHEMA_VERSION
    call_projection = {
        "schema_version": capture_schema,
        "calls": calls,
        "performance_claimed": False,
    }
    core_rows: list[dict[str, object]] = []
    for raw in cores:
        pre_raw = raw.get("pre_snapshot")
        post_raw = raw.get("post_snapshot")
        if not isinstance(pre_raw, Mapping) or not isinstance(post_raw, Mapping):
            raise TypeError("RVIR-v4 core snapshot rows differ")
        pre = production_snapshot_from_payload_v4(pre_raw)
        post = production_snapshot_from_payload_v4(post_raw)
        mutations = diff_production_state_v4(pre, post)
        if [mutation.__dict__ for mutation in mutations] != raw.get("mutations"):
            raise ValueError("RVIR-v4 mutation semantic replay differs")
        lower = raw.get("lower")
        upper = raw.get("upper")
        if not torch.is_tensor(lower) or not torch.is_tensor(upper):
            raise TypeError("RVIR-v4 core result tensors differ")
        lower_tensor = cast(torch.Tensor, lower)
        upper_tensor = cast(torch.Tensor, upper)
        if production_tensor_sha256(lower_tensor) != raw.get(
            "lower_sha256"
        ) or production_tensor_sha256(upper_tensor) != raw.get("upper_sha256"):
            raise ValueError("RVIR-v4 core result digest differs")
        pre_beta_count = sum(
            tensor.role == ProductionTensorRole.BETA_VALUE for tensor in pre.tensors
        )
        alpha_shape_count = sum(
            tensor.role == ProductionTensorRole.ALPHA_FEATURE_SHAPE
            for tensor in pre.tensors
        )
        alpha_index_count = sum(
            tensor.role == ProductionTensorRole.ALPHA_FEATURE_INDEX
            for tensor in pre.tensors
        )
        changed_count = sum(mutation.changed for mutation in mutations)
        core_row: dict[str, object] = {
            "core_id": int(raw["core_id"]),
            "pre_snapshot_hash": pre.stable_hash(),
            "post_snapshot_hash": post.stable_hash(),
            "tensor_count": len(pre.tensors),
            "history_entry_count": len(pre.history),
            "beta_value_tensor_count": pre_beta_count,
            "mutation_receipt_count": len(mutations),
            "changed_mutation_count": changed_count,
            "result_shape": list(lower_tensor.shape),
            "lower_sha256": raw["lower_sha256"],
            "upper_sha256": raw["upper_sha256"],
            "branching_decision": raw["branching_decision"],
            "branching_points": raw["branching_points"],
            "split_depth": int(raw["split_depth"]),
            "batch_size": int(raw["batch_size"]),
            "n_verified": int(raw["n_verified"]),
            "n_splits": int(raw["n_splits"]),
        }
        if layout_required:
            core_row["alpha_feature_shape_tensor_count"] = alpha_shape_count
            core_row["alpha_feature_index_tensor_count"] = alpha_index_count
        core_rows.append(core_row)
    core_projection = {
        "schema_version": capture_schema,
        "cores": core_rows,
        "performance_claimed": False,
    }
    phase_counts = {
        phase: sum(call["phase"] == phase for call in calls)
        for phase in ("initial_crown", "alpha_optimize", "beta_split", "unclassified")
    }
    beta_calls = [call for call in calls if call["phase"] == "beta_split"]
    all_beta_visible = bool(beta_calls) and all(
        int(call["pre_beta_value_count"]) > 0 and int(call["post_beta_value_count"]) > 0
        for call in beta_calls
    )
    passed = (
        len(calls) == 24
        and phase_counts
        == {
            "initial_crown": 12,
            "alpha_optimize": 1,
            "beta_split": 11,
            "unclassified": 0,
        }
        and len(core_rows) == 1
        and all_beta_visible
        and all(row["beta_value_tensor_count"] != 0 for row in core_rows)
        and all(row["history_entry_count"] != 0 for row in core_rows)
        and (
            not layout_required
            or all(row["alpha_feature_shape_tensor_count"] == 6 for row in core_rows)
        )
        and (
            not layout_required
            or all(row["alpha_feature_index_tensor_count"] == 16 for row in core_rows)
        )
    )
    summary: dict[str, object] = {
        "status": "validated_corrected_capture" if passed else "no_go",
        "workload_id": "cifar10_resnet:000",
        "call_count": len(calls),
        "phase_call_counts": phase_counts,
        "core_count": len(core_rows),
        "all_beta_split_calls_expose_plural_sparse_betas": all_beta_visible,
        "core_beta_value_tensor_counts": [
            row["beta_value_tensor_count"] for row in core_rows
        ],
        "core_history_entry_counts": [row["history_entry_count"] for row in core_rows],
        "core_mutation_receipt_counts": [
            row["mutation_receipt_count"] for row in core_rows
        ],
        "core_changed_mutation_counts": [
            row["changed_mutation_count"] for row in core_rows
        ],
        "frozen_state_evaluation_admitted": False,
        "optimizer_replacement_admitted": False,
        "b2_same_solver_timing_admitted": False,
        "performance_claimed": False,
    }
    if layout_required:
        summary["core_alpha_feature_shape_tensor_counts"] = [
            row["alpha_feature_shape_tensor_count"] for row in core_rows
        ]
        summary["core_alpha_feature_index_tensor_counts"] = [
            row["alpha_feature_index_tensor_count"] for row in core_rows
        ]
    summary["summary_hash"] = canonical_hash(summary)
    if not passed:
        raise ValueError("RVIR-v4 corrected production capture gate failed")
    return call_projection, core_projection, summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "call_count": summary["call_count"],
        "core_count": summary["core_count"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 Corrected Production-State Capture\n\n"
        "This artifact captures the 24-call compute tree and the typed update_bounds_core "
        "pre/post ownership boundary, including plural SparseBeta val/loc/sign/bias and "
        "history consistency. It makes no performance or replacement claim.\n"
    )


def _external_env() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _validate_inputs(
    benchmark_root: Path, abcrown_root: Path, abcrown_python: Path
) -> None:
    checks = (
        _git_value(benchmark_root, "rev-parse", "HEAD") == VNNCOMP_COMMIT,
        _git_value(abcrown_root, "rev-parse", "HEAD") == ABCROWN_COMMIT,
        _git_value(abcrown_root / "auto_LiRPA", "rev-parse", "HEAD")
        == AUTO_LIRPA_COMMIT,
        abcrown_python.is_file(),
        file_sha256(benchmark_root / MODEL_RELATIVE_PATH) == MODEL_SHA256,
        file_sha256(benchmark_root / PROPERTY_RELATIVE_PATH) == PROPERTY_SHA256,
    )
    if not all(checks):
        raise ValueError("RVIR-v4 production capture source inputs differ")


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError(
            "RVIR-v4 capture code paths must be clean before formal generation"
        )
    artifact_dir = args.artifact_dir.resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = args.benchmark_root.resolve()
    abcrown_root = args.abcrown_root.resolve()
    abcrown_python = Path(os.path.abspath(args.abcrown_python))
    _validate_inputs(benchmark_root, abcrown_root, abcrown_python)
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-capture-") as temporary:
        result_path = Path(temporary) / CAPTURE_FILE
        completed = subprocess.run(
            (
                str(abcrown_python),
                str(Path(__file__).resolve()),
                "worker",
                "--benchmark-root",
                str(benchmark_root),
                "--abcrown-root",
                str(abcrown_root),
                "--model",
                str(benchmark_root / MODEL_RELATIVE_PATH),
                "--property",
                str(benchmark_root / PROPERTY_RELATIVE_PATH),
                "--result",
                str(result_path),
            ),
            cwd=_repo_root(),
            env=_external_env(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                f"RVIR-v4 capture worker failed: {completed.stdout[-12000:]}"
            )
        shutil.copy2(result_path, artifact_dir / CAPTURE_FILE)
        print(completed.stdout.strip()[-3000:], flush=True)
    capture = _load_capture(artifact_dir / CAPTURE_FILE)
    calls, cores, summary = _projections(capture)
    _write_json(artifact_dir / "calls.json", calls)
    _write_json(artifact_dir / "cores.json", cores)
    _write_json(artifact_dir / "summary.json", summary)
    result = _replay_result(summary)
    (artifact_dir / "replay_stdout.txt").write_text(
        canonical_json(result) + "\n", encoding="utf-8"
    )
    (artifact_dir / "README.md").write_text(_readme(), encoding="utf-8")
    root = _repo_root()
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_git_head": _git_value(root, "rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "files": {name: file_sha256(artifact_dir / name) for name in ARTIFACT_FILES},
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact_dir / "manifest.json", manifest)
    return result


def _replay(artifact_dir: Path) -> dict[str, object]:
    manifest = _load_json(artifact_dir / "manifest.json")
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    if (
        manifest.get("schema_version")
        not in {ARTIFACT_SCHEMA_VERSION, LEGACY_ARTIFACT_SCHEMA_VERSION}
        or manifest.get("manifest_hash") != canonical_hash(semantic_manifest)
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 capture manifest envelope differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 capture artifact file inventory differs")
    for name, digest in files.items():
        if file_sha256(artifact_dir / name) != digest:
            raise ValueError("RVIR-v4 capture artifact file digest differs")
    calls, cores, summary = _projections(_load_capture(artifact_dir / CAPTURE_FILE))
    if _load_json(artifact_dir / "calls.json") != calls:
        raise ValueError("RVIR-v4 call projection replay differs")
    if _load_json(artifact_dir / "cores.json") != cores:
        raise ValueError("RVIR-v4 core projection replay differs")
    if _load_json(artifact_dir / "summary.json") != summary:
        raise ValueError("RVIR-v4 summary semantic replay differs")
    if manifest.get("summary_hash") != summary["summary_hash"]:
        raise ValueError("RVIR-v4 manifest summary projection differs")
    result = _replay_result(summary)
    if (artifact_dir / "replay_stdout.txt").read_text(
        encoding="utf-8"
    ) != canonical_json(result) + "\n":
        raise ValueError("RVIR-v4 replay stdout differs")
    if (artifact_dir / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 README differs")
    return result


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _worker(args)
        return
    result = (
        _generate(args)
        if args.command == "generate"
        else _replay(args.artifact_dir.resolve())
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
