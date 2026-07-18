"""Optional, reversible query profiling adapter for an external αβ-CROWN run."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import MethodType
from typing import Any, Callable, Iterator, Mapping, Sequence, Tuple

import torch

from ..planner.materialization import BoundMethod, OptimizationStage
from .bab_query import BoundQuery, QueryCompatibilityKey
from .verification_profile import (
    VerificationCoverageReport,
    VerificationQueryProfile,
    write_verification_profiles_jsonl,
)

ABCROWN_ADAPTER_SCHEMA_VERSION = "boundflow.abcrown-adapter/v1"


def file_sha256(path: Path) -> str:
    """Hash a workload input without embedding it in the trace."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _argument_map(
    original: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return dict(inspect.signature(original).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _bound_options(bounded_module: Any) -> Mapping[str, Any]:
    options = getattr(bounded_module, "bound_opts", {})
    return options if isinstance(options, Mapping) else {}


def _beta_enabled(bounded_module: Any) -> bool:
    optimize = _bound_options(bounded_module).get("optimize_bound_args", {})
    return bool(
        isinstance(optimize, Mapping) and optimize.get("enable_beta_crown", False)
    )


def _method_kind(raw_method: object, *, beta_enabled: bool) -> BoundMethod:
    method = str(raw_method or "backward").strip().lower().replace("_", "-")
    optimized = "optimized" in method
    if beta_enabled and optimized:
        return BoundMethod.ALPHA_BETA_CROWN
    if "ibp" in method:
        return BoundMethod.IBP
    if "forward" in method:
        return BoundMethod.ALPHA_FORWARD if optimized else BoundMethod.FORWARD
    if optimized:
        return BoundMethod.ALPHA_CROWN
    return BoundMethod.CROWN


def _phase_from_stack() -> str:
    for frame in inspect.stack(context=0)[2:18]:
        filename = frame.filename.replace("\\", "/")
        if frame.function == "update_bounds_core":
            return "activation_bab_bound"
        if "/input_split/" in filename:
            return "input_bab_bound"
        if "incomplete_verifier" in filename:
            return "incomplete_verification"
        if "beta_CROWN_solver" in filename:
            return "alpha_crown_initialization"
    return "unclassified_compute_bounds"


def _optimization_stage(method: BoundMethod, solver_phase: str) -> OptimizationStage:
    if "bab" in solver_phase:
        return OptimizationStage.BAB_NODE_EVAL
    if method in {BoundMethod.ALPHA_CROWN, BoundMethod.ALPHA_FORWARD}:
        return OptimizationStage.ALPHA_OPTIMIZE
    return OptimizationStage.INFERENCE


def _normalize_external_op(node: Any) -> str | None:
    name = type(node).__name__.lower()
    if name in {
        "boundinput",
        "boundparams",
        "boundbuffers",
        "boundconstant",
    }:
        return None
    mapping = (
        ("conv", "conv2d"),
        ("linear", "linear"),
        ("gemm", "linear"),
        ("relu", "relu"),
        ("flatten", "flatten"),
        ("reshape", "reshape"),
        ("concat", "concat"),
        ("add", "add"),
        ("mul", "mul"),
    )
    for needle, normalized in mapping:
        if needle in name:
            return normalized
    class_name = type(node).__name__.removeprefix("Bound")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower() or None


def _layer_pattern(bounded_module: Any) -> Tuple[str, ...]:
    nodes_method = getattr(bounded_module, "nodes", None)
    if not callable(nodes_method):
        raise ValueError("external BoundedModule does not expose nodes()")
    pattern = tuple(
        normalized
        for node in nodes_method()
        if (normalized := _normalize_external_op(node)) is not None
    )
    if not pattern:
        raise ValueError("could not derive a supported layer pattern")
    return pattern


def _state_tensors(
    bounded_module: Any, attribute_names: Sequence[str]
) -> dict[str, torch.Tensor]:
    nodes_method = getattr(bounded_module, "nodes", None)
    if not callable(nodes_method):
        return {}
    found: dict[str, torch.Tensor] = {}
    for node_index, node in enumerate(nodes_method()):
        for attribute_name in attribute_names:
            value = getattr(node, attribute_name, None)
            if torch.is_tensor(value):
                found[f"{node_index}:{attribute_name}"] = value
                continue
            if isinstance(value, Mapping):
                for key, item in value.items():
                    tensor = _first_tensor(item)
                    if tensor is not None:
                        found[f"{node_index}:{attribute_name}:{key}"] = tensor
    return found


def _mapping_digest(values: Mapping[str, torch.Tensor]) -> str | None:
    if not values:
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(tensor).encode("utf-8"))
    return digest.hexdigest()


def _input_bounds(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    perturbation = getattr(input_tensor, "ptb", None)
    lower = getattr(perturbation, "x_L", None)
    upper = getattr(perturbation, "x_U", None)
    if torch.is_tensor(lower) and torch.is_tensor(upper):
        return lower, upper
    return input_tensor, input_tensor


def _input_region_digest(input_tensor: torch.Tensor) -> str:
    lower, upper = _input_bounds(input_tensor)
    digest = hashlib.sha256()
    digest.update(_tensor_sha256(lower).encode("utf-8"))
    digest.update(_tensor_sha256(upper).encode("utf-8"))
    return digest.hexdigest()


@dataclass
class ABCrownBoundQueryProfiler:  # pylint: disable=too-many-instance-attributes
    """Build PR-13 ``BoundQuery`` records at real ``compute_bounds`` calls."""

    model_structure_hash: str
    weight_version: str
    query_prefix: str = "abcrown"
    phase_resolver: Callable[[], str] = _phase_from_stack
    precondition_rejections: Tuple[str, ...] = ()
    queries: list[BoundQuery] = field(default_factory=list)
    profiles: list[VerificationQueryProfile] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.model_structure_hash or not self.weight_version:
            raise ValueError(
                "model_structure_hash and weight_version must be non-empty"
            )
        if not self.query_prefix:
            raise ValueError("query_prefix must be non-empty")

    def _record(  # pylint: disable=too-many-locals
        self,
        bounded_module: Any,
        original: Callable[..., Any],
        call_args: tuple[Any, ...],
        call_kwargs: Mapping[str, Any],
    ) -> None:
        values = _argument_map(original, call_args, call_kwargs)
        values.pop("self", None)
        input_tensor = _first_tensor(values.get("x"))
        if input_tensor is None:
            raise ValueError("compute_bounds call does not expose tensor input x")
        linear_spec = _first_tensor(values.get("C"))
        raw_method = values.get("method", call_kwargs.get("method", "backward"))
        beta_enabled = _beta_enabled(bounded_module)
        bound_method = _method_kind(raw_method, beta_enabled=beta_enabled)
        solver_phase = self.phase_resolver()
        stage = _optimization_stage(bound_method, solver_phase)
        pattern = _layer_pattern(bounded_module)

        alpha_tensors = _state_tensors(bounded_module, ("alpha",))
        beta_tensors = _state_tensors(
            bounded_module, ("sparse_beta", "beta", "split_beta")
        )
        alpha_version = _mapping_digest(alpha_tensors)
        beta_version = _mapping_digest(beta_tensors)
        split_state_present = bool(beta_tensors) or beta_enabled
        split_signature = beta_version or (
            "abcrown:beta-enabled:state-unresolved" if beta_enabled else "empty"
        )
        split_shapes = tuple(
            (name, tuple(int(dim) for dim in tensor.shape))
            for name, tensor in sorted(beta_tensors.items())
        )
        input_shape = tuple(int(dim) for dim in input_tensor.shape)
        spec_shape = (
            () if linear_spec is None else tuple(int(dim) for dim in linear_spec.shape)
        )
        dtype = str(input_tensor.dtype)
        device = str(input_tensor.device)
        requires_grad = bound_method in {
            BoundMethod.ALPHA_FORWARD,
            BoundMethod.ALPHA_CROWN,
            BoundMethod.ALPHA_BETA_CROWN,
        }
        source_options = {
            "adapter_schema_version": ABCROWN_ADAPTER_SCHEMA_VERSION,
            "external_method": str(raw_method),
            "solver_phase": solver_phase,
            "split_state_present": split_state_present,
            "identity_limitations": (
                []
                if beta_version is not None or not beta_enabled
                else ["split_state_values_unresolved"]
            ),
        }
        options_hash = hashlib.sha256(
            json.dumps(source_options, sort_keys=True).encode("utf-8")
        ).hexdigest()
        capability_class = (
            "alpha_beta_dense_split"
            if bound_method == BoundMethod.ALPHA_BETA_CROWN
            else "alpha_dense" if requires_grad else "original_bound_executor"
        )
        compatibility = QueryCompatibilityKey(
            model_structure_hash=self.model_structure_hash,
            weight_version=self.weight_version,
            bound_method=bound_method.value,
            optimization_stage=stage.value,
            requires_grad=requires_grad,
            input_value_name="abcrown_input",
            input_shape=input_shape,
            spec_shape=spec_shape,
            split_tensor_shapes=split_shapes,
            dtype=dtype,
            device=device,
            perturbation_signature="abcrown_explicit_box",
            execution_options_hash=options_hash,
            backend_capability_class=capability_class,
            numeric_policy="fp32_strict" if dtype == "torch.float32" else dtype,
        )
        sequence_number = len(self.queries)
        query = BoundQuery(
            query_id=f"{self.query_prefix}-{sequence_number:08d}",
            parent_query_id=None,
            sequence_number=sequence_number,
            example_idx=0,
            model_structure_hash=self.model_structure_hash,
            weight_version=self.weight_version,
            input_region_hash=_input_region_digest(input_tensor),
            output_spec_hash=(
                "none" if linear_spec is None else _tensor_sha256(linear_spec)
            ),
            split_signature=split_signature,
            bound_method=bound_method,
            optimization_stage=stage,
            requires_grad=requires_grad,
            alpha_state_version=alpha_version,
            beta_state_version=beta_version,
            cuts_version=None,
            dtype=dtype,
            device=device,
            numeric_policy=compatibility.numeric_policy,
            requested_outputs=("bounds",),
            compatibility_key=compatibility,
            execution_options=source_options,
        )
        query.validate()
        profile = VerificationQueryProfile.from_bound_query(
            query,
            solver_phase=solver_phase,
            layer_pattern=pattern,
            source="alpha-beta-CROWN",
            precondition_rejections=self.precondition_rejections,
        )
        self.queries.append(query)
        self.profiles.append(profile)

    @contextmanager
    def instrument(self, target: Any) -> Iterator["ABCrownBoundQueryProfiler"]:
        """Wrap a BoundedModule instance or class, restoring it exactly on exit."""

        if not hasattr(target, "compute_bounds"):
            raise TypeError("instrument target must expose compute_bounds")
        original = getattr(target, "compute_bounds")
        had_instance_override = not inspect.isclass(target) and (
            "compute_bounds" in getattr(target, "__dict__", {})
        )
        if inspect.isclass(target):

            def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
                self._record(instance, original, (instance, *args), kwargs)
                return original(instance, *args, **kwargs)

            setattr(target, "compute_bounds", wrapped)
        else:

            def wrapped_instance(_instance: Any, *args: Any, **kwargs: Any) -> Any:
                self._record(target, original, args, kwargs)
                return original(*args, **kwargs)

            setattr(target, "compute_bounds", MethodType(wrapped_instance, target))
        try:
            yield self
        finally:
            if inspect.isclass(target) or had_instance_override:
                setattr(target, "compute_bounds", original)
            else:
                delattr(target, "compute_bounds")

    def validate(self) -> None:
        """Prove one profile exists for every captured query and IDs are contiguous."""

        if len(self.queries) != len(self.profiles):
            raise ValueError("query/profile accounting mismatch")
        expected_ids = [
            f"{self.query_prefix}-{index:08d}" for index in range(len(self.queries))
        ]
        if [query.query_id for query in self.queries] != expected_ids:
            raise ValueError("external query IDs are not contiguous")
        if [profile.query_id for profile in self.profiles] != expected_ids:
            raise ValueError("external profile IDs do not match queries")

    def write_artifacts(self, output_dir: Path) -> None:
        """Write identity, coverage rows, and aggregate coverage as separate files."""

        self.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        query_lines = [
            json.dumps(query.to_dict(), sort_keys=True, allow_nan=False) + "\n"
            for query in self.queries
        ]
        (output_dir / "queries.jsonl").write_text(
            "".join(query_lines), encoding="utf-8"
        )
        write_verification_profiles_jsonl(output_dir / "profiles.jsonl", self.profiles)
        report = VerificationCoverageReport.from_profiles(self.profiles)
        (output_dir / "coverage.json").write_text(
            json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )


__all__ = [
    "ABCROWN_ADAPTER_SCHEMA_VERSION",
    "ABCrownBoundQueryProfiler",
    "file_sha256",
]
