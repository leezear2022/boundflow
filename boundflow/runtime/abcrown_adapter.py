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

from ..domains.interval import IntervalState
from ..planner.materialization import BoundMethod, OptimizationStage
from .bab_query import BoundQuery, QueryCompatibilityKey
from .verification_profile import (
    VerificationCoverageReport,
    VerificationQueryProfile,
    write_verification_profiles_jsonl,
)
from .verifier_ir_integration import (
    ExternalVerifierCallSpec,
    compile_external_verifier_call,
    execute_external_verifier_call,
)

ABCROWN_ADAPTER_SCHEMA_VERSION = "boundflow.abcrown-adapter/v2"
INTERMEDIATE_BOUNDS_PAYLOAD_SCHEMA_VERSION = "boundflow.external-intermediate-bounds/v1"


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


@dataclass(frozen=True)
class CapturedIntermediateBound:
    """Owned pre-activation interval from one external ReLU in graph order."""

    ordinal: int
    external_relu_name: str
    external_preactivation_name: str
    lower: torch.Tensor
    upper: torch.Tensor

    def validate(self) -> None:
        """Reject missing identity, malformed order, or invalid intervals."""

        if self.ordinal < 0:
            raise ValueError("external intermediate-bound ordinal must be non-negative")
        if not self.external_relu_name or not self.external_preactivation_name:
            raise ValueError("external intermediate-bound names must be non-empty")
        if self.lower.shape != self.upper.shape:
            raise ValueError("external intermediate lower/upper shapes differ")
        if (
            self.lower.dtype != self.upper.dtype
            or self.lower.device != self.upper.device
        ):
            raise ValueError("external intermediate lower/upper tensor types differ")
        if not torch.is_floating_point(self.lower):
            raise TypeError("external intermediate bounds must be floating tensors")
        if not bool(
            torch.isfinite(self.lower).all() and torch.isfinite(self.upper).all()
        ):
            raise ValueError("external intermediate bounds must be finite")
        if not bool((self.lower <= self.upper).all()):
            raise ValueError("external intermediate lower exceeds upper")


def _external_node_name(node: Any, *, fallback: str) -> str:
    name = getattr(node, "name", None)
    return str(name) if name else fallback


def _capture_intermediate_bounds(
    bounded_module: Any,
) -> tuple[CapturedIntermediateBound, ...]:
    """Own every available external ReLU pre-activation interval in graph order."""

    nodes_method = getattr(bounded_module, "nodes", None)
    if not callable(nodes_method):
        return ()
    captured: list[CapturedIntermediateBound] = []
    for node_index, node in enumerate(nodes_method()):
        if _normalize_external_op(node) != "relu":
            continue
        inputs = getattr(node, "inputs", ())
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 1:
            continue
        preactivation = inputs[0]
        lower = getattr(preactivation, "lower", None)
        upper = getattr(preactivation, "upper", None)
        if not torch.is_tensor(lower) or not torch.is_tensor(upper):
            continue
        item = CapturedIntermediateBound(
            ordinal=len(captured),
            external_relu_name=_external_node_name(
                node, fallback=f"external-relu-{node_index}"
            ),
            external_preactivation_name=_external_node_name(
                preactivation, fallback=f"external-preactivation-{node_index}"
            ),
            lower=lower.detach().clone(),
            upper=upper.detach().clone(),
        )
        item.validate()
        captured.append(item)
    return tuple(captured)


def intermediate_bounds_sha256(
    bounds: Sequence[CapturedIntermediateBound],
) -> str:
    """Hash ordered external intermediate-bound identity and tensor content."""

    digest = hashlib.sha256()
    for expected, item in enumerate(bounds):
        item.validate()
        if item.ordinal != expected:
            raise ValueError("external intermediate-bound ordinals are not contiguous")
        digest.update(str(item.ordinal).encode("utf-8"))
        digest.update(item.external_relu_name.encode("utf-8"))
        digest.update(item.external_preactivation_name.encode("utf-8"))
        digest.update(_tensor_sha256(item.lower).encode("utf-8"))
        digest.update(_tensor_sha256(item.upper).encode("utf-8"))
    return digest.hexdigest()


def serialize_intermediate_bounds(
    bounds: Sequence[CapturedIntermediateBound],
) -> dict[str, Any]:
    """Create a portable, ``weights_only=True`` compatible tensor payload."""

    records: list[dict[str, Any]] = []
    for expected, item in enumerate(bounds):
        item.validate()
        if item.ordinal != expected:
            raise ValueError("external intermediate-bound ordinals are not contiguous")
        lower = item.lower.detach().cpu().contiguous().clone()
        upper = item.upper.detach().cpu().contiguous().clone()
        records.append(
            {
                "ordinal": item.ordinal,
                "external_relu_name": item.external_relu_name,
                "external_preactivation_name": item.external_preactivation_name,
                "shape": list(lower.shape),
                "dtype": str(lower.dtype),
                "lower": lower,
                "upper": upper,
                "lower_sha256": _tensor_sha256(lower),
                "upper_sha256": _tensor_sha256(upper),
            }
        )
    return {
        "schema_version": INTERMEDIATE_BOUNDS_PAYLOAD_SCHEMA_VERSION,
        "count": len(records),
        "sha256": intermediate_bounds_sha256(bounds),
        "records": records,
    }


# pylint: disable-next=too-many-locals,too-many-branches
def deserialize_intermediate_bounds(
    payload: Mapping[str, Any],
) -> tuple[CapturedIntermediateBound, ...]:
    """Load and fully validate a portable external intermediate-bound payload."""

    if payload.get("schema_version") != INTERMEDIATE_BOUNDS_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("unsupported external intermediate-bound payload schema")
    raw_count = payload.get("count")
    raw_records = payload.get("records")
    if not isinstance(raw_count, int) or isinstance(raw_count, bool):
        raise TypeError("external intermediate-bound payload count must be an integer")
    if not isinstance(raw_records, (tuple, list)):
        raise TypeError(
            "external intermediate-bound payload records must be a sequence"
        )
    if raw_count != len(raw_records):
        raise ValueError("external intermediate-bound payload count mismatch")
    bounds: list[CapturedIntermediateBound] = []
    required_keys = {
        "ordinal",
        "external_relu_name",
        "external_preactivation_name",
        "shape",
        "dtype",
        "lower",
        "upper",
        "lower_sha256",
        "upper_sha256",
    }
    for expected, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise TypeError("external intermediate-bound record must be a mapping")
        if set(raw) != required_keys:
            raise ValueError("external intermediate-bound record fields differ")
        ordinal = raw["ordinal"]
        relu_name = raw["external_relu_name"]
        preactivation_name = raw["external_preactivation_name"]
        shape = raw["shape"]
        dtype = raw["dtype"]
        lower = raw["lower"]
        upper = raw["upper"]
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise TypeError("external intermediate-bound ordinal must be an integer")
        if ordinal != expected:
            raise ValueError("external intermediate-bound ordinals are not contiguous")
        if not isinstance(relu_name, str) or not isinstance(preactivation_name, str):
            raise TypeError("external intermediate-bound names must be strings")
        if not isinstance(shape, (tuple, list)) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in shape
        ):
            raise TypeError("external intermediate-bound shape must be integer-valued")
        if not isinstance(dtype, str):
            raise TypeError("external intermediate-bound dtype must be a string")
        if not torch.is_tensor(lower) or not torch.is_tensor(upper):
            raise TypeError("external intermediate bounds must be tensors")
        if list(lower.shape) != list(shape) or list(upper.shape) != list(shape):
            raise ValueError("external intermediate-bound recorded shape differs")
        if str(lower.dtype) != dtype or str(upper.dtype) != dtype:
            raise ValueError("external intermediate-bound recorded dtype differs")
        if raw["lower_sha256"] != _tensor_sha256(lower):
            raise ValueError("external intermediate-bound lower digest differs")
        if raw["upper_sha256"] != _tensor_sha256(upper):
            raise ValueError("external intermediate-bound upper digest differs")
        item = CapturedIntermediateBound(
            ordinal=ordinal,
            external_relu_name=relu_name,
            external_preactivation_name=preactivation_name,
            lower=lower.detach().cpu().contiguous().clone(),
            upper=upper.detach().cpu().contiguous().clone(),
        )
        item.validate()
        bounds.append(item)
    frozen = tuple(bounds)
    if payload.get("sha256") != intermediate_bounds_sha256(frozen):
        raise ValueError("external intermediate-bound aggregate digest differs")
    return frozen


def bind_intermediate_bounds(
    external_items: Sequence[CapturedIntermediateBound],
    local_relu_pre: Mapping[str, IntervalState],
) -> dict[str, IntervalState]:
    """Map external ReLU intervals onto a local graph by exact order and shape."""

    local_items = tuple(local_relu_pre.items())
    if len(local_items) != len(external_items):
        raise ValueError(
            "external/local ReLU intermediate count mismatch: "
            f"external={len(external_items)} local={len(local_items)}"
        )
    bound: dict[str, IntervalState] = {}
    for expected, ((local_name, local), external) in enumerate(
        zip(local_items, external_items)
    ):
        external.validate()
        if external.ordinal != expected:
            raise ValueError("external intermediate-bound ordinals are not contiguous")
        if external.lower.shape != local.lower.shape:
            raise ValueError(
                "external/local ReLU intermediate shape mismatch at "
                f"ordinal {expected}: external={tuple(external.lower.shape)} "
                f"local={tuple(local.lower.shape)}"
            )
        lower = external.lower.to(
            device=local.lower.device, dtype=local.lower.dtype
        ).contiguous()
        upper = external.upper.to(
            device=local.upper.device, dtype=local.upper.dtype
        ).contiguous()
        bound[local_name] = IntervalState(lower=lower, upper=upper)
    return bound


def bind_captured_intermediate_bounds(
    captured: "CapturedABCrownQuery",
    local_relu_pre: Mapping[str, IntervalState],
) -> dict[str, IntervalState]:
    """Map process-local captured intervals onto a local graph."""

    return bind_intermediate_bounds(captured.intermediate_bounds, local_relu_pre)


@dataclass(frozen=True)
class CapturedABCrownQuery:  # pylint: disable=too-many-instance-attributes
    """Owned tensor payload plus a process-local exact-call replay closure."""

    input_lower: torch.Tensor
    input_upper: torch.Tensor
    linear_spec_c: torch.Tensor
    external_lower: torch.Tensor
    external_upper: torch.Tensor | None
    method: str
    solver_phase: str
    bound_lower_requested: bool
    bound_upper_requested: bool
    intermediate_bounds: tuple[CapturedIntermediateBound, ...]
    intermediate_bounds_hash: str
    relu_lower_slope_policy: str
    replay_external: Callable[[], Any] = field(repr=False, compare=False)


@dataclass
class ABCrownInitialCrownCapture:
    """Capture the first real, split-free plain-CROWN ``compute_bounds`` call."""

    phase_resolver: Callable[[], str] = _phase_from_stack
    captured: CapturedABCrownQuery | None = None

    def _candidate_payload(
        self,
        bounded_module: Any,
        original: Callable[..., Any],
        call_args: tuple[Any, ...],
        call_kwargs: Mapping[str, Any],
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            str,
            str,
            bool,
            bool,
        ]
        | None
    ):
        if self.captured is not None:
            return None
        values = _argument_map(original, call_args, call_kwargs)
        values.pop("self", None)
        method = str(values.get("method", call_kwargs.get("method", "backward")))
        if (
            _method_kind(method, beta_enabled=_beta_enabled(bounded_module))
            != BoundMethod.CROWN
        ):
            return None
        solver_phase = self.phase_resolver()
        if solver_phase not in {
            "alpha_crown_initialization",
            "incomplete_verification",
        }:
            return None
        input_tensor = _first_tensor(values.get("x"))
        linear_spec = _first_tensor(values.get("C"))
        if input_tensor is None or linear_spec is None:
            return None
        lower, upper = _input_bounds(input_tensor)
        return (
            lower.detach().clone(),
            upper.detach().clone(),
            linear_spec.detach().clone(),
            method,
            solver_phase,
            bool(values.get("bound_lower", True)),
            bool(values.get("bound_upper", True)),
        )

    def _finish_capture(
        self,
        bounded_module: Any,
        candidate: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            str,
            str,
            bool,
            bool,
        ],
        result: Any,
        replay_external: Callable[[], Any],
    ) -> None:
        if self.captured is not None:
            return
        if not isinstance(result, (tuple, list)) or not result:
            raise TypeError("plain-CROWN compute_bounds must return a tuple/list")
        external_lower = _first_tensor(result[0])
        external_upper = _first_tensor(result[1]) if len(result) > 1 else None
        if external_lower is None:
            raise ValueError("plain-CROWN result does not contain a lower bound")
        (
            lower,
            upper,
            linear_spec,
            method,
            solver_phase,
            bound_lower_requested,
            bound_upper_requested,
        ) = candidate
        intermediate_bounds = _capture_intermediate_bounds(bounded_module)
        self.captured = CapturedABCrownQuery(
            input_lower=lower,
            input_upper=upper,
            linear_spec_c=linear_spec,
            external_lower=external_lower.detach().clone(),
            external_upper=(
                None if external_upper is None else external_upper.detach().clone()
            ),
            method=method,
            solver_phase=solver_phase,
            bound_lower_requested=bound_lower_requested,
            bound_upper_requested=bound_upper_requested,
            intermediate_bounds=intermediate_bounds,
            intermediate_bounds_hash=intermediate_bounds_sha256(intermediate_bounds),
            relu_lower_slope_policy="adaptive",
            replay_external=replay_external,
        )

    @contextmanager
    def instrument(self, target: Any) -> Iterator["ABCrownInitialCrownCapture"]:
        """Wrap a BoundedModule instance/class and restore it exactly on exit."""

        if not hasattr(target, "compute_bounds"):
            raise TypeError("instrument target must expose compute_bounds")
        original = getattr(target, "compute_bounds")
        had_instance_override = not inspect.isclass(target) and (
            "compute_bounds" in getattr(target, "__dict__", {})
        )
        if inspect.isclass(target):

            def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
                candidate = self._candidate_payload(
                    instance, original, (instance, *args), kwargs
                )
                result = original(instance, *args, **kwargs)
                if candidate is not None:
                    replay_args = tuple(args)
                    replay_kwargs = dict(kwargs)
                    self._finish_capture(
                        instance,
                        candidate,
                        result,
                        lambda: original(instance, *replay_args, **replay_kwargs),
                    )
                return result

            setattr(target, "compute_bounds", wrapped)
        else:

            def wrapped_instance(_instance: Any, *args: Any, **kwargs: Any) -> Any:
                candidate = self._candidate_payload(target, original, args, kwargs)
                result = original(*args, **kwargs)
                if candidate is not None:
                    replay_args = tuple(args)
                    replay_kwargs = dict(kwargs)
                    self._finish_capture(
                        target,
                        candidate,
                        result,
                        lambda: original(*replay_args, **replay_kwargs),
                    )
                return result

            setattr(target, "compute_bounds", MethodType(wrapped_instance, target))
        try:
            yield self
        finally:
            if inspect.isclass(target) or had_instance_override:
                setattr(target, "compute_bounds", original)
            else:
                delattr(target, "compute_bounds")


@dataclass
class ABCrownBoundQueryProfiler:  # pylint: disable=too-many-instance-attributes
    """Build PR-13 ``BoundQuery`` records at real ``compute_bounds`` calls."""

    model_structure_hash: str
    weight_version: str
    query_prefix: str = "abcrown"
    phase_resolver: Callable[[], str] = _phase_from_stack
    precondition_rejections: Tuple[str, ...] = ()
    typed_ir_enabled: bool = False
    queries: list[BoundQuery] = field(default_factory=list)
    profiles: list[VerificationQueryProfile] = field(default_factory=list)
    typed_ir_records: list[dict[str, object]] = field(default_factory=list)
    _active_query_ids: list[str] = field(default_factory=list, init=False, repr=False)

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
    ) -> BoundQuery:
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
            "bound_lower_requested": bool(values.get("bound_lower", True)),
            "bound_upper_requested": bool(values.get("bound_upper", True)),
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
        requested_outputs = tuple(
            name
            for name, enabled in (
                ("lower", source_options["bound_lower_requested"]),
                ("upper", source_options["bound_upper_requested"]),
            )
            if enabled
        )
        if not requested_outputs:
            raise ValueError("compute_bounds call requests neither lower nor upper")
        query = BoundQuery(
            query_id=f"{self.query_prefix}-{sequence_number:08d}",
            parent_query_id=(
                self._active_query_ids[-1] if self._active_query_ids else None
            ),
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
            requested_outputs=requested_outputs,
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
        return query

    def _execute(self, query: BoundQuery, exact_call: Callable[[], Any]) -> Any:
        """Optionally route one exact provider call through typed IR."""

        if not self.typed_ir_enabled:
            self._active_query_ids.append(query.query_id)
            try:
                return exact_call()
            finally:
                completed_query_id = self._active_query_ids.pop()
                if completed_query_id != query.query_id:
                    raise RuntimeError("external query lineage stack is corrupted")
        compilation = compile_external_verifier_call(
            ExternalVerifierCallSpec.from_query_dict(query.to_dict())
        )
        record_index = len(self.typed_ir_records)
        record = {
            "query_id": query.query_id,
            "sequence_number": query.sequence_number,
            "completed": False,
            "error_type": "in_flight",
            "result_hash": None,
            "ir_hashes": compilation.hashes(),
            "semantics_owner": "external_verifier",
            "performance_claimed": False,
        }
        # Reserve the slot before entering the provider. auto_LiRPA can make a
        # nested compute_bounds call, so completion order is not call order.
        self.typed_ir_records.append(record)
        self._active_query_ids.append(query.query_id)
        try:
            execution = execute_external_verifier_call(compilation, exact_call)
        except BaseException as error:
            record["error_type"] = type(error).__name__
            self.typed_ir_records[record_index] = record
            raise
        finally:
            completed_query_id = self._active_query_ids.pop()
            if completed_query_id != query.query_id:
                raise RuntimeError("external query lineage stack is corrupted")
        record.update(
            completed=True,
            error_type=None,
            result_hash=execution.result_hash,
            ir_hashes=dict(execution.ir_hashes),
        )
        self.typed_ir_records[record_index] = record
        return execution.result

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
                query = self._record(instance, original, (instance, *args), kwargs)
                return self._execute(query, lambda: original(instance, *args, **kwargs))

            setattr(target, "compute_bounds", wrapped)
        else:

            def wrapped_instance(_instance: Any, *args: Any, **kwargs: Any) -> Any:
                query = self._record(target, original, args, kwargs)
                return self._execute(query, lambda: original(*args, **kwargs))

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
        if self.typed_ir_enabled:
            record_ids = [str(record["query_id"]) for record in self.typed_ir_records]
            if record_ids != expected_ids:
                first_mismatch = next(
                    (
                        (expected, actual)
                        for expected, actual in zip(expected_ids, record_ids)
                        if expected != actual
                    ),
                    None,
                )
                raise ValueError(
                    "typed IR execution records do not match queries: "
                    f"queries={len(expected_ids)} records={len(record_ids)} "
                    f"first_mismatch={first_mismatch}"
                )

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
        if self.typed_ir_enabled:
            typed_lines = [
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                for record in self.typed_ir_records
            ]
            (output_dir / "typed_ir.jsonl").write_text(
                "".join(typed_lines), encoding="utf-8"
            )


__all__ = [
    "ABCROWN_ADAPTER_SCHEMA_VERSION",
    "INTERMEDIATE_BOUNDS_PAYLOAD_SCHEMA_VERSION",
    "CapturedIntermediateBound",
    "bind_intermediate_bounds",
    "bind_captured_intermediate_bounds",
    "deserialize_intermediate_bounds",
    "intermediate_bounds_sha256",
    "serialize_intermediate_bounds",
    "ABCrownBoundQueryProfiler",
    "ABCrownInitialCrownCapture",
    "CapturedABCrownQuery",
    "file_sha256",
]
