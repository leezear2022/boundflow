"""S4-1B Pass A: ordered production coefficient-selector capture.

The owner is intentionally backend-neutral.  Production coefficient kernels
call :meth:`record` at the frozen 19 insertion points; only six insertion
points consume a coefficient tensor and materialize a compact selector.  A
failed attempt poisons the owner and cannot be retried.
"""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-instance-attributes,too-many-lines
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-many-boolean-expressions
# pylint: disable=too-many-arguments,duplicate-code
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=import-error,import-outside-toplevel

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import NoReturn

import torch

S4_SELECTOR_PASS_SCHEMA = "boundflow.asplos27-s4-selector-pass/v1"

S4_SELECTOR_ACTIONS = (
    "seed",
    "linear16_right",
    "relu31_coefficient",
    "linear14_right",
    "pack_a29",
    "relu28_coefficient",
    "residual11_stage1",
    "pack_a26",
    "residual11_stage2",
    "pack_a24",
    "relu23_coefficient",
    "residual6_stage1",
    "pack_a20",
    "residual6_stage2",
    "pack_a18",
    "relu17_coefficient",
    "conv0_right",
    "pack_ainput",
    "box_concretize",
)

S4_SELECTOR_SPECS = (
    ("endpoint_ainput_v2", "pack_ainput", 18432, "ternary"),
    ("sign_a18", "pack_a18", 12288, "binary"),
    ("sign_a20", "pack_a20", 6144, "binary"),
    ("sign_a24", "pack_a24", 6144, "binary"),
    ("sign_a26", "pack_a26", 6144, "binary"),
    ("sign_a29", "pack_a29", 6144, "binary"),
)


class S4SelectorPhase(str, Enum):
    PREPARED = "PREPARED"
    PASS_A_RUNNING = "PASS_A_RUNNING"
    SELECTORS_READY = "SELECTORS_READY"
    POISONED = "POISONED"
    CLOSED = "CLOSED"


class S4SelectorPassError(RuntimeError):
    """Stable fail-closed selector-pass error."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _reject(reason: str) -> NoReturn:
    raise S4SelectorPassError(reason)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class S4SelectorDescriptorV1:
    name: str
    action: str
    shape: tuple[int, ...]
    numel: int
    policy: str
    dtype: str
    device: str
    generation: int

    def validate(self) -> None:
        expected = {
            name: (action, numel, policy)
            for name, action, numel, policy in S4_SELECTOR_SPECS
        }
        if (
            self.name not in expected
            or expected[self.name] != (self.action, self.numel, self.policy)
            or self.shape != (self.numel,)
            or self.dtype != "torch.int8"
            or not self.device.startswith("cuda:")
            or self.generation <= 0
        ):
            _reject("SELECTOR_DESCRIPTOR_MISMATCH")


@dataclass(frozen=True)
class S4SelectorPassReceiptV1:
    action_order: tuple[str, ...]
    action_sequence_hash: str
    selector_descriptors: tuple[S4SelectorDescriptorV1, ...]
    evaluation_generation: int
    parameter_generation: int
    coefficient_generation: int
    selector_generation: int
    action_count: int
    selector_count: int
    endpoint_selector_count: int
    binary_selector_count: int
    invalid_selector_count: int
    launch_count: int
    compiled_pack_launch_count: int
    eager_pack_count: int
    fallback_count: int
    phase: str
    schema_version: str = S4_SELECTOR_PASS_SCHEMA
    timing_recorded: bool = False
    performance_claimed: bool = False

    def validate(self) -> None:
        for descriptor in self.selector_descriptors:
            descriptor.validate()
        if (
            self.schema_version != S4_SELECTOR_PASS_SCHEMA
            or self.action_order != S4_SELECTOR_ACTIONS
            or self.action_sequence_hash != _canonical_hash(list(S4_SELECTOR_ACTIONS))
            or min(
                self.evaluation_generation,
                self.parameter_generation,
                self.coefficient_generation,
                self.selector_generation,
            )
            <= 0
            or len(
                {
                    self.evaluation_generation,
                    self.parameter_generation,
                    self.coefficient_generation,
                    self.selector_generation,
                }
            )
            != 4
            or self.action_count != 19
            or self.selector_count != 6
            or self.endpoint_selector_count != 1
            or self.binary_selector_count != 5
            or self.invalid_selector_count < 0
            or self.launch_count != 19
            or self.compiled_pack_launch_count + self.eager_pack_count != 6
            or self.fallback_count != 0
            or self.phase != S4SelectorPhase.SELECTORS_READY.value
            or self.timing_recorded
            or self.performance_claimed
        ):
            _reject("SELECTOR_RECEIPT_MISMATCH")

    def stable_hash(self) -> str:
        self.validate()
        payload = asdict(self)
        payload["action_order"] = list(self.action_order)
        return _canonical_hash(payload)


class PreparedS4CoefficientSelectorPassV1:
    """Single-attempt ordered selector owner for one exact S4 evaluation."""

    def __init__(
        self,
        *,
        device: torch.device,
        exact_call_id: str,
        evaluation_generation: int = 1,
        parameter_generation: int = 2,
        coefficient_generation: int = 3,
        selector_generation: int = 4,
    ) -> None:
        if device.type != "cuda" or not exact_call_id:
            _reject("SELECTOR_OWNER_CONTEXT_MISMATCH")
        generations = (
            evaluation_generation,
            parameter_generation,
            coefficient_generation,
            selector_generation,
        )
        if min(generations) <= 0 or len(set(generations)) != 4:
            _reject("SELECTOR_GENERATION_MISMATCH")
        self.device = device
        self.exact_call_id = exact_call_id
        self.evaluation_generation = evaluation_generation
        self.parameter_generation = parameter_generation
        self.coefficient_generation = coefficient_generation
        self.selector_generation = selector_generation
        self.phase = S4SelectorPhase.PREPARED
        self._next_action = 0
        self._expected_stream: int | None = None
        self._captured_actions: list[str] = []
        self._selectors = {
            name: torch.empty(numel, dtype=torch.int8, device=device)
            for name, _action, numel, _policy in S4_SELECTOR_SPECS
        }
        self._action_to_spec = {
            action: (name, numel, policy)
            for name, action, numel, policy in S4_SELECTOR_SPECS
        }
        self._invalid_selector_count = 0
        self._compiled_pack: object | None = None
        self._compiled_views: dict[str, tuple[str, object, object, int]] = {}
        self._compiled_pack_launch_count = 0
        self._eager_pack_count = 0

    def bind_compiled_sources(self, sources: dict[str, torch.Tensor]) -> None:
        """Prepare six source/output DLPack pairs before Pass A begins."""

        import tvm

        from boundflow.backends.tvm.asplos27_s4_six_site_value import (
            S4_SELECTOR_PACK_SPECS,
            compile_s4_selector_pack_v1,
        )

        if self.phase != S4SelectorPhase.PREPARED or self._compiled_pack is not None:
            _reject("SELECTOR_COMPILED_BINDING_ORDER_MISMATCH")
        expected_actions = {
            action for _name, action, _numel, _policy in S4_SELECTOR_SPECS
        }
        if set(sources) != expected_actions:
            _reject("SELECTOR_COMPILED_SOURCE_SET_MISMATCH")
        ordinal = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        compiled = compile_s4_selector_pack_v1(device_index=ordinal)
        compiled.validate()
        backend_by_name = {
            name: (symbol, numel, policy)
            for name, symbol, numel, policy in S4_SELECTOR_PACK_SPECS
        }
        for name, action, numel, policy in S4_SELECTOR_SPECS:
            source = sources[action]
            backend = backend_by_name.get(name)
            if (
                backend is None
                or backend[1:] != (numel, policy)
                or source.device != self.device
                or source.dtype != torch.float32
                or source.numel() != numel
                or not source.is_contiguous()
            ):
                _reject("SELECTOR_COMPILED_SOURCE_LAYOUT_MISMATCH")
            symbol = backend[0]
            self._compiled_views[action] = (
                symbol,
                tvm.runtime.from_dlpack(source.reshape(-1)),
                tvm.runtime.from_dlpack(self._selectors[name]),
                source.data_ptr(),
            )
        self._compiled_pack = compiled

    @property
    def selectors(self) -> tuple[torch.Tensor, ...]:
        if self.phase != S4SelectorPhase.SELECTORS_READY:
            _reject("SELECTORS_NOT_READY")
        return tuple(self._selectors[name] for name, *_rest in S4_SELECTOR_SPECS)

    def selector(self, name: str) -> torch.Tensor:
        if self.phase != S4SelectorPhase.SELECTORS_READY:
            _reject("SELECTORS_NOT_READY")
        value = self._selectors.get(name)
        if value is None:
            _reject("SELECTOR_NAME_MISMATCH")
        return value

    def begin(self) -> None:
        if self.phase != S4SelectorPhase.PREPARED:
            _reject("SELECTOR_PASS_ALREADY_ATTEMPTED")
        current = torch.cuda.current_stream(self.device)
        self._expected_stream = int(current.cuda_stream)
        self.phase = S4SelectorPhase.PASS_A_RUNNING

    def _poison(self, reason: str) -> NoReturn:
        self.phase = S4SelectorPhase.POISONED
        _reject(reason)

    def _validate_context(self) -> None:
        if self.phase != S4SelectorPhase.PASS_A_RUNNING:
            self._poison("SELECTOR_PHASE_MISMATCH")
        current = int(torch.cuda.current_stream(self.device).cuda_stream)
        if self._expected_stream is None or current != self._expected_stream:
            self._poison("SELECTOR_STREAM_MISMATCH")

    def record(self, action: str, coefficient: torch.Tensor | None = None) -> None:
        """Record one frozen insertion point and optionally pack its selector."""

        self._validate_context()
        if self._next_action >= len(S4_SELECTOR_ACTIONS):
            self._poison("SELECTOR_ACTION_OVERFLOW")
        expected = S4_SELECTOR_ACTIONS[self._next_action]
        if action != expected:
            self._poison("SELECTOR_ACTION_ORDER_MISMATCH")
        selector_spec = self._action_to_spec.get(action)
        if selector_spec is None:
            if coefficient is not None:
                self._poison("SELECTOR_UNEXPECTED_COEFFICIENT")
        else:
            if coefficient is None:
                self._poison("SELECTOR_COEFFICIENT_ABSENT")
            name, numel, policy = selector_spec
            assert coefficient is not None
            if (
                coefficient.device != self.device
                or coefficient.dtype != torch.float32
                or not coefficient.is_contiguous()
                or coefficient.numel() != numel
            ):
                self._poison("SELECTOR_COEFFICIENT_LAYOUT_MISMATCH")
            flat = coefficient.reshape(-1)
            target = self._selectors[name]
            compiled_binding = self._compiled_views.get(action)
            if compiled_binding is not None:
                import tvm_ffi

                symbol, source_view, target_view, source_pointer = compiled_binding
                if (
                    coefficient.data_ptr() != source_pointer
                    or self._compiled_pack is None
                ):
                    self._poison("SELECTOR_COMPILED_SOURCE_IDENTITY_MISMATCH")
                current = torch.cuda.current_stream(self.device)
                with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
                    executable = getattr(self._compiled_pack, "executable")
                    executable[symbol](source_view, target_view)
                self._compiled_pack_launch_count += 1
            else:
                finite = torch.isfinite(flat)
                if policy == "ternary":
                    target.copy_(
                        torch.where(
                            finite,
                            torch.where(
                                flat > 0,
                                torch.ones_like(target),
                                torch.where(
                                    flat < 0,
                                    -torch.ones_like(target),
                                    torch.zeros_like(target),
                                ),
                            ),
                            torch.full_like(target, -128),
                        )
                    )
                else:
                    target.copy_(
                        torch.where(
                            finite,
                            (flat >= 0).to(torch.int8),
                            torch.full_like(target, -128),
                        )
                    )
                self._invalid_selector_count += int((~finite).sum().item())
                self._eager_pack_count += 1
        self._captured_actions.append(action)
        self._next_action += 1
        if self._next_action == len(S4_SELECTOR_ACTIONS):
            self.phase = S4SelectorPhase.SELECTORS_READY

    def receipt(self) -> S4SelectorPassReceiptV1:
        if self.phase != S4SelectorPhase.SELECTORS_READY:
            _reject("SELECTORS_NOT_READY")
        descriptors = tuple(
            S4SelectorDescriptorV1(
                name=name,
                action=action,
                shape=tuple(self._selectors[name].shape),
                numel=numel,
                policy=policy,
                dtype=str(self._selectors[name].dtype),
                device=str(self._selectors[name].device),
                generation=self.selector_generation,
            )
            for name, action, numel, policy in S4_SELECTOR_SPECS
        )
        receipt = S4SelectorPassReceiptV1(
            action_order=tuple(self._captured_actions),
            action_sequence_hash=_canonical_hash(list(self._captured_actions)),
            selector_descriptors=descriptors,
            evaluation_generation=self.evaluation_generation,
            parameter_generation=self.parameter_generation,
            coefficient_generation=self.coefficient_generation,
            selector_generation=self.selector_generation,
            action_count=len(self._captured_actions),
            selector_count=len(descriptors),
            endpoint_selector_count=1,
            binary_selector_count=5,
            invalid_selector_count=self._invalid_selector_count,
            launch_count=len(self._captured_actions),
            compiled_pack_launch_count=self._compiled_pack_launch_count,
            eager_pack_count=self._eager_pack_count,
            fallback_count=0,
            phase=self.phase.value,
        )
        receipt.validate()
        return receipt

    def close(self) -> None:
        self._selectors.clear()
        self._compiled_views.clear()
        self._compiled_pack = None
        self._captured_actions.clear()
        self.phase = S4SelectorPhase.CLOSED


def capture_r31b2_production_selectors_v1(
    executor: object,
    owner: PreparedS4CoefficientSelectorPassV1,
) -> S4SelectorPassReceiptV1:
    """Capture six selectors at the real staged R31B2 coefficient boundaries.

    This correctness-stage bridge deliberately calls the existing compiled B1
    and staged residual kernels.  Selector packing remains owned by ``owner``;
    timing and optimizer mutation stay closed.
    """

    from boundflow.backends.tvm.r3_d1c_wrapper_schedule import (
        R3D1C_RESIDUAL11_STAGE1,
        R3D1C_RESIDUAL11_STAGE2,
        R3D1C_RESIDUAL6_STAGE1,
        R3D1C_RESIDUAL6_STAGE2,
    )
    from boundflow.backends.tvm.r3_full_lower_forward import (
        R31B1_CONV0_SYMBOL,
        R31B1_LINEAR14_SYMBOL,
        R31B1_LINEAR16_SYMBOL,
        R31B1_SEED_SYMBOL,
    )
    from boundflow.backends.tvm.r3_p_alpha_vjp import R31B2_CLEAR_SYMBOL

    required = (
        "forward_executor",
        "_tensor",
        "_launch_b1",
        "_launch_b2",
        "_launch_d2b",
        "_relu_coefficient",
        "_residual11_scratch",
        "_residual6_scratch",
    )
    if any(not hasattr(executor, name) for name in required):
        _reject("SELECTOR_PRODUCTION_EXECUTOR_MISMATCH")
    forward = getattr(executor, "forward_executor")
    s0 = forward.scratch_0
    s1 = forward.scratch_1
    bias = forward.bias_accumulator
    tensor = getattr(executor, "_tensor")
    launch_b1 = getattr(executor, "_launch_b1")
    launch_b2 = getattr(executor, "_launch_b2")
    launch_d2b = getattr(executor, "_launch_d2b")
    relu_coefficient = getattr(executor, "_relu_coefficient")
    residual11_scratch = getattr(executor, "_residual11_scratch")
    residual6_scratch = getattr(executor, "_residual6_scratch")

    owner.bind_compiled_sources(
        {
            "pack_a29": s0[:6144],
            "pack_a26": residual11_scratch,
            "pack_a24": s1[:6144],
            "pack_a20": residual6_scratch,
            "pack_a18": s0[:12288],
            "pack_ainput": s1[:18432],
        }
    )
    launch_b2(R31B2_CLEAR_SYMBOL, s0, s1)
    owner.begin()
    launch_b1(R31B1_SEED_SYMBOL, tensor("objective"), s0[:60], bias)
    owner.record("seed")
    launch_b1(
        R31B1_LINEAR16_SYMBOL,
        s0[:60],
        tensor("param/linear2.weight"),
        tensor("param/linear2.bias"),
        bias,
        s1[:600],
        bias,
    )
    owner.record("linear16_right")
    relu_coefficient("31", s1[:600])
    owner.record("relu31_coefficient")
    launch_b1(
        R31B1_LINEAR14_SYMBOL,
        s1[:600],
        tensor("param/linear1.weight"),
        tensor("param/linear1.bias"),
        bias,
        s0[:6144],
        bias,
    )
    owner.record("linear14_right")
    owner.record("pack_a29", s0[:6144])
    relu_coefficient("28", s0[:6144])
    owner.record("relu28_coefficient")

    launch_d2b(
        R3D1C_RESIDUAL11_STAGE1,
        s0,
        tensor("param/layer1.1.conv2.weight"),
        residual11_scratch,
    )
    owner.record("residual11_stage1")
    owner.record("pack_a26", residual11_scratch)
    launch_d2b(
        R3D1C_RESIDUAL11_STAGE2,
        s0,
        residual11_scratch,
        tensor("relu/25/lower").reshape(6, 1024),
        tensor("relu/25/upper").reshape(6, 1024),
        tensor("relu/25/alpha"),
        forward.alpha_maps["25"],
        tensor("param/layer1.1.conv1.weight"),
        tensor("param/layer1.1.conv2.bias"),
        tensor("param/layer1.1.conv1.bias"),
        bias,
        s1[:6144],
        bias,
    )
    owner.record("residual11_stage2")
    owner.record("pack_a24", s1[:6144])
    relu_coefficient("23", s1[:6144])
    owner.record("relu23_coefficient")

    launch_d2b(
        R3D1C_RESIDUAL6_STAGE1,
        s1,
        tensor("param/layer1.0.conv2.weight"),
        residual6_scratch,
    )
    owner.record("residual6_stage1")
    owner.record("pack_a20", residual6_scratch)
    launch_d2b(
        R3D1C_RESIDUAL6_STAGE2,
        s1,
        residual6_scratch,
        tensor("relu/19/lower").reshape(6, 1024),
        tensor("relu/19/upper").reshape(6, 1024),
        tensor("relu/19/alpha"),
        forward.alpha_maps["19"],
        tensor("param/layer1.0.conv1.weight"),
        tensor("param/layer1.0.shortcut.0.weight"),
        tensor("param/layer1.0.conv2.bias"),
        tensor("param/layer1.0.conv1.bias"),
        tensor("param/layer1.0.shortcut.0.bias"),
        bias,
        s0[:12288],
        bias,
    )
    owner.record("residual6_stage2")
    owner.record("pack_a18", s0[:12288])
    relu_coefficient("17", s0[:12288])
    owner.record("relu17_coefficient")
    launch_b1(
        R31B1_CONV0_SYMBOL,
        s0[:12288],
        tensor("param/conv1.weight"),
        tensor("param/conv1.bias"),
        bias,
        s1,
        bias,
    )
    owner.record("conv0_right")
    owner.record("pack_ainput", s1[:18432])
    owner.record("box_concretize")
    return owner.receipt()


__all__ = [
    "PreparedS4CoefficientSelectorPassV1",
    "S4_SELECTOR_ACTIONS",
    "S4_SELECTOR_PASS_SCHEMA",
    "S4_SELECTOR_SPECS",
    "S4SelectorPassError",
    "S4SelectorPassReceiptV1",
    "S4SelectorPhase",
    "capture_r31b2_production_selectors_v1",
]
