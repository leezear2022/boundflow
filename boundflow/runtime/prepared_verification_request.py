"""AOT preparation for model/property state at a verifier API boundary."""

# pylint: disable=protected-access,too-many-arguments,too-many-instance-attributes
# pylint: disable=too-many-locals

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
import time
from types import MethodType
from typing import Any, Callable, ContextManager, Iterator, Mapping


@dataclass(frozen=True)
class PreparedVerificationRequestReceiptV1:
    """Static preparation and warm-reuse accounting."""

    prepare_wall_ns: int
    prepare_environment_ns: int
    prepare_model_ns: int
    prepare_runtime_spec_ns: int
    prepare_handler_ns: int
    model_reuse_count: int
    runtime_spec_clone_count: int
    handler_clone_count: int
    model_reuse_ns: int
    runtime_spec_clone_ns: int
    handler_clone_ns: int
    handler_clone_policy: str
    performance_claimed: bool = False

    def to_dict(self) -> dict[str, int | bool | str]:
        """Return one deterministic receipt projection."""

        return {
            "schema_version": "boundflow.prepared-verification-request/v1",
            "prepare_wall_ns": self.prepare_wall_ns,
            "prepare_environment_ns": self.prepare_environment_ns,
            "prepare_model_ns": self.prepare_model_ns,
            "prepare_runtime_spec_ns": self.prepare_runtime_spec_ns,
            "prepare_handler_ns": self.prepare_handler_ns,
            "model_reuse_count": self.model_reuse_count,
            "runtime_spec_clone_count": self.runtime_spec_clone_count,
            "handler_clone_count": self.handler_clone_count,
            "model_reuse_ns": self.model_reuse_ns,
            "runtime_spec_clone_ns": self.runtime_spec_clone_ns,
            "handler_clone_ns": self.handler_clone_ns,
            "handler_clone_policy": self.handler_clone_policy,
            "static_prepare_excluded_from_query": True,
            "performance_claimed": self.performance_claimed,
        }


def _timed_call(function: Callable[[], Any]) -> tuple[Any, int]:
    started_ns = time.perf_counter_ns()
    value = function()
    elapsed_ns = time.perf_counter_ns() - started_ns
    if elapsed_ns <= 0:
        raise ValueError("prepared verification phase timing must be positive")
    return value, elapsed_ns


def _clone_handler(template: Any, torch_module: Any, *, copy_on_prune: bool) -> Any:
    """Clone mutable vnnlib handler state without reparsing the property."""

    clone = copy.copy(template)
    if copy_on_prune:
        # ``rhs_offset`` and sanity-check mutation are rejected at admission,
        # so the parsed vnnlib payload and initial BatchedSpecs are immutable.
        clone.vnnlib = template.vnnlib
        clone.current_index = 0
        # The admitted verifier only reads all_specs and replaces tensor fields
        # on prune.  Sharing this immutable initial projection avoids a second
        # GPU materialization; each later prune remains local to ``clone``.
        clone.all_specs = template.all_specs
        return clone
    for name, value in vars(template).items():
        if name == "all_specs":
            continue
        if torch_module.is_tensor(value):
            copied = value.detach().clone()
        elif isinstance(value, (dict, list, set, tuple)):
            copied = copy.deepcopy(value)
        else:
            copied = value
        setattr(clone, name, copied)
    clone.current_index = 0
    clone._set_all_specs()  # pylint: disable=protected-access
    return clone


class PreparedVerificationRequestV1:
    """Own static model/spec/handler state and provide fresh query-local views."""

    def __init__(
        self,
        *,
        solver: Any,
        constraint: Any,
        model: Any,
        runtime_spec: Any,
        handler: Any,
        torch_module: Any,
        prepare_timings: Mapping[str, int],
        copy_on_prune_handler: bool,
    ) -> None:
        self.solver = solver
        self.constraint = constraint
        self.model = model
        self.runtime_spec = runtime_spec
        self.handler = handler
        self.torch_module = torch_module
        self.prepare_timings = dict(prepare_timings)
        self.copy_on_prune_handler = copy_on_prune_handler
        self._counts = {
            "model_reuse_count": 0,
            "runtime_spec_clone_count": 0,
            "handler_clone_count": 0,
        }
        self._reuse_ns = {
            "model_reuse_ns": 0,
            "runtime_spec_clone_ns": 0,
            "handler_clone_ns": 0,
        }

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Replace only three static preparation methods for one query."""

        names = ("_prepare_model", "_prepare_runtime_spec", "_build_vnnlib_handler")
        preexisting = {name: name in vars(self.solver) for name in names}
        originals = {name: getattr(self.solver, name) for name in names}

        def prepared_model(_solver: Any, _device: str) -> Any:
            started_ns = time.perf_counter_ns()
            self._counts["model_reuse_count"] += 1
            try:
                return self.model
            finally:
                self._reuse_ns["model_reuse_ns"] += time.perf_counter_ns() - started_ns

        def prepared_runtime_spec(_solver: Any) -> Any:
            started_ns = time.perf_counter_ns()
            self._counts["runtime_spec_clone_count"] += 1
            try:
                runtime_spec = copy.deepcopy(self.runtime_spec)
                _solver._runtime_spec = runtime_spec  # pylint: disable=protected-access
                return runtime_spec
            finally:
                self._reuse_ns["runtime_spec_clone_ns"] += (
                    time.perf_counter_ns() - started_ns
                )

        def prepared_handler(_solver: Any, _constraints: Any) -> Any:
            started_ns = time.perf_counter_ns()
            self._counts["handler_clone_count"] += 1
            try:
                return _clone_handler(
                    self.handler,
                    self.torch_module,
                    copy_on_prune=self.copy_on_prune_handler,
                )
            finally:
                self._reuse_ns["handler_clone_ns"] += (
                    time.perf_counter_ns() - started_ns
                )

        setattr(self.solver, "_prepare_model", MethodType(prepared_model, self.solver))
        setattr(
            self.solver,
            "_prepare_runtime_spec",
            MethodType(prepared_runtime_spec, self.solver),
        )
        setattr(
            self.solver,
            "_build_vnnlib_handler",
            MethodType(prepared_handler, self.solver),
        )
        try:
            yield
        finally:
            for name in reversed(names):
                if preexisting[name]:
                    setattr(self.solver, name, originals[name])
                else:
                    delattr(self.solver, name)

    def receipt(self) -> PreparedVerificationRequestReceiptV1:
        """Fail closed unless one query consumed each prepared component once."""

        if set(self.prepare_timings) != {
            "prepare_wall_ns",
            "prepare_environment_ns",
            "prepare_model_ns",
            "prepare_runtime_spec_ns",
            "prepare_handler_ns",
        } or any(value <= 0 for value in self.prepare_timings.values()):
            raise ValueError("prepared verification timing receipt differs")
        if any(value != 1 for value in self._counts.values()):
            raise ValueError("prepared verification reuse count differs")
        if any(value <= 0 for value in self._reuse_ns.values()):
            raise ValueError("prepared verification reuse timing differs")
        return PreparedVerificationRequestReceiptV1(
            prepare_wall_ns=self.prepare_timings["prepare_wall_ns"],
            prepare_environment_ns=self.prepare_timings["prepare_environment_ns"],
            prepare_model_ns=self.prepare_timings["prepare_model_ns"],
            prepare_runtime_spec_ns=self.prepare_timings["prepare_runtime_spec_ns"],
            prepare_handler_ns=self.prepare_timings["prepare_handler_ns"],
            model_reuse_count=self._counts["model_reuse_count"],
            runtime_spec_clone_count=self._counts["runtime_spec_clone_count"],
            handler_clone_count=self._counts["handler_clone_count"],
            model_reuse_ns=self._reuse_ns["model_reuse_ns"],
            runtime_spec_clone_ns=self._reuse_ns["runtime_spec_clone_ns"],
            handler_clone_ns=self._reuse_ns["handler_clone_ns"],
            handler_clone_policy=(
                "share-immutable-initial-copy-on-prune"
                if self.copy_on_prune_handler
                else "eager-deep-tensor-clone"
            ),
        )


def prepare_verification_request_v1(
    *,
    solver: Any,
    constraint: Any,
    device: str,
    torch_module: Any,
    config_context: Callable[[Mapping[str, Any]], ContextManager[Any]],
    copy_on_prune_handler: bool = False,
) -> PreparedVerificationRequestV1:
    """Prepare static verifier inputs once outside a warm query window."""

    if copy_on_prune_handler:
        specification = solver.config.get("specification", {})
        debug = solver.config.get("debug", {})
        if (
            not isinstance(specification, Mapping)
            or specification.get("rhs_offset") is not None
            or not isinstance(debug, Mapping)
            or debug.get("sanity_check") not in {None, False, ""}
        ):
            raise ValueError("copy-on-prune handler policy is not admissible")
    started_ns = time.perf_counter_ns()
    solver.constraint = constraint
    solver.spec = constraint
    with config_context(solver.config):
        _, environment_ns = _timed_call(lambda: solver._prepare_environment(device))
        model, model_ns = _timed_call(lambda: solver._prepare_model(device))
        runtime_spec, runtime_spec_ns = _timed_call(solver._prepare_runtime_spec)
        handler, handler_ns = _timed_call(
            lambda: solver._build_vnnlib_handler(runtime_spec)
        )
    prepare_wall_ns = time.perf_counter_ns() - started_ns
    return PreparedVerificationRequestV1(
        solver=solver,
        constraint=constraint,
        model=model,
        runtime_spec=runtime_spec,
        handler=handler,
        torch_module=torch_module,
        prepare_timings={
            "prepare_wall_ns": prepare_wall_ns,
            "prepare_environment_ns": environment_ns,
            "prepare_model_ns": model_ns,
            "prepare_runtime_spec_ns": runtime_spec_ns,
            "prepare_handler_ns": handler_ns,
        },
        copy_on_prune_handler=copy_on_prune_handler,
    )
