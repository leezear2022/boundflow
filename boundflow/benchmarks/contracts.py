"""Frozen PR-12 benchmark contracts shared by runners and artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

BENCHMARK_CONTRACT_SCHEMA_VERSION = "boundflow.pr12-benchmark-contract/v1"


class BenchmarkContractLevel(Enum):
    """The three non-interchangeable PR-12 measurement levels."""

    KERNEL = "kernel"
    REGION_RUNTIME = "region_runtime"
    END_TO_END_FINAL_BOUND = "end_to_end_final_bound"


@dataclass(frozen=True)
class BenchmarkContract:  # pylint: disable=too-many-instance-attributes
    """Auditable inclusion and measurement rules for one benchmark level."""

    contract_id: str
    level: BenchmarkContractLevel
    inputs_resident_on_cuda: bool
    requires_preallocated_outputs: bool
    includes_required_output_allocation: bool
    includes_backend_dispatch: bool
    includes_interop_and_stream_management: bool
    includes_region_matching: bool
    includes_planner: bool
    includes_concretization: bool
    steady_state_excludes_compile: bool
    reports_compile_separately: bool
    measures_allocator_peak: bool

    def validate(self) -> None:
        """Reject combinations that would blur the three evidence levels."""

        if not self.contract_id:
            raise ValueError("benchmark contract_id must be non-empty")
        if not self.inputs_resident_on_cuda:
            raise ValueError("PR-12 contracts require CUDA-resident inputs")
        if (
            not self.steady_state_excludes_compile
            or not self.reports_compile_separately
        ):
            raise ValueError(
                "compile must be excluded from steady state and reported separately"
            )
        if (
            self.requires_preallocated_outputs
            == self.includes_required_output_allocation
        ):
            raise ValueError(
                "output allocation must be either excluded or included, not both"
            )
        if self.level == BenchmarkContractLevel.KERNEL:
            if any(
                (
                    self.includes_required_output_allocation,
                    self.includes_backend_dispatch,
                    self.includes_interop_and_stream_management,
                    self.includes_region_matching,
                    self.includes_planner,
                    self.includes_concretization,
                    self.measures_allocator_peak,
                )
            ):
                raise ValueError(
                    "kernel contract may only measure preallocated device kernels"
                )
        elif self.level == BenchmarkContractLevel.REGION_RUNTIME:
            if not all(
                (
                    self.includes_required_output_allocation,
                    self.includes_backend_dispatch,
                    self.includes_interop_and_stream_management,
                    self.measures_allocator_peak,
                )
            ):
                raise ValueError(
                    "region runtime must include dispatch, allocation and interop"
                )
            if any(
                (
                    self.includes_region_matching,
                    self.includes_planner,
                    self.includes_concretization,
                )
            ):
                raise ValueError("region runtime must exclude graph-level work")
        else:
            if not all(
                (
                    self.includes_required_output_allocation,
                    self.includes_backend_dispatch,
                    self.includes_interop_and_stream_management,
                    self.includes_region_matching,
                    self.includes_planner,
                    self.includes_concretization,
                    self.measures_allocator_peak,
                )
            ):
                raise ValueError(
                    "end-to-end contract must include the complete final-bound path"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible payload embedded in every result row."""

        self.validate()
        payload = asdict(self)
        payload["schema_version"] = BENCHMARK_CONTRACT_SCHEMA_VERSION
        payload["compliant"] = True
        payload["level"] = self.level.value
        payload["timing"] = {
            "cuda": "events_on_measured_stream",
            "host": "wall_time_with_measured_stream_sync",
            "timed_global_synchronize": False,
        }
        payload["memory"] = {
            "enabled": self.measures_allocator_peak,
            "allocator": "torch_cuda_allocator_peak_delta",
            "fields": (
                [
                    "peak_allocated_delta_bytes",
                    "peak_reserved_delta_bytes",
                    "output_bytes",
                    "temporary_workspace_upper_bound_bytes",
                ]
                if self.measures_allocator_peak
                else []
            ),
        }
        return payload

    def sha256(self) -> str:
        """Hash the canonical payload so manifests can pin the exact contract."""

        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


KERNEL_CONTRACT = BenchmarkContract(
    contract_id="pr12-kernel-preallocated-v1",
    level=BenchmarkContractLevel.KERNEL,
    inputs_resident_on_cuda=True,
    requires_preallocated_outputs=True,
    includes_required_output_allocation=False,
    includes_backend_dispatch=False,
    includes_interop_and_stream_management=False,
    includes_region_matching=False,
    includes_planner=False,
    includes_concretization=False,
    steady_state_excludes_compile=True,
    reports_compile_separately=True,
    measures_allocator_peak=False,
)

REGION_RUNTIME_CONTRACT = BenchmarkContract(
    contract_id="pr12-region-runtime-v1",
    level=BenchmarkContractLevel.REGION_RUNTIME,
    inputs_resident_on_cuda=True,
    requires_preallocated_outputs=False,
    includes_required_output_allocation=True,
    includes_backend_dispatch=True,
    includes_interop_and_stream_management=True,
    includes_region_matching=False,
    includes_planner=False,
    includes_concretization=False,
    steady_state_excludes_compile=True,
    reports_compile_separately=True,
    measures_allocator_peak=True,
)

END_TO_END_FINAL_BOUND_CONTRACT = BenchmarkContract(
    contract_id="pr12-end-to-end-final-bound-v1",
    level=BenchmarkContractLevel.END_TO_END_FINAL_BOUND,
    inputs_resident_on_cuda=True,
    requires_preallocated_outputs=False,
    includes_required_output_allocation=True,
    includes_backend_dispatch=True,
    includes_interop_and_stream_management=True,
    includes_region_matching=True,
    includes_planner=True,
    includes_concretization=True,
    steady_state_excludes_compile=True,
    reports_compile_separately=True,
    measures_allocator_peak=True,
)


def contract_from_payload(payload: Mapping[str, Any]) -> BenchmarkContract:
    """Reconstruct and validate a contract embedded in JSONL evidence."""

    if payload.get("schema_version") != BENCHMARK_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported PR-12 benchmark contract schema")
    field_names = {
        "contract_id",
        "level",
        "inputs_resident_on_cuda",
        "requires_preallocated_outputs",
        "includes_required_output_allocation",
        "includes_backend_dispatch",
        "includes_interop_and_stream_management",
        "includes_region_matching",
        "includes_planner",
        "includes_concretization",
        "steady_state_excludes_compile",
        "reports_compile_separately",
        "measures_allocator_peak",
    }
    values = {name: payload[name] for name in field_names}
    values["level"] = BenchmarkContractLevel(str(values["level"]))
    contract = BenchmarkContract(**values)
    contract.validate()
    return contract


__all__ = [
    "BENCHMARK_CONTRACT_SCHEMA_VERSION",
    "END_TO_END_FINAL_BOUND_CONTRACT",
    "KERNEL_CONTRACT",
    "REGION_RUNTIME_CONTRACT",
    "BenchmarkContract",
    "BenchmarkContractLevel",
    "contract_from_payload",
]
