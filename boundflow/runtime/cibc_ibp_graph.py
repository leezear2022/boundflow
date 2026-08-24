"""Prepared CUDA-graph execution for CIBC horizontal IBP model flow."""

# pylint: disable=too-many-locals,too-many-arguments,missing-function-docstring
# pylint: disable=too-many-instance-attributes,import-outside-toplevel
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, cast

import torch

from boundflow.domains.interval import IntervalDomain, IntervalState
from boundflow.runtime.dag_utils import (
    normalize_concat_axis,
    validate_concat_tensor_shapes,
)
from boundflow.ir.task import BFTaskModule
from boundflow.runtime.cibc_ibp_conv import use_cibc_ibp_conv_v1

OpContextFactory = Callable[[int, str], AbstractContextManager[None]]


def run_cibc_ibp_graph_once_v1(
    module: BFTaskModule,
    *,
    input_value: str,
    input_lower: torch.Tensor,
    input_upper: torch.Tensor,
    threads_per_block: int | None,
    op_context_factory: OpContextFactory | None = None,
) -> tuple[dict[str, IntervalState], int]:
    task = module.get_entry_task()
    params = dict(module.bindings.get("params", {}))
    env: dict[str, IntervalState] = {
        input_value: IntervalState(lower=input_lower, upper=input_upper)
    }
    domain = IntervalDomain()

    def state(name: str) -> IntervalState:
        value = env.get(name)
        if value is not None:
            return value
        tensor = params[name]
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor, device=input_lower.device)
        return IntervalState(lower=tensor, upper=tensor)

    context_manager = (
        use_cibc_ibp_conv_v1(threads_per_block=threads_per_block)
        if threads_per_block is not None
        else None
    )

    def execute() -> None:
        for ordinal, op in enumerate(task.ops):
            context = (
                op_context_factory(ordinal, op.op_type)
                if op_context_factory is not None
                else nullcontext()
            )
            with context:
                if op.op_type in {"linear", "conv2d"}:
                    x = state(op.inputs[0])
                    weight = params[op.inputs[1]]
                    bias = params[op.inputs[2]] if len(op.inputs) == 3 else None
                    attrs: dict[str, Any] = dict(op.attrs)
                    attrs.setdefault("op", op.op_type)
                    env[op.outputs[0]] = cast(
                        IntervalState,
                        domain.affine_transformer(x, weight, bias, **attrs),
                    )
                elif op.op_type == "relu":
                    env[op.outputs[0]] = cast(
                        IntervalState, domain.relu_transformer(state(op.inputs[0]))
                    )
                elif op.op_type == "add":
                    env[op.outputs[0]] = cast(
                        IntervalState,
                        domain.elementwise_transformer(
                            [state(op.inputs[0]), state(op.inputs[1])], "add"
                        ),
                    )
                elif op.op_type == "flatten":
                    x = state(op.inputs[0])
                    env[op.outputs[0]] = IntervalState(
                        lower=torch.flatten(x.lower, 1, -1),
                        upper=torch.flatten(x.upper, 1, -1),
                    )
                elif op.op_type == "concat":
                    parts = [state(name) for name in op.inputs]
                    axis = normalize_concat_axis(
                        op.attrs.get("axis", 1),
                        rank_with_batch=parts[0].lower.dim(),
                        caller="run_cibc_ibp_graph_once_v1",
                    )
                    validate_concat_tensor_shapes(
                        [tuple(item.lower.shape) for item in parts],
                        axis=axis,
                        caller="run_cibc_ibp_graph_once_v1",
                    )
                    env[op.outputs[0]] = IntervalState(
                        lower=torch.cat([item.lower for item in parts], dim=axis),
                        upper=torch.cat([item.upper for item in parts], dim=axis),
                    )
                else:
                    raise NotImplementedError(
                        f"CIBC IBP graph op differs: {op.op_type}"
                    )

    if context_manager is None:
        execute()
        return env, 0
    with context_manager as context:
        execute()
    return env, context.launch_count


@dataclass
class CIBCIBPCUDAGraphPlanV1:
    """Static-address CUDA graph with prepared input and output buffers."""

    module: BFTaskModule
    input_value: str
    input_lower: torch.Tensor
    input_upper: torch.Tensor
    threads_per_block: int | None
    op_context_factory: OpContextFactory | None = None

    def __post_init__(self) -> None:
        import tvm_ffi

        if self.input_lower.device.type != "cuda":
            raise ValueError("CIBC IBP CUDA graph requires CUDA")
        self.input_lower = self.input_lower.contiguous().clone()
        self.input_upper = self.input_upper.contiguous().clone()
        capture_stream = torch.cuda.Stream(device=self.input_lower.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.input_lower.device))
        with torch.cuda.stream(capture_stream):
            with tvm_ffi.use_torch_stream():
                for _ in range(3):
                    _outputs, _launch_count = run_cibc_ibp_graph_once_v1(
                        self.module,
                        input_value=self.input_value,
                        input_lower=self.input_lower,
                        input_upper=self.input_upper,
                        threads_per_block=self.threads_per_block,
                        op_context_factory=self.op_context_factory,
                    )
        capture_stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            with tvm_ffi.use_torch_stream():
                with torch.cuda.graph(self.graph, stream=capture_stream):
                    self.outputs, self.launch_count = run_cibc_ibp_graph_once_v1(
                        self.module,
                        input_value=self.input_value,
                        input_lower=self.input_lower,
                        input_upper=self.input_upper,
                        threads_per_block=self.threads_per_block,
                        op_context_factory=self.op_context_factory,
                    )
        if self.threads_per_block is not None and self.launch_count != 6:
            raise ValueError("CIBC IBP CUDA graph Conv coverage differs")

    def replay(
        self,
        *,
        input_lower: torch.Tensor | None = None,
        input_upper: torch.Tensor | None = None,
    ) -> dict[str, IntervalState]:
        if (input_lower is None) != (input_upper is None):
            raise ValueError("CIBC IBP CUDA graph replay input pair differs")
        if input_lower is not None and input_upper is not None:
            if (
                tuple(input_lower.shape) != tuple(self.input_lower.shape)
                or tuple(input_upper.shape) != tuple(self.input_upper.shape)
                or input_lower.dtype != self.input_lower.dtype
                or input_upper.dtype != self.input_upper.dtype
                or input_lower.device != self.input_lower.device
                or input_upper.device != self.input_upper.device
            ):
                raise ValueError("CIBC IBP CUDA graph replay input contract differs")
            self.input_lower.copy_(input_lower)
            self.input_upper.copy_(input_upper)
        self.graph.replay()
        return self.outputs


__all__ = ["CIBCIBPCUDAGraphPlanV1", "run_cibc_ibp_graph_once_v1"]
