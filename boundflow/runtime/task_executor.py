from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import torch

from ..domains.interval import IntervalDomain, IntervalState
from ..ir.task import BFTaskModule, TaskKind


@dataclass(frozen=True)
class LinfInputSpec:
    value_name: str
    center: torch.Tensor
    eps: float


class TaskExecutor(Protocol):
    def run_ibp(
        self, module: BFTaskModule, input_spec: LinfInputSpec, *, output_value: Optional[str] = None
    ) -> IntervalState: ...


class PythonTaskExecutor:
    """
    执行 BFTaskModule 的 reference executor。
    v0.1 仅支持 TaskKind.INTERVAL_IBP，并覆盖少量 primitive（linear/conv2d/relu/add/mul/flatten/reshape）。
    """

    def __init__(self, domain: Optional[IntervalDomain] = None):
        self.domain = domain or IntervalDomain()

    def run_ibp(
        self, module: BFTaskModule, input_spec: LinfInputSpec, *, output_value: Optional[str] = None
    ) -> IntervalState:
        module.validate()
        task = module.get_entry_task()
        if task.kind != TaskKind.INTERVAL_IBP:
            raise NotImplementedError(f"PythonTaskExecutor only supports INTERVAL_IBP, got {task.kind}")

        if input_spec.value_name not in task.input_values:
            raise ValueError(
                f"input_spec.value_name '{input_spec.value_name}' not in task.input_values: {task.input_values}"
            )

        params: Dict[str, Any] = {}
        raw_params = module.bindings.get("params", {})
        if isinstance(raw_params, dict):
            params.update(raw_params)

        x0 = input_spec.center
        eps = float(input_spec.eps)
        env: Dict[str, IntervalState] = {
            input_spec.value_name: IntervalState(lower=x0 - eps, upper=x0 + eps)
        }

        def get_interval(value_name: str) -> IntervalState:
            if value_name in env:
                return env[value_name]
            if value_name in params:
                t = params[value_name]
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t, device=x0.device)
                return IntervalState(lower=t, upper=t)
            raise KeyError(f"missing value in env/params: {value_name}")

        def get_tensor(value_name: str) -> Any:
            if value_name in params:
                return params[value_name]
            raise KeyError(f"missing param tensor: {value_name}")

        for op in task.ops:
            if op.op_type == "spec_linear":
                # y = C @ logits, where C shape [B,S,O], logits shape [B,O]
                logits = get_interval(op.inputs[0])
                C = get_tensor(op.inputs[1])
                if not torch.is_tensor(C):
                    C = torch.as_tensor(C, device=x0.device)
                if C.dim() != 3:
                    raise ValueError(f"spec_linear expects C rank-3 [B,S,O], got {tuple(C.shape)}")
                if logits.lower.dim() != 2:
                    raise ValueError(f"spec_linear expects logits rank-2 [B,O], got {tuple(logits.lower.shape)}")
                if C.shape[0] != logits.lower.shape[0] or C.shape[2] != logits.lower.shape[1]:
                    raise ValueError(
                        f"spec_linear shape mismatch: C={tuple(C.shape)} logits={tuple(logits.lower.shape)}"
                    )
                C_pos = torch.clamp(C, min=0.0)
                C_neg = torch.clamp(C, max=0.0)
                l = logits.lower.unsqueeze(1)
                u = logits.upper.unsqueeze(1)
                lb = (C_pos * l + C_neg * u).sum(dim=-1)
                ub = (C_pos * u + C_neg * l).sum(dim=-1)
                env[op.outputs[0]] = IntervalState(lower=lb, upper=ub)  # type: ignore[assignment]
                continue

            if op.op_type == "linear":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                if torch.is_tensor(w) and w.dim() == 3:
                    # Batched linear: w shape [B, O, I], b shape [B, O] (optional)
                    if b is None:
                        b = 0.0
                    if not torch.is_tensor(b):
                        b = torch.as_tensor(b, device=x0.device)
                    mid = (x.lower + x.upper) / 2.0
                    diff = (x.upper - x.lower) / 2.0
                    center = torch.bmm(mid.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
                    deviation = torch.bmm(diff.unsqueeze(1), w.abs().transpose(-1, -2)).squeeze(1)
                    lb = center - deviation + b
                    ub = center + deviation + b
                    env[op.outputs[0]] = IntervalState(lower=lb, upper=ub)  # type: ignore[assignment]
                else:
                    out = self.domain.affine_transformer(x, w, b, op="linear")
                    env[op.outputs[0]] = out  # type: ignore[assignment]
                continue

            if op.op_type == "conv2d":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                attrs = dict(op.attrs)
                attrs.setdefault("op", "conv2d")
                out = self.domain.affine_transformer(x, w, b, **attrs)
                env[op.outputs[0]] = out  # type: ignore[assignment]
                continue

            if op.op_type == "relu":
                x = get_interval(op.inputs[0])
                env[op.outputs[0]] = self.domain.relu_transformer(x)  # type: ignore[assignment]
                continue

            if op.op_type in ("add", "mul"):
                a = get_interval(op.inputs[0])
                b = get_interval(op.inputs[1])
                env[op.outputs[0]] = self.domain.elementwise_transformer(  # type: ignore[assignment]
                    [a, b], op=op.op_type
                )
                continue

            if op.op_type == "flatten":
                x = get_interval(op.inputs[0])
                start_dim = int(op.attrs.get("start_dim", 0))
                end_dim = int(op.attrs.get("end_dim", -1))
                env[op.outputs[0]] = IntervalState(  # type: ignore[assignment]
                    lower=torch.flatten(x.lower, start_dim=start_dim, end_dim=end_dim),
                    upper=torch.flatten(x.upper, start_dim=start_dim, end_dim=end_dim),
                )
                continue

            if op.op_type == "reshape":
                x = get_interval(op.inputs[0])
                shape = op.attrs.get("shape")
                if shape is None:
                    env[op.outputs[0]] = x
                    continue
                env[op.outputs[0]] = IntervalState(  # type: ignore[assignment]
                    lower=x.lower.reshape(shape),
                    upper=x.upper.reshape(shape),
                )
                continue

            if op.op_type in ("permute", "transpose"):
                x = get_interval(op.inputs[0])
                dims = op.attrs.get("dims")
                if not isinstance(dims, (list, tuple)):
                    raise ValueError(f"transpose missing dims for op '{op.name}': {dims}")
                dims = [int(d) for d in dims]
                env[op.outputs[0]] = IntervalState(  # type: ignore[assignment]
                    lower=x.lower.permute(*dims),
                    upper=x.upper.permute(*dims),
                )
                continue

            raise NotImplementedError(f"unsupported op_type in task executor: {op.op_type}")

        if output_value is None:
            if len(task.output_values) != 1:
                raise ValueError(f"task has {len(task.output_values)} outputs; specify output_value explicitly")
            output_value = task.output_values[0]
        return get_interval(output_value)
