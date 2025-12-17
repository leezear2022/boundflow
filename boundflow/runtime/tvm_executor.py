from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ..domains.interval import IntervalDomain, IntervalState
from ..ir.task import BFTaskModule, TaskKind
from ..runtime.task_executor import LinfInputSpec
from ..backends.tvm.interval_linear import IntervalLinearKey, build_interval_linear_module


@dataclass(frozen=True)
class TVMExecutorOptions:
    target: Optional[str] = None  # e.g. "llvm" or "cuda"


class TVMTaskExecutor:
    """
    v0: Python driver + TVM compiled kernels.
    目标是打通“同一个 BFTaskModule：PythonTaskExecutor vs TVMTaskExecutor 输出一致”的通路，
    暂不把复杂编排/控制流塞进 Relax VM。
    """

    def __init__(self, *, options: Optional[TVMExecutorOptions] = None):
        self.options = options or TVMExecutorOptions()
        self.domain = IntervalDomain()

    def _select_target(self) -> str:
        import tvm

        if self.options.target:
            return self.options.target
        if tvm.runtime.enabled("llvm"):
            return "llvm"
        if tvm.runtime.enabled("cuda"):
            return "cuda"
        return "llvm"

    def run_ibp(
        self, module: BFTaskModule, input_spec: LinfInputSpec, *, output_value: Optional[str] = None
    ) -> IntervalState:
        module.validate()
        task = module.get_entry_task()
        if task.kind != TaskKind.INTERVAL_IBP:
            raise NotImplementedError(f"TVMTaskExecutor only supports INTERVAL_IBP, got {task.kind}")

        target = self._select_target()

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

        # v0: only accelerate non-batched linear (2D weight). Other ops run on torch.
        import tvm
        from tvm.runtime import _tensor as rt

        dev = tvm.cpu(0) if target == "llvm" else tvm.cuda(0)

        for op in task.ops:
            if op.op_type == "linear":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                if not torch.is_tensor(w):
                    w = torch.as_tensor(w)
                if b is None:
                    b = torch.zeros((w.shape[0],), dtype=w.dtype)
                if not torch.is_tensor(b):
                    b = torch.as_tensor(b)

                if w.dim() != 2:
                    # fallback for batched/spec-fused linear (handled by PythonTaskExecutor today)
                    out = self.domain.affine_transformer(x, w, b, op="linear")
                    env[op.outputs[0]] = out  # type: ignore[assignment]
                    continue

                B, I = x.lower.shape
                O = w.shape[0]
                dtype = str(x.lower.dtype).replace("torch.", "")
                key = IntervalLinearKey(batch=int(B), in_features=int(I), out_features=int(O), dtype=dtype, target=target)
                func = build_interval_linear_module(key)

                x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
                x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
                w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
                b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
                y_l_t = rt.empty((B, O), dtype=dtype, device=dev)
                y_u_t = rt.empty((B, O), dtype=dtype, device=dev)
                func(x_l_t, x_u_t, w_t, b_t, y_l_t, y_u_t)
                y_l = torch.from_numpy(y_l_t.numpy()).to(x0.device)
                y_u = torch.from_numpy(y_u_t.numpy()).to(x0.device)
                env[op.outputs[0]] = IntervalState(lower=y_l, upper=y_u)
                continue

            if op.op_type == "relu":
                x = get_interval(op.inputs[0])
                env[op.outputs[0]] = self.domain.relu_transformer(x)  # type: ignore[assignment]
                continue

            if op.op_type == "add":
                a = get_interval(op.inputs[0])
                b = get_interval(op.inputs[1])
                env[op.outputs[0]] = self.domain.elementwise_transformer([a, b], op="add")  # type: ignore[assignment]
                continue

            # v0: fallback to torch semantics for remaining ops
            if op.op_type == "flatten":
                x = get_interval(op.inputs[0])
                start_dim = int(op.attrs.get("start_dim", 0))
                end_dim = int(op.attrs.get("end_dim", -1))
                env[op.outputs[0]] = IntervalState(
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
                env[op.outputs[0]] = IntervalState(lower=x.lower.reshape(shape), upper=x.upper.reshape(shape))
                continue

            if op.op_type in ("permute", "transpose"):
                x = get_interval(op.inputs[0])
                dims = op.attrs.get("dims")
                if not isinstance(dims, (list, tuple)):
                    raise ValueError(f"transpose missing dims for op '{op.name}': {dims}")
                dims = [int(d) for d in dims]
                env[op.outputs[0]] = IntervalState(lower=x.lower.permute(*dims), upper=x.upper.permute(*dims))
                continue

            if op.op_type == "spec_linear":
                # v0: keep spec_linear on torch (not performance critical compared to affine core).
                logits = get_interval(op.inputs[0])
                C = get_tensor(op.inputs[1])
                if not torch.is_tensor(C):
                    C = torch.as_tensor(C, device=x0.device)
                C_pos = torch.clamp(C, min=0.0)
                C_neg = torch.clamp(C, max=0.0)
                l = logits.lower.unsqueeze(1)
                u = logits.upper.unsqueeze(1)
                lb = (C_pos * l + C_neg * u).sum(dim=-1)
                ub = (C_pos * u + C_neg * l).sum(dim=-1)
                env[op.outputs[0]] = IntervalState(lower=lb, upper=ub)
                continue

            raise NotImplementedError(f"unsupported op_type in TVMTaskExecutor: {op.op_type}")

        if output_value is None:
            if len(task.output_values) != 1:
                raise ValueError(f"task has {len(task.output_values)} outputs; specify output_value explicitly")
            output_value = task.output_values[0]
        return get_interval(output_value)
