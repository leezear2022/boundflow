from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ..domains.interval import IntervalDomain, IntervalState
from ..ir.task import BFTaskModule, TaskKind
from ..ir.task import StoragePlan
from ..ir.task import TaskOp
from ..runtime.task_executor import LinfInputSpec
from ..backends.tvm.interval_linear import IntervalLinearKey, build_interval_linear_module
from ..backends.tvm.interval_conv2d import IntervalConv2dKey, build_interval_conv2d_module
from ..backends.tvm.relax_interval_linear import build_relax_interval_linear_vm_exec
from ..backends.tvm.relax_interval_conv2d import build_relax_interval_conv2d_vm_exec
from ..backends.tvm.relax_task_lowering import RelaxLoweringMode, build_interval_linear_relax_ir_module


@dataclass(frozen=True)
class TVMExecutorOptions:
    target: Optional[str] = None  # e.g. "llvm" or "cuda"
    kernel_style: str = "relax"  # "relax" (preferred), "call_tir", or "te" (legacy demo)


@dataclass
class TVMRunStats:
    tvm_ops: List[str]
    fallback_ops: List[str]
    linear_kernel_cache: Dict[str, int]
    conv2d_kernel_cache: Dict[str, int]
    kernel_style: str


class TVMTaskExecutor:
    """
    v0: Python driver + TVM compiled kernels.
    目标是打通“同一个 BFTaskModule：PythonTaskExecutor vs TVMTaskExecutor 输出一致”的通路，
    暂不把复杂编排/控制流塞进 Relax VM。
    """

    def __init__(self, *, options: Optional[TVMExecutorOptions] = None):
        self.options = options or TVMExecutorOptions()
        self.domain = IntervalDomain()
        self.last_stats: Optional[TVMRunStats] = None
        # (kernel_style, IntervalLinearKey) -> relax.Executable
        self._linear_exec_cache: Dict[tuple[str, IntervalLinearKey], Any] = {}

    def _select_target(self) -> str:
        import tvm

        if self.options.target:
            return self.options.target
        if tvm.runtime.enabled("llvm"):
            return "llvm"
        if tvm.runtime.enabled("cuda"):
            return "cuda"
        return "llvm"

    def _compile_interval_linear_executable(self, key: IntervalLinearKey):
        import tvm
        from tvm import relax

        k = (str(self.options.kernel_style), key)
        if k in self._linear_exec_cache:
            return self._linear_exec_cache[k]

        if self.options.kernel_style == "call_tir":
            ir_mod = build_interval_linear_relax_ir_module(key, mode=RelaxLoweringMode.CALL_TIR, relax_func_name="main")
            ex = relax.build(ir_mod, target=key.target)
        elif self.options.kernel_style == "relax":
            ex = build_relax_interval_linear_vm_exec(key)
        else:
            raise ValueError(f"unsupported kernel_style for relax executable: {self.options.kernel_style}")

        self._linear_exec_cache[k] = ex
        return ex

    def _buf(self, storage_plan: StoragePlan, value_name: str) -> str:
        logical = storage_plan.value_to_buffer.get(value_name)
        if logical is None:
            raise KeyError(f"value not found in storage_plan: {value_name}")
        phys = storage_plan.to_physical(logical)
        if storage_plan.physical_buffers and phys not in storage_plan.physical_buffers:
            raise KeyError(f"physical buffer_id not found in storage_plan.physical_buffers: {phys} (value={value_name})")
        return phys

    def run_ibp_task(self, task, *, env: Dict[str, IntervalState], params: Dict[str, Any], storage_plan) -> None:
        """
        Execute a single INTERVAL_IBP task in-place on a shared env (BufferEnv: physical_id -> IntervalState).

        v0: accelerate `linear` via TVM Relax (kernel_style=relax/call_tir); other ops fall back to torch.
        """
        from ..ir.task import TaskKind  # local import to avoid cycles

        if task.kind != TaskKind.INTERVAL_IBP:
            raise NotImplementedError(f"TVMTaskExecutor only supports INTERVAL_IBP, got {task.kind}")
        if not isinstance(storage_plan, StoragePlan):
            storage_plan = storage_plan  # type: ignore[assignment]

        # Device anchor for param constants.
        device = None
        for v in env.values():
            device = v.lower.device
            break
        if device is None:
            for p in params.values():
                if torch.is_tensor(p):
                    device = p.device
                    break

        def get_interval(value_name: str) -> IntervalState:
            bid = self._buf(storage_plan, value_name)
            if bid in env:
                return env[bid]
            if value_name in params:
                t = params[value_name]
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t, device=device)
                s = IntervalState(lower=t, upper=t)
                env[bid] = s
                return s
            raise KeyError(f"missing value in env/params: {value_name}")

        def get_tensor(value_name: str) -> Any:
            if value_name in params:
                return params[value_name]
            raise KeyError(f"missing param tensor: {value_name}")

        target = self._select_target()
        import tvm
        from tvm.runtime import _tensor as rt
        from tvm import relax

        dev = tvm.cpu(0) if target == "llvm" else tvm.cuda(0)

        def _run_linear(x: IntervalState, w: torch.Tensor, b: torch.Tensor) -> IntervalState:
            B, I = x.lower.shape
            O = w.shape[0]
            dtype = str(x.lower.dtype).replace("torch.", "")
            key = IntervalLinearKey(batch=int(B), in_features=int(I), out_features=int(O), dtype=dtype, target=target)
            ex = self._compile_interval_linear_executable(key)
            vm = relax.VirtualMachine(ex, dev)
            x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
            x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
            w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
            b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
            out = vm["main"](x_l_t, x_u_t, w_t, b_t)
            y_l_t, y_u_t = out[0], out[1]
            y_l = torch.from_numpy(y_l_t.numpy()).to(x.lower.device)
            y_u = torch.from_numpy(y_u_t.numpy()).to(x.lower.device)
            return IntervalState(lower=y_l, upper=y_u)

        for op in task.ops:
            if not isinstance(op, TaskOp):
                raise TypeError(f"unexpected op type: {type(op)}")

            if op.op_type == "linear":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                if not torch.is_tensor(w):
                    w = torch.as_tensor(w, device=x.lower.device)
                if b is None:
                    b = torch.zeros((w.shape[0],), dtype=w.dtype, device=w.device)
                if not torch.is_tensor(b):
                    b = torch.as_tensor(b, device=w.device)

                # Only accelerate 2D weight.
                if w.dim() == 2 and x.lower.dim() == 2 and self.options.kernel_style in ("relax", "call_tir"):
                    out = _run_linear(x, w, b)
                else:
                    out = self.domain.affine_transformer(x, w, b, op="linear")
                env[self._buf(storage_plan, op.outputs[0])] = out  # type: ignore[assignment]
                continue

            if op.op_type == "relu":
                x = get_interval(op.inputs[0])
                env[self._buf(storage_plan, op.outputs[0])] = self.domain.relu_transformer(x)  # type: ignore[assignment]
                continue

            if op.op_type in ("add", "mul"):
                a = get_interval(op.inputs[0])
                b = get_interval(op.inputs[1])
                env[self._buf(storage_plan, op.outputs[0])] = self.domain.elementwise_transformer([a, b], op=op.op_type)  # type: ignore[assignment]
                continue

            # Fallback: keep behavior consistent with PythonTaskExecutor by delegating to domain for known ops.
            raise NotImplementedError(f"unsupported op_type in TVMTaskExecutor.run_ibp_task: {op.op_type}")

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
        from tvm import relax

        dev = tvm.cpu(0) if target == "llvm" else tvm.cuda(0)
        tvm_ops: List[str] = []
        fallback_ops: List[str] = []

        def _cache_info_dict(info: Any) -> Dict[str, int]:
            # functools._lru_cache_wrapper.cache_info() returns a namedtuple.
            return {
                "hits": int(getattr(info, "hits", 0)),
                "misses": int(getattr(info, "misses", 0)),
                "maxsize": int(getattr(info, "maxsize", 0) or 0),
                "currsize": int(getattr(info, "currsize", 0)),
            }

        def _as_int_tuple(value: Any, *, dim: int) -> List[int]:
            if isinstance(value, int):
                return [int(value)] * dim
            if isinstance(value, (list, tuple)):
                if len(value) == dim:
                    return [int(v) for v in value]
                if len(value) == 1:
                    return [int(value[0])] * dim
            raise ValueError(f"invalid int tuple: {value} (expected dim={dim})")

        def _run_relax_linear(key: IntervalLinearKey, x: IntervalState, w: torch.Tensor, b: torch.Tensor) -> IntervalState:
            ex = build_relax_interval_linear_vm_exec(key)
            vm = relax.VirtualMachine(ex, dev)
            x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
            x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
            w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
            b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
            out = vm["main"](x_l_t, x_u_t, w_t, b_t)
            y_l_t, y_u_t = out[0], out[1]
            y_l = torch.from_numpy(y_l_t.numpy()).to(x0.device)
            y_u = torch.from_numpy(y_u_t.numpy()).to(x0.device)
            return IntervalState(lower=y_l, upper=y_u)

        def _run_relax_conv2d(key: IntervalConv2dKey, x: IntervalState, w: torch.Tensor, b: torch.Tensor) -> IntervalState:
            ex = build_relax_interval_conv2d_vm_exec(key)
            vm = relax.VirtualMachine(ex, dev)
            x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
            x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
            w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
            b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
            out = vm["main"](x_l_t, x_u_t, w_t, b_t)
            y_l_t, y_u_t = out[0], out[1]
            y_l = torch.from_numpy(y_l_t.numpy()).to(x0.device)
            y_u = torch.from_numpy(y_u_t.numpy()).to(x0.device)
            return IntervalState(lower=y_l, upper=y_u)

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
                    fallback_ops.append("linear")
                    out = self.domain.affine_transformer(x, w, b, op="linear")
                    env[op.outputs[0]] = out  # type: ignore[assignment]
                    continue

                B, I = x.lower.shape
                O = w.shape[0]
                dtype = str(x.lower.dtype).replace("torch.", "")
                key = IntervalLinearKey(batch=int(B), in_features=int(I), out_features=int(O), dtype=dtype, target=target)
                if self.options.kernel_style == "te":
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
                else:
                    env[op.outputs[0]] = _run_relax_linear(key, x, w, b)  # type: ignore[assignment]
                tvm_ops.append("linear")
                continue

            if op.op_type == "conv2d":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) >= 3 else None
                if not torch.is_tensor(w):
                    w = torch.as_tensor(w)
                if b is None:
                    b = torch.zeros((w.shape[0],), dtype=w.dtype)
                if not torch.is_tensor(b):
                    b = torch.as_tensor(b)

                stride = _as_int_tuple(op.attrs.get("stride", 1), dim=2)
                padding = _as_int_tuple(op.attrs.get("padding", 0), dim=2)
                dilation = _as_int_tuple(op.attrs.get("dilation", 1), dim=2)
                groups = int(op.attrs.get("groups", 1))

                if x.lower.dim() != 4 or w.dim() != 4 or groups != 1:
                    fallback_ops.append("conv2d")
                    out = self.domain.affine_transformer(
                        x,
                        w,
                        b,
                        op="conv2d",
                        stride=tuple(stride),
                        padding=tuple(padding),
                        dilation=tuple(dilation),
                        groups=groups,
                    )
                    env[op.outputs[0]] = out  # type: ignore[assignment]
                    continue

                N, CI, H, W = x.lower.shape
                CO, CI_w, KH, KW = w.shape
                if CI_w != CI:
                    fallback_ops.append("conv2d")
                    out = self.domain.affine_transformer(
                        x,
                        w,
                        b,
                        op="conv2d",
                        stride=tuple(stride),
                        padding=tuple(padding),
                        dilation=tuple(dilation),
                        groups=groups,
                    )
                    env[op.outputs[0]] = out  # type: ignore[assignment]
                    continue

                dtype = str(x.lower.dtype).replace("torch.", "")
                key = IntervalConv2dKey(
                    batch=int(N),
                    in_channels=int(CI),
                    in_h=int(H),
                    in_w=int(W),
                    out_channels=int(CO),
                    k_h=int(KH),
                    k_w=int(KW),
                    stride_h=int(stride[0]),
                    stride_w=int(stride[1]),
                    pad_h=int(padding[0]),
                    pad_w=int(padding[1]),
                    dilation_h=int(dilation[0]),
                    dilation_w=int(dilation[1]),
                    groups=int(groups),
                    dtype=dtype,
                    target=target,
                )
                if self.options.kernel_style == "te":
                    func = build_interval_conv2d_module(key)
                    oh = (int(H) + 2 * int(padding[0]) - int(dilation[0]) * (int(KH) - 1) - 1) // int(stride[0]) + 1
                    ow = (int(W) + 2 * int(padding[1]) - int(dilation[1]) * (int(KW) - 1) - 1) // int(stride[1]) + 1
                    x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
                    x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
                    w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
                    b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
                    y_l_t = rt.empty((int(N), int(CO), int(oh), int(ow)), dtype=dtype, device=dev)
                    y_u_t = rt.empty((int(N), int(CO), int(oh), int(ow)), dtype=dtype, device=dev)
                    func(x_l_t, x_u_t, w_t, b_t, y_l_t, y_u_t)
                    y_l = torch.from_numpy(y_l_t.numpy()).to(x0.device)
                    y_u = torch.from_numpy(y_u_t.numpy()).to(x0.device)
                    env[op.outputs[0]] = IntervalState(lower=y_l, upper=y_u)
                else:
                    env[op.outputs[0]] = _run_relax_conv2d(key, x, w, b)  # type: ignore[assignment]
                tvm_ops.append("conv2d")
                continue

            if op.op_type == "relu":
                x = get_interval(op.inputs[0])
                env[op.outputs[0]] = self.domain.relu_transformer(x)  # type: ignore[assignment]
                fallback_ops.append("relu")
                continue

            if op.op_type == "add":
                a = get_interval(op.inputs[0])
                b = get_interval(op.inputs[1])
                env[op.outputs[0]] = self.domain.elementwise_transformer([a, b], op="add")  # type: ignore[assignment]
                fallback_ops.append("add")
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
                fallback_ops.append("flatten")
                continue

            if op.op_type == "reshape":
                x = get_interval(op.inputs[0])
                shape = op.attrs.get("shape")
                if shape is None:
                    env[op.outputs[0]] = x
                    fallback_ops.append("reshape")
                    continue
                env[op.outputs[0]] = IntervalState(lower=x.lower.reshape(shape), upper=x.upper.reshape(shape))
                fallback_ops.append("reshape")
                continue

            if op.op_type in ("permute", "transpose"):
                x = get_interval(op.inputs[0])
                dims = op.attrs.get("dims")
                if not isinstance(dims, (list, tuple)):
                    raise ValueError(f"transpose missing dims for op '{op.name}': {dims}")
                dims = [int(d) for d in dims]
                env[op.outputs[0]] = IntervalState(lower=x.lower.permute(*dims), upper=x.upper.permute(*dims))
                fallback_ops.append("permute")
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
                fallback_ops.append("spec_linear")
                continue

            raise NotImplementedError(f"unsupported op_type in TVMTaskExecutor: {op.op_type}")

        if output_value is None:
            if len(task.output_values) != 1:
                raise ValueError(f"task has {len(task.output_values)} outputs; specify output_value explicitly")
            output_value = task.output_values[0]
        self.last_stats = TVMRunStats(
            tvm_ops=tvm_ops,
            fallback_ops=fallback_ops,
            linear_kernel_cache=_cache_info_dict(
                (build_interval_linear_module if self.options.kernel_style == "te" else build_relax_interval_linear_vm_exec).cache_info()
            ),
            conv2d_kernel_cache=_cache_info_dict(
                (build_interval_conv2d_module if self.options.kernel_style == "te" else build_relax_interval_conv2d_vm_exec).cache_info()
            ),
            kernel_style=self.options.kernel_style,
        )
        return get_interval(output_value)
