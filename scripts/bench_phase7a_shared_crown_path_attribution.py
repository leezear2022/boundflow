from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch

from boundflow.domains.interval import IntervalState
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.bound_planner import (
    FinalConcretizationPolicy,
    FinalConcretizationRequest,
    phase7a_capability_table_jsonable,
    phase7b_cost_model_rules_jsonable,
    plan_phase7b_shared_crown,
)
from boundflow.runtime import crown_ibp as crown_mod
from boundflow.runtime import linear_operator as linear_op_mod
from boundflow.runtime.crown_ibp import AffineBackwardState, get_crown_ibp_mlp_stats, run_crown_ibp_mlp
from boundflow.runtime.linear_operator import DenseLinearOperator
from boundflow.runtime.perturbation import final_concretization_policy
from boundflow.runtime.task_executor import InputSpec

_TimerMode = Literal["perf_counter", "torch_benchmark"]
_Variant = Literal["structured", "dense_relu", "dense_layout"]
_CompareTarget = Literal["relu_barrier", "layout_only"]


@dataclass(frozen=True)
class Row:
    workload: str
    compare_target: str
    structured_ms_p50: float
    baseline_ms_p50: float
    speedup: float
    counts_structured: Dict[str, Any]
    counts_baseline: Dict[str, Any]
    planner_decision: Dict[str, Any]


@dataclass
class _Counters:
    relu_backward_calls: int = 0
    permute_backward_calls: int = 0
    split_pos_neg_dense_total: int = 0
    split_pos_neg_dense_by_op: Dict[str, int] = field(default_factory=dict)
    dense_relu_barrier_calls: int = 0
    dense_layout_barrier_calls: int = 0
    operator_attribution: Optional[Dict[str, Any]] = None

    def to_jsonable(self) -> Dict[str, Any]:
        out = {
            "relu_backward_calls": int(self.relu_backward_calls),
            "permute_backward_calls": int(self.permute_backward_calls),
            "split_pos_neg_dense_total": int(self.split_pos_neg_dense_total),
            "split_pos_neg_dense_by_op": {
                k: int(v) for k, v in sorted(self.split_pos_neg_dense_by_op.items())
            },
            "dense_relu_barrier_calls": int(self.dense_relu_barrier_calls),
            "dense_layout_barrier_calls": int(self.dense_layout_barrier_calls),
        }
        if self.operator_attribution is not None:
            out["operator_attribution"] = self.operator_attribution
        return out


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _percentile_ms(samples_s: Iterable[float], q: float) -> float:
    xs = sorted(samples_s)
    if not xs:
        return 0.0
    k = int(round((len(xs) - 1) * q))
    return float(xs[k]) * 1000.0


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_call_perf_counter(fn: Callable[[], None], *, warmup: int, iters: int, device: torch.device) -> float:
    with torch.inference_mode():
        for _ in range(int(warmup)):
            fn()
        _sync_device(device)
        times: List[float] = []
        for _ in range(int(iters)):
            t0 = time.perf_counter()
            fn()
            _sync_device(device)
            times.append(time.perf_counter() - t0)
    return _percentile_ms(times, 0.5)


def _time_call_torch_benchmark(fn: Callable[[], None], *, warmup: int, device: torch.device, min_run_time_s: float) -> float:
    import torch.utils.benchmark as benchmark

    def wrapped() -> None:
        fn()
        _sync_device(device)

    with torch.inference_mode():
        for _ in range(int(warmup)):
            wrapped()
        measurement = benchmark.Timer(
            stmt="wrapped()",
            globals={"wrapped": wrapped},
        ).blocked_autorange(min_run_time=float(min_run_time_s))
    return float(measurement.median) * 1000.0


def _time_variant(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    timer: _TimerMode,
    torch_benchmark_min_run_time_s: float,
    device: torch.device,
) -> float:
    if timer == "torch_benchmark":
        return _time_call_torch_benchmark(
            fn,
            warmup=warmup,
            device=device,
            min_run_time_s=float(torch_benchmark_min_run_time_s),
        )
    return _time_call_perf_counter(fn, warmup=warmup, iters=iters, device=device)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    if device.type == "mps":
        return "apple_mps"
    return "cpu"


def _mps_built() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _mps_fallback_enabled() -> bool:
    return str(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")).strip().lower() in {"1", "true", "yes", "on"}


def _make_device(raw: str, *, dtype_name: str, allow_mps_fallback: bool = False) -> torch.device:
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device.type == "mps":
        if not _mps_built():
            raise RuntimeError("MPS requested but this PyTorch build does not include MPS")
        if not _mps_available():
            raise RuntimeError("MPS requested but not available on this machine")
        if str(dtype_name) != "float32":
            raise RuntimeError("MPS benchmark lane currently supports only --dtype float32")
        if _mps_fallback_enabled() and not bool(allow_mps_fallback):
            raise RuntimeError(
                "PYTORCH_ENABLE_MPS_FALLBACK is enabled; rerun with fallback disabled or pass --allow-mps-fallback"
            )
    return device


def _device_meta(device: torch.device, *, allow_mps_fallback: bool = False) -> Dict[str, Any]:
    return {
        "mps_built": _mps_built(),
        "mps_available": _mps_available(),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "mps_fallback_enabled": _mps_fallback_enabled(),
        "mps_fallback_allowed": bool(allow_mps_fallback),
        "mps_prefer_metal_env": os.environ.get("PYTORCH_MPS_PREFER_METAL"),
        "mps_fast_math_env": os.environ.get("PYTORCH_MPS_FAST_MATH"),
    }


def _dense_relu_barrier_step(
    state: AffineBackwardState,
    *,
    pre: IntervalState,
    x_name: str,
    relu_alpha: Optional[Dict[str, torch.Tensor]],
    relu_pre_add_coeff_u: Optional[Dict[str, torch.Tensor]],
    relu_pre_add_coeff_l: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> AffineBackwardState:
    batch = int(pre.lower.shape[0])
    input_shape = tuple(int(dim) for dim in pre.lower.shape[1:])
    pre_flat = IntervalState(lower=pre.lower.reshape(batch, -1), upper=pre.upper.reshape(batch, -1))
    if pre_flat.lower.shape[1] != state.A_u.input_numel or pre_flat.lower.shape[1] != state.A_l.input_numel:
        raise ValueError(
            f"{caller} relu backward shape mismatch: pre={tuple(pre.lower.shape)} "
            f"A_u.input_shape={state.A_u.input_shape} A_l.input_shape={state.A_l.input_shape}"
        )

    with linear_op_mod.operator_attribution_reason("dense_baseline_materialization"):
        A_u = state.A_u.to_dense()
        A_l = state.A_l.to_dense()
    b_u = state.b_u.clone()
    b_l = state.b_l.clone()

    alpha_u, beta_u, alpha_l, beta_l = crown_mod._relu_relax(pre_flat.lower, pre_flat.upper)
    if relu_alpha is not None and x_name in relu_alpha:
        alpha = crown_mod._broadcast_relu_alpha(
            relu_alpha[x_name],
            pre=pre,
            x_name=x_name,
            device=device,
            dtype=dtype,
            caller=caller,
        )
        amb = (pre_flat.lower < 0) & (pre_flat.upper > 0)
        if amb.any():
            alpha_l = torch.where(amb, alpha, alpha_l)

    sel_alpha_u = torch.where(A_u >= 0, alpha_u.unsqueeze(1), alpha_l.unsqueeze(1))
    sel_beta_u = torch.where(A_u >= 0, beta_u.unsqueeze(1), beta_l.unsqueeze(1))
    b_u = b_u + (A_u * sel_beta_u).sum(dim=2)
    A_u = A_u * sel_alpha_u
    if relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u:
        add_u = crown_mod._broadcast_relu_pre_add_coeff(
            relu_pre_add_coeff_u[x_name],
            batch=batch,
            flat_dim=int(pre_flat.lower.shape[1]),
            x_name=x_name,
            label="relu_pre_add_coeff_u",
            device=device,
            dtype=dtype,
        )
        A_u = A_u + add_u.unsqueeze(1)

    sel_alpha_l = torch.where(A_l >= 0, alpha_l.unsqueeze(1), alpha_u.unsqueeze(1))
    sel_beta_l = torch.where(A_l >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1))
    b_l = b_l + (A_l * sel_beta_l).sum(dim=2)
    A_l = A_l * sel_alpha_l
    if relu_pre_add_coeff_l is not None and x_name in relu_pre_add_coeff_l:
        add_l = crown_mod._broadcast_relu_pre_add_coeff(
            relu_pre_add_coeff_l[x_name],
            batch=batch,
            flat_dim=int(pre_flat.lower.shape[1]),
            x_name=x_name,
            label="relu_pre_add_coeff_l",
            device=device,
            dtype=dtype,
        )
        A_l = A_l + add_l.unsqueeze(1)

    return AffineBackwardState(
        A_u=DenseLinearOperator(A_u, input_shape=input_shape),
        A_l=DenseLinearOperator(A_l, input_shape=input_shape),
        b_u=b_u,
        b_l=b_l,
    )


def _dense_layout_barrier_step(
    state: AffineBackwardState,
    *,
    input_shape: Tuple[int, ...],
    dims_with_batch: Tuple[int, ...],
    device: torch.device,
    caller: str,
) -> AffineBackwardState:
    dims = crown_mod._normalize_batch_preserving_permute_dims(
        dims_with_batch,
        rank_with_batch=len(input_shape) + 1,
        caller=caller,
    )
    output_shape = crown_mod._permute_output_shape(input_shape=input_shape, dims_with_batch=dims)
    base = crown_mod._align_backward_state_input_shape(state, input_shape=output_shape)
    gather_index = crown_mod._make_permute_gather_index(
        input_shape=input_shape,
        dims_with_batch=dims,
        device=device,
    )
    scatter_index = torch.empty_like(gather_index)
    scatter_index[gather_index] = torch.arange(int(gather_index.numel()), device=device, dtype=torch.long)
    with linear_op_mod.operator_attribution_reason("dense_baseline_materialization"):
        A_u = base.A_u.to_dense().index_select(2, scatter_index)
        A_l = base.A_l.to_dense().index_select(2, scatter_index)
    return AffineBackwardState(
        A_u=DenseLinearOperator(A_u, input_shape=input_shape),
        A_l=DenseLinearOperator(A_l, input_shape=input_shape),
        b_u=base.b_u,
        b_l=base.b_l,
    )


@contextlib.contextmanager
def _patch_variant(variant: _Variant, counts: Optional[_Counters] = None):
    orig_split_dense = linear_op_mod._split_pos_neg_dense
    orig_relu_step = crown_mod._backprop_relu_step
    orig_permute_step = crown_mod._backprop_permute_step

    def wrapped_split_dense(op):
        if counts is not None:
            op_name = type(op).__name__
            counts.split_pos_neg_dense_total += 1
            counts.split_pos_neg_dense_by_op[op_name] = counts.split_pos_neg_dense_by_op.get(op_name, 0) + 1
        return orig_split_dense(op)

    def wrapped_relu_step(*args, **kwargs):
        if counts is not None:
            counts.relu_backward_calls += 1
        if variant == "dense_relu":
            if counts is not None:
                counts.dense_relu_barrier_calls += 1
            return _dense_relu_barrier_step(*args, **kwargs)
        return orig_relu_step(*args, **kwargs)

    def wrapped_permute_step(*args, **kwargs):
        if counts is not None:
            counts.permute_backward_calls += 1
        if variant == "dense_layout":
            if counts is not None:
                counts.dense_layout_barrier_calls += 1
            return _dense_layout_barrier_step(*args, **kwargs)
        return orig_permute_step(*args, **kwargs)

    if counts is not None:
        linear_op_mod._split_pos_neg_dense = wrapped_split_dense
        crown_mod._backprop_relu_step = wrapped_relu_step
        crown_mod._backprop_permute_step = wrapped_permute_step
    else:
        if variant == "dense_relu":
            crown_mod._backprop_relu_step = wrapped_relu_step
        elif variant == "dense_layout":
            crown_mod._backprop_permute_step = wrapped_permute_step

    try:
        yield
    finally:
        linear_op_mod._split_pos_neg_dense = orig_split_dense
        crown_mod._backprop_relu_step = orig_relu_step
        crown_mod._backprop_permute_step = orig_permute_step


def _run_variant_once(
    module: BFTaskModule,
    spec: InputSpec,
    *,
    variant: _Variant,
    final_policy: FinalConcretizationPolicy = "structured",
) -> IntervalState:
    with final_concretization_policy(final_policy), _patch_variant(variant):
        with torch.inference_mode():
            return run_crown_ibp_mlp(module, spec)


def _attribution_path_kind(variant: _Variant) -> str:
    return "structured" if variant == "structured" else "dense_baseline"


def _attribution_phase(variant: _Variant) -> str:
    return "structured_execution" if variant == "structured" else "benchmark_baseline"


def _collect_counts(
    module: BFTaskModule,
    spec: InputSpec,
    *,
    variant: _Variant,
    use_dense_cache: bool = True,
    final_policy: FinalConcretizationPolicy = "structured",
) -> Dict[str, Any]:
    counts = _Counters()
    with linear_op_mod.collect_operator_attribution(
        path_kind=_attribution_path_kind(variant),
        phase=_attribution_phase(variant),
    ) as trace:
        with final_concretization_policy(final_policy), linear_op_mod.operator_dense_cache(enabled=bool(use_dense_cache)):
            with _patch_variant(variant, counts=counts):
                with torch.inference_mode():
                    _ = run_crown_ibp_mlp(module, spec)
        counts.operator_attribution = trace.to_jsonable()
    return counts.to_jsonable()


def _make_relu_heavy_mlp_case(*, device: torch.device, dtype: torch.dtype, profile: str, seed: int) -> Tuple[BFTaskModule, InputSpec]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    if profile == "smoke":
        batch, in_dim, hidden_dims, out_dim, eps = 4, 32, (64, 64, 64), 16, 0.05
    elif profile == "small":
        batch, in_dim, hidden_dims, out_dim, eps = 8, 64, (128, 128, 128), 32, 0.04
    else:
        batch, in_dim, hidden_dims, out_dim, eps = 32, 256, (512, 512, 512, 512), 128, 0.03

    ops: List[TaskOp] = []
    params: Dict[str, torch.Tensor] = {}
    prev_name = "input"
    prev_dim = int(in_dim)
    for idx, hidden in enumerate(hidden_dims, start=1):
        w_name = f"W{idx}"
        b_name = f"b{idx}"
        h_name = f"h{idx}"
        r_name = f"r{idx}"
        params[w_name] = torch.randn(int(hidden), prev_dim, device=device, dtype=dtype)
        params[b_name] = torch.randn(int(hidden), device=device, dtype=dtype)
        ops.append(TaskOp(op_type="linear", name=f"linear{idx}", inputs=[prev_name, w_name, b_name], outputs=[h_name]))
        ops.append(TaskOp(op_type="relu", name=f"relu{idx}", inputs=[h_name], outputs=[r_name]))
        prev_name = r_name
        prev_dim = int(hidden)

    params["W_out"] = torch.randn(int(out_dim), prev_dim, device=device, dtype=dtype)
    params["b_out"] = torch.randn(int(out_dim), device=device, dtype=dtype)
    ops.append(TaskOp(op_type="linear", name="linear_out", inputs=[prev_name, "W_out", "b_out"], outputs=["out"]))
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=ops,
        input_values=["input"],
        output_values=["out"],
    )
    module = BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})
    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    return module, InputSpec.linf(value_name="input", center=x0, eps=eps)


def _make_residual_relu_case(*, device: torch.device, dtype: torch.dtype, profile: str, seed: int) -> Tuple[BFTaskModule, InputSpec]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    if profile == "smoke":
        batch, dim, hidden, out_dim, eps = 4, 48, 48, 16, 0.05
    elif profile == "small":
        batch, dim, hidden, out_dim, eps = 8, 96, 96, 32, 0.04
    else:
        batch, dim, hidden, out_dim, eps = 32, 256, 256, 128, 0.03

    params = {
        "W1": torch.randn(hidden, dim, device=device, dtype=dtype),
        "b1": torch.randn(hidden, device=device, dtype=dtype),
        "W2": torch.randn(dim, hidden, device=device, dtype=dtype),
        "b2": torch.randn(dim, device=device, dtype=dtype),
        "W3": torch.randn(out_dim, dim, device=device, dtype=dtype),
        "b3": torch.randn(out_dim, device=device, dtype=dtype),
    }
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["r1", "W2", "b2"], outputs=["h2"]),
            TaskOp(op_type="add", name="add0", inputs=["input", "h2"], outputs=["sum0"]),
            TaskOp(op_type="relu", name="relu2", inputs=["sum0"], outputs=["r2"]),
            TaskOp(op_type="linear", name="linear3", inputs=["r2", "W3", "b3"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    module = BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})
    x0 = torch.randn(batch, dim, device=device, dtype=dtype)
    return module, InputSpec.linf(value_name="input", center=x0, eps=eps)


def _make_concat_relu_case(*, device: torch.device, dtype: torch.dtype, profile: str, seed: int) -> Tuple[BFTaskModule, InputSpec]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    if profile == "smoke":
        batch, in_dim, branch_dim, mid_dim, out_dim, eps = 4, 32, 24, 32, 16, 0.05
    elif profile == "small":
        batch, in_dim, branch_dim, mid_dim, out_dim, eps = 8, 64, 48, 64, 32, 0.04
    else:
        batch, in_dim, branch_dim, mid_dim, out_dim, eps = 32, 256, 192, 256, 128, 0.03

    params = {
        "W1": torch.randn(branch_dim, in_dim, device=device, dtype=dtype),
        "b1": torch.randn(branch_dim, device=device, dtype=dtype),
        "W2": torch.randn(branch_dim, in_dim, device=device, dtype=dtype),
        "b2": torch.randn(branch_dim, device=device, dtype=dtype),
        "W3": torch.randn(mid_dim, branch_dim * 2, device=device, dtype=dtype),
        "b3": torch.randn(mid_dim, device=device, dtype=dtype),
        "W4": torch.randn(out_dim, mid_dim, device=device, dtype=dtype),
        "b4": torch.randn(out_dim, device=device, dtype=dtype),
    }
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["input", "W2", "b2"], outputs=["h2"]),
            TaskOp(op_type="relu", name="relu2", inputs=["h2"], outputs=["r2"]),
            TaskOp(op_type="concat", name="concat0", inputs=["r1", "r2"], outputs=["cat0"], attrs={"axis": 1}),
            TaskOp(op_type="linear", name="linear3", inputs=["cat0", "W3", "b3"], outputs=["h3"]),
            TaskOp(op_type="relu", name="relu3", inputs=["h3"], outputs=["r3"]),
            TaskOp(op_type="linear", name="linear4", inputs=["r3", "W4", "b4"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    module = BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})
    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    return module, InputSpec.linf(value_name="input", center=x0, eps=eps)


def _make_permute_reshape_case(*, device: torch.device, dtype: torch.dtype, profile: str, seed: int) -> Tuple[BFTaskModule, InputSpec]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    if profile == "smoke":
        batch, channels, height, width, out_dim, eps = 4, 2, 4, 4, 16, 0.05
    elif profile == "small":
        batch, channels, height, width, out_dim, eps = 8, 4, 8, 8, 64, 0.04
    else:
        batch, channels, height, width, out_dim, eps = 32, 8, 16, 16, 256, 0.03
    flat_dim = int(channels * height * width)
    params = {
        "W": torch.randn(out_dim, flat_dim, device=device, dtype=dtype),
        "b": torch.randn(out_dim, device=device, dtype=dtype),
    }
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="permute",
                name="perm0",
                inputs=["input"],
                outputs=["p0"],
                attrs={"dims": [0, 2, 3, 1], "layout_only": True},
            ),
            TaskOp(op_type="reshape", name="reshape0", inputs=["p0"], outputs=["flat"], attrs={"shape": [batch, -1]}),
            TaskOp(op_type="linear", name="linear0", inputs=["flat", "W", "b"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    module = BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})
    x0 = torch.randn(batch, channels, height, width, device=device, dtype=dtype)
    return module, InputSpec.linf(value_name="input", center=x0, eps=eps)


_WORKLOAD_BUILDERS: Dict[str, Tuple[_CompareTarget, Callable[..., Tuple[BFTaskModule, InputSpec]]]] = {
    "relu_heavy_mlp": ("relu_barrier", _make_relu_heavy_mlp_case),
    "residual_relu_mlp": ("relu_barrier", _make_residual_relu_case),
    "concat_relu_mlp": ("relu_barrier", _make_concat_relu_case),
    "permute_reshape_linear": ("layout_only", _make_permute_reshape_case),
}


def _parse_workloads(raw: str) -> List[str]:
    if raw.strip() == "all":
        return list(_WORKLOAD_BUILDERS)
    out: List[str] = []
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in _WORKLOAD_BUILDERS:
            raise ValueError(f"unknown workload: {name}")
        out.append(name)
    if not out:
        raise ValueError("empty --workloads")
    return out


def _build_case(
    workload: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    profile: str,
    seed: int,
) -> Tuple[_CompareTarget, BFTaskModule, InputSpec]:
    compare_target, builder = _WORKLOAD_BUILDERS[workload]
    module, spec = builder(device=device, dtype=dtype, profile=profile, seed=seed)
    stats = get_crown_ibp_mlp_stats(module)
    if not stats.supported:
        raise RuntimeError(f"shared CROWN does not support workload {workload}: {stats.reason}")
    return compare_target, module, spec


def _baseline_variant(compare_target: _CompareTarget) -> _Variant:
    if compare_target == "relu_barrier":
        return "dense_relu"
    return "dense_layout"


def _collect_row(
    workload: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    profile: str,
    seed: int,
    warmup: int,
    iters: int,
    timer: _TimerMode,
    torch_benchmark_min_run_time_s: float,
    final_policy_request: FinalConcretizationRequest,
) -> Row:
    compare_target, module, spec = _build_case(
        workload,
        device=device,
        dtype=dtype,
        profile=profile,
        seed=seed,
    )
    baseline_variant = _baseline_variant(compare_target)
    planner_decision = plan_phase7b_shared_crown(
        compare_target=compare_target,
        workload=workload,
        scale_id=profile,
        device=str(device),
        requested_final_concretization_policy=final_policy_request,
    )
    resolved_final_policy = planner_decision.final_concretization_policy
    structured_counts = _collect_counts(
        module,
        spec,
        variant="structured",
        use_dense_cache=planner_decision.use_dense_cache,
        final_policy=resolved_final_policy,
    )
    baseline_counts = _collect_counts(
        module,
        spec,
        variant=baseline_variant,
        use_dense_cache=planner_decision.use_dense_cache,
        final_policy=resolved_final_policy,
    )

    structured_ms = _time_variant(
        lambda: _run_variant_once(module, spec, variant="structured", final_policy=resolved_final_policy),
        warmup=warmup,
        iters=iters,
        timer=timer,
        torch_benchmark_min_run_time_s=float(torch_benchmark_min_run_time_s),
        device=device,
    )
    baseline_ms = _time_variant(
        lambda: _run_variant_once(module, spec, variant=baseline_variant, final_policy=resolved_final_policy),
        warmup=warmup,
        iters=iters,
        timer=timer,
        torch_benchmark_min_run_time_s=float(torch_benchmark_min_run_time_s),
        device=device,
    )
    speedup = float("inf") if structured_ms == 0.0 else float(baseline_ms / structured_ms)
    _eprint(
        f"{workload}: target={compare_target} structured_ms_p50={structured_ms:.3f} "
        f"baseline_ms_p50={baseline_ms:.3f} speedup={speedup:.2f}x"
    )
    return Row(
        workload=workload,
        compare_target=compare_target,
        structured_ms_p50=float(structured_ms),
        baseline_ms_p50=float(baseline_ms),
        speedup=float(speedup),
        counts_structured=structured_counts,
        counts_baseline=baseline_counts,
        planner_decision=planner_decision.to_jsonable(),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7A PR-11: benchmark structured shared CROWN path attribution."
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--profile", type=str, default="bench", choices=["smoke", "small", "bench"])
    parser.add_argument("--workloads", type=str, default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timer", type=str, default="perf_counter", choices=["perf_counter", "torch_benchmark"])
    parser.add_argument("--torch-benchmark-min-run-time-s", type=float, default=0.2)
    parser.add_argument("--final-concretization-policy", type=str, default="structured", choices=["structured", "dense_barrier", "auto"])
    parser.add_argument("--allow-mps-fallback", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = _make_device(str(args.device), dtype_name=str(args.dtype), allow_mps_fallback=bool(args.allow_mps_fallback))
    workloads = _parse_workloads(args.workloads)
    timer: _TimerMode = args.timer

    rows: List[Row] = []
    for idx, workload in enumerate(workloads):
        rows.append(
            _collect_row(
                workload,
                device=device,
                dtype=dtype,
                profile=str(args.profile),
                seed=int(args.seed) + idx,
                warmup=int(args.warmup),
                iters=int(args.iters),
                timer=timer,
                torch_benchmark_min_run_time_s=float(args.torch_benchmark_min_run_time_s),
                final_policy_request=args.final_concretization_policy,
            )
        )

    payload = {
        "meta": {
            "schema_version": "phase7a_shared_crown_path_attribution.v1",
            "script": "bench_phase7a_shared_crown_path_attribution",
            "git_sha": _git_sha(),
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": _device_name(device),
            "device_meta": _device_meta(device, allow_mps_fallback=bool(args.allow_mps_fallback)),
            "dtype": str(dtype).replace("torch.", ""),
            "profile": str(args.profile),
            "workloads": workloads,
            "timer": timer,
            "final_concretization_policy": str(args.final_concretization_policy),
            "capability_table": phase7a_capability_table_jsonable(),
            "cost_model_rules": phase7b_cost_model_rules_jsonable(),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "seed": int(args.seed),
            "torch_benchmark_min_run_time_s": float(args.torch_benchmark_min_run_time_s),
        },
        "rows": [asdict(row) for row in rows],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
