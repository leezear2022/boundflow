from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from ..domains.interval import IntervalState
from ..ir.task import BFTaskModule
from .alpha_beta_crown import BetaState, check_first_layer_infeasible_split, run_alpha_beta_crown_mlp
from .alpha_crown import AlphaState, SpecReduce, run_alpha_crown_mlp
from .crown_ibp import _forward_ibp_trace_mlp
from .perturbation import LpBallPerturbation
from .task_executor import InputSpecLike, _normalize_input_spec

BabStatus = Literal["proven", "unsafe", "unknown"]


@dataclass(frozen=True)
class ReluSplitState:
    """
    Split constraints for chain-structured ReLU nodes.

    Mapping: relu_input_name -> split tensor int{-1,0,+1} with logical shape [*S].
    - +1: active (x >= 0)
    - -1: inactive (x <= 0)
    -  0: unsplit
    """

    by_relu_input: Dict[str, torch.Tensor]

    def get(self, relu_input: str) -> Optional[torch.Tensor]:
        return self.by_relu_input.get(relu_input)

    def with_split(self, *, relu_input: str, neuron_idx: int, split_value: int) -> "ReluSplitState":
        if split_value not in (-1, 0, 1):
            raise ValueError(f"split_value must be in (-1,0,1), got {split_value}")
        cur = self.by_relu_input.get(relu_input)
        if cur is None:
            raise KeyError(f"unknown relu_input in split state: {relu_input}")
        cur_flat = cur.reshape(-1)
        if int(neuron_idx) < 0 or int(neuron_idx) >= int(cur_flat.numel()):
            raise IndexError(
                f"neuron_idx out of range for {relu_input}: idx={int(neuron_idx)} numel={int(cur_flat.numel())}"
            )
        cur_i = int(cur_flat[int(neuron_idx)].item())
        if cur_i != 0 and cur_i != int(split_value):
            raise ValueError(f"conflicting split for {relu_input}[{neuron_idx}]: existing={cur_i} new={split_value}")
        new_by = dict(self.by_relu_input)
        new_t = cur.clone()
        new_t.reshape(-1)[int(neuron_idx)] = int(split_value)
        new_by[relu_input] = new_t
        return ReluSplitState(by_relu_input=new_by)

    @staticmethod
    def empty(
        module: BFTaskModule,
        *,
        device: torch.device,
        input_spec: Optional[InputSpecLike] = None,
    ) -> "ReluSplitState":
        module.validate()
        if input_spec is not None:
            spec = _normalize_input_spec(input_spec)
            _interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
            return ReluSplitState(
                by_relu_input={
                    name: torch.zeros(tuple(pre.lower.shape[1:]), device=device, dtype=torch.int8)
                    for name, pre in relu_pre.items()
                }
            )
        task = module.get_entry_task()
        raw_params = module.bindings.get("params", {})
        params: Dict[str, object] = dict(raw_params) if isinstance(raw_params, dict) else {}
        by_relu: Dict[str, torch.Tensor] = {}
        for op in task.ops:
            if op.op_type != "relu":
                continue
            x_name = op.inputs[0]
            if x_name in by_relu:
                continue
            # Find producer linear weight to infer hidden dim.
            prod = None
            for prev in task.ops:
                if x_name in prev.outputs:
                    prod = prev
                    break
            if prod is None or prod.op_type != "linear":
                raise NotImplementedError(f"ReluSplitState only supports relu inputs produced by linear, got {prod}")
            w_name = prod.inputs[1]
            w = params.get(w_name)
            if w is None:
                raise KeyError(f"missing weight param tensor: {w_name}")
            w_t = w if torch.is_tensor(w) else torch.as_tensor(w)
            if w_t.dim() != 2:
                raise NotImplementedError(
                    "ReluSplitState.empty needs input_spec for non-rank-2 ReLU producers; "
                    f"got producer {prod.op_type} weight shape {tuple(w_t.shape)}"
                )
            h = int(w_t.shape[0])
            by_relu[x_name] = torch.zeros(h, device=device, dtype=torch.int8)
        return ReluSplitState(by_relu_input=by_relu)


@dataclass(frozen=True)
class BabConfig:
    max_nodes: int = 10_000
    oracle: Literal["alpha", "alpha_beta"] = "alpha"
    use_1d_linf_input_restriction_patch: bool = False
    node_batch_size: int = 1
    use_branch_hint: bool = True
    enable_node_eval_cache: bool = True
    enable_batch_infeasible_prune: bool = False
    alpha_steps: int = 20
    alpha_lr: float = 0.2
    alpha_init: float = 0.5
    beta_init: float = 0.0
    objective: Literal["lower", "upper", "gap", "both"] = "lower"
    spec_reduce: SpecReduce = "mean"
    soft_tau: float = 1.0
    lb_weight: float = 1.0
    ub_weight: float = 1.0
    threshold: float = 0.0
    tol: float = 1e-6


@dataclass(frozen=True)
class BabResult:
    status: BabStatus
    nodes_visited: int
    nodes_evaluated: int
    nodes_expanded: int
    max_queue: int
    batch_rounds: int
    avg_batch_fill_rate: float
    best_lower: float
    best_upper: float
    per_example: Tuple["BabPerExampleResult", ...] = ()


@dataclass(frozen=True)
class BabPerExampleResult:
    example_idx: int
    status: BabStatus
    nodes_visited: int
    nodes_evaluated: int
    nodes_expanded: int
    best_lower: float
    best_upper: float


@dataclass(order=True)
class _QueueItem:
    # heap key: smallest lower bound first (hardest node).
    priority: float
    node_id: int
    example_idx: int
    split_state: ReluSplitState
    warm_start: Optional[AlphaState] = None
    warm_start_beta: Optional[BetaState] = None


@dataclass(frozen=True)
class NodeEvalCacheKey:
    base: str
    split: str


@dataclass(frozen=True)
class NodeEvalCacheValue:
    bounds: IntervalState
    best_alpha: AlphaState
    best_beta: BetaState
    stats: Any
    branch: Optional[Tuple[str, int]]


def _hash_tensor_cpu_bytes(t: torch.Tensor) -> str:
    tt = t.detach()
    if tt.device.type != "cpu":
        tt = tt.cpu()
    tt = tt.contiguous()
    h = hashlib.sha256()
    h.update(str(tt.dtype).encode("utf-8"))
    h.update(str(tuple(tt.shape)).encode("utf-8"))
    h.update(memoryview(tt.numpy()).tobytes())
    return h.hexdigest()


def _hash_split_state(split_state: ReluSplitState) -> str:
    h = hashlib.sha256()
    for name in sorted(split_state.by_relu_input.keys()):
        h.update(name.encode("utf-8"))
        t = split_state.by_relu_input[name]
        h.update(_hash_tensor_cpu_bytes(t).encode("utf-8"))
    return h.hexdigest()


class NodeEvalCache:
    """
    Phase 6G PR-3A: in-memory node evaluation cache.

    Scope: single-process / single-run BaB.
    """

    def __init__(
        self,
        *,
        module: BFTaskModule,
        input_spec: InputSpecLike,
        linear_spec_C: Optional[torch.Tensor],
        cfg: BabConfig,
    ) -> None:
        spec = _normalize_input_spec(input_spec)
        c_hash = "none" if linear_spec_C is None else _hash_tensor_cpu_bytes(torch.as_tensor(linear_spec_C))
        spec_hash = _hash_tensor_cpu_bytes(spec.center)
        pert = spec.perturbation
        pert_hash = f"{type(pert).__name__}:{getattr(pert, '_normalize_p', lambda: 'unknown')()}:{getattr(pert, 'eps', 'unknown')}"
        cfg_key = (
            f"oracle={cfg.oracle}|steps={int(cfg.alpha_steps)}|lr={float(cfg.alpha_lr)}|"
            f"alpha_init={float(cfg.alpha_init)}|beta_init={float(cfg.beta_init)}|"
            f"objective={cfg.objective}|spec_reduce={cfg.spec_reduce}|soft_tau={float(cfg.soft_tau)}|"
            f"lb_w={float(cfg.lb_weight)}|ub_w={float(cfg.ub_weight)}"
        )
        self._base = f"module={id(module)}|spec={spec.value_name}:{spec_hash}:{pert_hash}|C={c_hash}|{cfg_key}"
        self._store: Dict[NodeEvalCacheKey, NodeEvalCacheValue] = {}
        self.hits = 0
        self.misses = 0

    def get(self, *, split_state: ReluSplitState) -> Optional[NodeEvalCacheValue]:
        key = NodeEvalCacheKey(base=self._base, split=_hash_split_state(split_state))
        v = self._store.get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, *, split_state: ReluSplitState, value: NodeEvalCacheValue) -> None:
        key = NodeEvalCacheKey(base=self._base, split=_hash_split_state(split_state))
        self._store[key] = value


def _slice_input_spec(input_spec: InputSpecLike, *, example_idx: int) -> InputSpecLike:
    spec = _normalize_input_spec(input_spec)
    center = spec.center[int(example_idx) : int(example_idx) + 1].clone()
    return type(spec)(value_name=spec.value_name, center=center, perturbation=spec.perturbation)


def _gather_input_spec(input_spec: InputSpecLike, *, example_indices: List[int]) -> InputSpecLike:
    spec = _normalize_input_spec(input_spec)
    center = torch.cat([spec.center[int(i) : int(i) + 1] for i in example_indices], dim=0).clone()
    return type(spec)(value_name=spec.value_name, center=center, perturbation=spec.perturbation)


def _slice_linear_spec_C(linear_spec_C: Optional[torch.Tensor], *, example_idx: int, device: torch.device) -> Optional[torch.Tensor]:
    if linear_spec_C is None:
        return None
    C = linear_spec_C if torch.is_tensor(linear_spec_C) else torch.as_tensor(linear_spec_C, device=device)
    C = C.to(device=device)
    if C.dim() == 2:
        return C
    if C.dim() == 3:
        return C[int(example_idx) : int(example_idx) + 1].clone()
    raise ValueError(f"linear_spec_C expects rank-2/3, got {tuple(C.shape)}")


def _gather_linear_spec_C(
    linear_spec_C: Optional[torch.Tensor], *, example_indices: List[int], device: torch.device
) -> Optional[torch.Tensor]:
    if linear_spec_C is None:
        return None
    C = linear_spec_C if torch.is_tensor(linear_spec_C) else torch.as_tensor(linear_spec_C, device=device)
    C = C.to(device=device)
    if C.dim() == 2:
        return C.unsqueeze(0).expand(len(example_indices), int(C.shape[0]), int(C.shape[1])).clone()
    if C.dim() == 3:
        return torch.cat([C[int(i) : int(i) + 1] for i in example_indices], dim=0).clone()
    raise ValueError(f"linear_spec_C expects rank-2/3, got {tuple(C.shape)}")


def _aggregate_bab_status(statuses: List[BabStatus]) -> BabStatus:
    if any(s == "unsafe" for s in statuses):
        return "unsafe"
    if all(s == "proven" for s in statuses):
        return "proven"
    return "unknown"


def prune_infeasible_first_layer_items(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    items: List[Tuple[int, _QueueItem]],
    cache_by_example: Optional[Dict[int, NodeEvalCache]],
    cfg: BabConfig,
) -> Tuple[List[Tuple[int, _QueueItem]], List[int]]:
    """
    Phase 6G PR-3C (minimal): per-node infeasible pruning for first-layer split constraints.

    Returns (kept_items, pruned_indices).
    """
    if not bool(cfg.enable_batch_infeasible_prune) or cfg.oracle != "alpha_beta":
        return items, []
    pruned: List[int] = []
    kept: List[Tuple[int, _QueueItem]] = []
    spec = _normalize_input_spec(input_spec)
    device = spec.center.device
    dtype = spec.center.dtype

    for orig_i, it in items:
        cache = None if cache_by_example is None else cache_by_example.get(int(it.example_idx))
        if cache is not None:
            v = cache.get(split_state=it.split_state)
            if v is not None and getattr(v.stats, "feasibility", None) == "infeasible":
                pruned.append(orig_i)
                continue
        node_spec = _slice_input_spec(spec, example_idx=int(it.example_idx))
        st = check_first_layer_infeasible_split(module, node_spec, relu_split_state=it.split_state.by_relu_input)
        if st.feasibility == "infeasible":
            pruned.append(orig_i)
            if cache is not None:
                zeros = torch.zeros((1, 1), device=device, dtype=dtype)
                cache.put(
                    split_state=it.split_state,
                    value=NodeEvalCacheValue(
                        bounds=IntervalState(lower=zeros, upper=zeros),
                        best_alpha=AlphaState(alpha_by_relu_input={}),
                        best_beta=BetaState(beta_by_relu_input={}),
                        stats=st,
                        branch=None,
                    ),
                )
            continue
        kept.append((orig_i, it))
    return kept, pruned


def eval_bab_alpha_beta_node(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    linear_spec_C: Optional[torch.Tensor],
    split_state: ReluSplitState,
    warm_start_alpha: Optional[AlphaState],
    warm_start_beta: Optional[BetaState],
    cfg: BabConfig,
    cache: Optional[NodeEvalCache],
) -> Tuple[IntervalState, AlphaState, BetaState, Any, Optional[Tuple[str, int]], bool]:
    if cache is not None:
        cached = cache.get(split_state=split_state)
        if cached is not None:
            return cached.bounds, cached.best_alpha, cached.best_beta, cached.stats, cached.branch, True
    bounds, best_alpha, best_beta, stats = run_alpha_beta_crown_mlp(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_split_state=split_state.by_relu_input,
        steps=int(cfg.alpha_steps),
        lr=float(cfg.alpha_lr),
        alpha_init=float(cfg.alpha_init),
        beta_init=float(cfg.beta_init),
        warm_start_alpha=warm_start_alpha,
        warm_start_beta=warm_start_beta,
        objective=cfg.objective,
        spec_reduce=cfg.spec_reduce,
        soft_tau=float(cfg.soft_tau),
        lb_weight=float(cfg.lb_weight),
        ub_weight=float(cfg.ub_weight),
    )
    branch: Optional[Tuple[str, int]] = None
    if hasattr(stats, "branch_choices") and stats.branch_choices:
        if len(stats.branch_choices) == 1:
            branch = stats.branch_choices[0]
    if cache is not None:
        cache.put(
            split_state=split_state,
            value=NodeEvalCacheValue(
                bounds=IntervalState(lower=bounds.lower.detach().clone(), upper=bounds.upper.detach().clone()),
                best_alpha=best_alpha.detach_clone(),
                best_beta=best_beta.detach_clone(),
                stats=stats,
                branch=branch,
            ),
        )
    return bounds, best_alpha, best_beta, stats, branch, False


def _pick_branch(
    module: BFTaskModule, input_spec: InputSpecLike, *, split_state: ReluSplitState
) -> Optional[Tuple[str, int]]:
    """
    Pick a ReLU neuron to split based on the widest ambiguous pre-activation interval.
    Returns (relu_input_name, neuron_idx) or None if no ambiguous neuron exists.
    """
    input_spec_n = _normalize_input_spec(input_spec)
    if int(input_spec_n.center.shape[0]) != 1:
        raise ValueError(
            f"_pick_branch expects a single-example InputSpec, got batch={int(input_spec_n.center.shape[0])}"
        )
    relu_split = {k: v for k, v in split_state.by_relu_input.items() if v is not None}
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, input_spec_n, relu_split_state=relu_split)
    _ = interval_env
    best = None
    best_gap = None
    for name, pre in relu_pre.items():
        l = pre.lower.reshape(int(pre.lower.shape[0]), -1)
        u = pre.upper.reshape(int(pre.upper.shape[0]), -1)
        amb = (l < 0) & (u > 0)
        if not amb.any():
            continue
        gap = (u - l).clamp_min(0.0)
        gap = torch.where(amb, gap, torch.full_like(gap, float("-inf")))
        gap = gap[0]
        idx = int(torch.argmax(gap).item())
        g = float(gap[idx].item())
        if best_gap is None or g > best_gap:
            best_gap = g
            best = (name, idx)
    return best


def _restrict_input_spec_linf_1d_for_first_layer_splits(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    split_state: ReluSplitState,
) -> Optional[InputSpecLike]:
    """
    Best-effort: restrict a 1D Linf input spec by intersecting first-layer ReLU split constraints.

    This handles the minimal "toy complete" case where each split constraint is of the form:
      (w_i * x + b_i) >= 0  or <= 0, with x being the original input scalar.

    Returns a new Linf InputSpec representing the restricted interval, or None if infeasible/unhandled.
    """
    spec = _normalize_input_spec(input_spec)
    if spec.center.dim() != 2 or int(spec.center.shape[1]) != 1:
        return spec
    if not isinstance(spec.perturbation, LpBallPerturbation):
        return spec
    if spec.perturbation._normalize_p() != "inf":
        return spec

    module.validate()
    task = module.get_entry_task()
    raw_params = module.bindings.get("params", {})
    params: Dict[str, object] = dict(raw_params) if isinstance(raw_params, dict) else {}

    if int(spec.center.shape[0]) != 1:
        return spec
    x0 = float(spec.center[0, 0].item())
    eps = float(spec.perturbation.eps)
    lo = x0 - eps
    hi = x0 + eps

    # Only handle splits for ReLU inputs produced directly from the original input by one linear layer.
    for op in task.ops:
        if op.op_type != "relu":
            continue
        x_name = op.inputs[0]
        split = split_state.get(x_name)
        if split is None:
            continue

        prod = None
        for prev in task.ops:
            if x_name in prev.outputs:
                prod = prev
                break
        if prod is None or prod.op_type != "linear":
            continue
        if not prod.inputs or prod.inputs[0] != spec.value_name:
            continue

        w_name = prod.inputs[1]
        b_name = prod.inputs[2] if len(prod.inputs) == 3 else None
        w = params.get(w_name)
        if w is None:
            continue
        w_t = w if torch.is_tensor(w) else torch.as_tensor(w)
        if w_t.dim() != 2 or int(w_t.shape[1]) != 1:
            continue
        b_t = None
        if b_name is not None:
            b = params.get(b_name)
            if b is None:
                continue
            b_t = b if torch.is_tensor(b) else torch.as_tensor(b)
            if b_t.dim() != 1 or int(b_t.shape[0]) != int(w_t.shape[0]):
                continue
        else:
            b_t = torch.zeros(int(w_t.shape[0]), dtype=w_t.dtype)

        for i in range(int(split.shape[0])):
            s = int(split[i].item())
            if s == 0:
                continue
            wi = float(w_t[i, 0].item())
            bi = float(b_t[i].item()) if b_t is not None else 0.0
            if wi == 0.0:
                if (s > 0 and bi < 0.0) or (s < 0 and bi > 0.0):
                    return None
                continue
            t = -bi / wi
            if s > 0:
                # wi*x + bi >= 0
                if wi > 0:
                    lo = max(lo, t)
                else:
                    hi = min(hi, t)
            else:
                # wi*x + bi <= 0
                if wi > 0:
                    hi = min(hi, t)
                else:
                    lo = max(lo, t)
            if lo > hi:
                return None

    center = torch.tensor([[0.5 * (lo + hi)]], device=spec.center.device, dtype=spec.center.dtype)
    new_eps = 0.5 * (hi - lo)
    return type(spec).linf(value_name=spec.value_name, center=center, eps=float(new_eps))


def solve_bab_mlp(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    linear_spec_C: Optional[torch.Tensor] = None,
    config: Optional[BabConfig] = None,
) -> BabResult:
    """
    Correctness-first BaB for the currently exposed chain-structured solver subset.

    Supported subset:
    - MLP: linear / relu / linear
    - CNN (alpha-beta oracle only): conv2d / relu / ... / flatten(start_dim=1,end_dim=-1) / linear

    Host-side semantics keep one independent search tree per input example. Node batching may
    mix nodes from different examples in a single alpha-beta oracle call.
    """
    cfg = config or BabConfig()
    if int(cfg.max_nodes) < 1:
        raise ValueError(f"max_nodes must be >=1, got {cfg.max_nodes}")
    input_spec_n = _normalize_input_spec(input_spec)
    device = input_spec_n.center.device
    module.validate()
    task = module.get_entry_task()
    has_conv = any(op.op_type in {"conv2d", "flatten"} for op in task.ops)
    if has_conv and cfg.oracle != "alpha_beta":
        raise NotImplementedError("alpha-only BaB does not yet support conv graphs")

    num_examples = int(input_spec_n.center.shape[0])
    heap: List[_QueueItem] = []
    live_nodes = [0 for _ in range(num_examples)]
    visited_by_example = [0 for _ in range(num_examples)]
    evaluated_by_example = [0 for _ in range(num_examples)]
    expanded_by_example = [0 for _ in range(num_examples)]
    best_lower_by_example = [float("-inf") for _ in range(num_examples)]
    best_upper_by_example = [float("inf") for _ in range(num_examples)]
    status_by_example: List[Optional[BabStatus]] = [None for _ in range(num_examples)]
    node_id = 0
    for example_idx in range(num_examples):
        root_spec = _slice_input_spec(input_spec_n, example_idx=example_idx)
        root_split = ReluSplitState.empty(module, device=device, input_spec=root_spec)
        heapq.heappush(
            heap,
            _QueueItem(
                priority=float("-inf"),
                node_id=node_id,
                example_idx=example_idx,
                split_state=root_split,
                warm_start=None,
                warm_start_beta=None,
            ),
        )
        node_id += 1
        live_nodes[example_idx] = 1

    batch_rounds = 0
    batch_popped_total = 0
    max_q = len(heap)

    if int(cfg.node_batch_size) < 1:
        raise ValueError(f"node_batch_size must be >=1, got {cfg.node_batch_size}")

    node_cache_by_example: Dict[int, NodeEvalCache] = {}
    if (
        cfg.oracle == "alpha_beta"
        and bool(cfg.enable_node_eval_cache)
        and not cfg.use_1d_linf_input_restriction_patch
    ):
        for example_idx in range(num_examples):
            node_cache_by_example[example_idx] = NodeEvalCache(
                module=module,
                input_spec=_slice_input_spec(input_spec_n, example_idx=example_idx),
                linear_spec_C=_slice_linear_spec_C(linear_spec_C, example_idx=example_idx, device=device),
                cfg=cfg,
            )

    def _mark_proven_if_finished(example_idx: int) -> None:
        if status_by_example[example_idx] is None and live_nodes[example_idx] == 0:
            status_by_example[example_idx] = "proven"

    def _decrement_live_node(example_idx: int) -> None:
        live_nodes[example_idx] = max(0, int(live_nodes[example_idx]) - 1)

    def _process_evaluated_node(
        item: _QueueItem,
        *,
        bounds: IntervalState,
        best_alpha: AlphaState,
        best_beta: Optional[BetaState],
        branch_hint: Optional[Tuple[str, int]],
    ) -> None:
        nonlocal node_id, max_q
        example_idx = int(item.example_idx)
        evaluated_by_example[example_idx] += 1
        node_lower = float(bounds.lower.min().item())
        node_upper = float(bounds.upper.max().item())
        best_lower_by_example[example_idx] = max(best_lower_by_example[example_idx], node_lower)
        best_upper_by_example[example_idx] = min(best_upper_by_example[example_idx], node_upper)

        _decrement_live_node(example_idx)
        if status_by_example[example_idx] is not None:
            return
        if node_lower >= float(cfg.threshold) - float(cfg.tol):
            _mark_proven_if_finished(example_idx)
            return

        branch = branch_hint if (cfg.oracle == "alpha_beta" and bool(cfg.use_branch_hint)) else None
        if branch is None:
            node_spec = _slice_input_spec(input_spec_n, example_idx=example_idx)
            branch = _pick_branch(module, node_spec, split_state=item.split_state)
        if branch is None:
            status_by_example[example_idx] = "unsafe"
            return
        if visited_by_example[example_idx] >= int(cfg.max_nodes):
            status_by_example[example_idx] = "unknown"
            return

        relu_input, idx = branch
        left = item.split_state.with_split(relu_input=relu_input, neuron_idx=idx, split_value=-1)
        right = item.split_state.with_split(relu_input=relu_input, neuron_idx=idx, split_value=+1)
        beta_warm = None if best_beta is None else best_beta.detach_clone()
        heapq.heappush(
            heap,
            _QueueItem(
                priority=node_lower,
                node_id=node_id,
                example_idx=example_idx,
                split_state=left,
                warm_start=best_alpha.detach_clone(),
                warm_start_beta=beta_warm,
            ),
        )
        node_id += 1
        heapq.heappush(
            heap,
            _QueueItem(
                priority=node_lower,
                node_id=node_id,
                example_idx=example_idx,
                split_state=right,
                warm_start=best_alpha.detach_clone(),
                warm_start_beta=beta_warm,
            ),
        )
        node_id += 1
        live_nodes[example_idx] += 2
        expanded_by_example[example_idx] += 1
        max_q = max(max_q, len(heap))

    while heap and any(status is None for status in status_by_example):
        do_batch = cfg.oracle == "alpha_beta" and int(cfg.node_batch_size) > 1 and not cfg.use_1d_linf_input_restriction_patch
        items: List[_QueueItem] = []
        want = int(cfg.node_batch_size) if do_batch else 1
        while len(items) < want and heap:
            item = heapq.heappop(heap)
            example_idx = int(item.example_idx)
            if status_by_example[example_idx] is not None:
                continue
            if visited_by_example[example_idx] >= int(cfg.max_nodes):
                status_by_example[example_idx] = "unknown"
                live_nodes[example_idx] = 0
                continue
            visited_by_example[example_idx] += 1
            items.append(item)
        if not items:
            break
        if do_batch:
            batch_rounds += 1
            batch_popped_total += len(items)

        if len(items) == 1:
            item = items[0]
            example_idx = int(item.example_idx)
            node_spec = _slice_input_spec(input_spec_n, example_idx=example_idx)
            restricted_spec = node_spec
            if cfg.use_1d_linf_input_restriction_patch:
                restricted_spec = _restrict_input_spec_linf_1d_for_first_layer_splits(
                    module, node_spec, split_state=item.split_state
                )
                if restricted_spec is None:
                    _decrement_live_node(example_idx)
                    _mark_proven_if_finished(example_idx)
                    continue
            try:
                if cfg.oracle == "alpha_beta":
                    bounds, best_alpha, best_beta, stats, branch_hint, _hit = eval_bab_alpha_beta_node(
                        module,
                        restricted_spec,
                        linear_spec_C=_slice_linear_spec_C(linear_spec_C, example_idx=example_idx, device=device),
                        split_state=item.split_state,
                        warm_start_alpha=item.warm_start,
                        warm_start_beta=item.warm_start_beta,
                        cfg=cfg,
                        cache=node_cache_by_example.get(example_idx),
                    )
                    if getattr(stats, "feasibility", None) == "infeasible":
                        _decrement_live_node(example_idx)
                        _mark_proven_if_finished(example_idx)
                        continue
                    best_beta = best_beta.detach_clone()
                else:
                    bounds, best_alpha, _stats = run_alpha_crown_mlp(
                        module,
                        restricted_spec,
                        linear_spec_C=_slice_linear_spec_C(linear_spec_C, example_idx=example_idx, device=device),
                        steps=int(cfg.alpha_steps),
                        lr=float(cfg.alpha_lr),
                        alpha_init=float(cfg.alpha_init),
                        objective=cfg.objective,
                        spec_reduce=cfg.spec_reduce,
                        soft_tau=float(cfg.soft_tau),
                        lb_weight=float(cfg.lb_weight),
                        ub_weight=float(cfg.ub_weight),
                        warm_start=item.warm_start,
                        relu_split_state=item.split_state.by_relu_input,
                    )
                    best_beta = None
                    branch_hint = None
            except ValueError:
                _decrement_live_node(example_idx)
                _mark_proven_if_finished(example_idx)
                continue

            _process_evaluated_node(
                item,
                bounds=bounds,
                best_alpha=best_alpha,
                best_beta=best_beta,
                branch_hint=branch_hint,
            )
            continue

        cached_results: Dict[int, Tuple[IntervalState, AlphaState, BetaState, Optional[Tuple[str, int]]]] = {}
        uncached: List[Tuple[int, _QueueItem]] = []
        pruned_indices: List[int] = []
        for i, item in enumerate(items):
            cache = node_cache_by_example.get(int(item.example_idx))
            if cache is None:
                uncached.append((i, item))
                continue
            value = cache.get(split_state=item.split_state)
            if value is None:
                uncached.append((i, item))
                continue
            if getattr(value.stats, "feasibility", None) == "infeasible":
                pruned_indices.append(i)
                continue
            cached_results[i] = (value.bounds, value.best_alpha, value.best_beta, value.branch)

        uncached, pruned_new = prune_infeasible_first_layer_items(
            module,
            input_spec_n,
            items=uncached,
            cache_by_example=node_cache_by_example if node_cache_by_example else None,
            cfg=cfg,
        )
        pruned_indices.extend(pruned_new)

        for i in pruned_indices:
            _decrement_live_node(int(items[i].example_idx))
            _mark_proven_if_finished(int(items[i].example_idx))

        batch_results: Dict[int, Tuple[IntervalState, AlphaState, BetaState, Optional[Tuple[str, int]]]] = {}
        if uncached:
            uncached_example_indices = [int(item.example_idx) for _, item in uncached]
            batch_spec = _gather_input_spec(input_spec_n, example_indices=uncached_example_indices)
            C_b = _gather_linear_spec_C(linear_spec_C, example_indices=uncached_example_indices, device=device)

            relu_split_b: Dict[str, torch.Tensor] = {}
            warm_alpha_b: Dict[str, torch.Tensor] = {}
            warm_beta_b: Dict[str, torch.Tensor] = {}
            for relu_input, base_split in uncached[0][1].split_state.by_relu_input.items():
                relu_split_b[relu_input] = torch.stack(
                    [item.split_state.by_relu_input[relu_input] for _, item in uncached], dim=0
                )
                alpha_rows: List[torch.Tensor] = []
                beta_rows: List[torch.Tensor] = []
                for _, item in uncached:
                    if item.warm_start is not None and relu_input in item.warm_start.alpha_by_relu_input:
                        a = item.warm_start.alpha_by_relu_input[relu_input]
                        a_t = a if torch.is_tensor(a) else torch.as_tensor(a, device=device)
                        alpha_rows.append(a_t.to(device=device, dtype=input_spec_n.center.dtype))
                    else:
                        alpha_rows.append(
                            torch.full(
                                tuple(base_split.shape),
                                float(cfg.alpha_init),
                                device=device,
                                dtype=input_spec_n.center.dtype,
                            )
                        )
                    if item.warm_start_beta is not None and relu_input in item.warm_start_beta.beta_by_relu_input:
                        b = item.warm_start_beta.beta_by_relu_input[relu_input]
                        b_t = b if torch.is_tensor(b) else torch.as_tensor(b, device=device)
                        beta_rows.append(b_t.to(device=device, dtype=input_spec_n.center.dtype))
                    else:
                        beta_rows.append(
                            torch.full(
                                tuple(base_split.shape),
                                float(cfg.beta_init),
                                device=device,
                                dtype=input_spec_n.center.dtype,
                            )
                        )
                warm_alpha_b[relu_input] = torch.stack(alpha_rows, dim=0)
                warm_beta_b[relu_input] = torch.stack(beta_rows, dim=0)

            try:
                bounds_b, alpha_b, beta_b, stats_var = run_alpha_beta_crown_mlp(
                    module,
                    batch_spec,
                    linear_spec_C=C_b,
                    relu_split_state=relu_split_b,
                    steps=int(cfg.alpha_steps),
                    lr=float(cfg.alpha_lr),
                    alpha_init=float(cfg.alpha_init),
                    beta_init=float(cfg.beta_init),
                    warm_start_alpha=AlphaState(alpha_by_relu_input=warm_alpha_b),
                    warm_start_beta=BetaState(beta_by_relu_input=warm_beta_b),
                    objective=cfg.objective,
                    spec_reduce=cfg.spec_reduce,
                    soft_tau=float(cfg.soft_tau),
                    lb_weight=float(cfg.lb_weight),
                    ub_weight=float(cfg.ub_weight),
                    per_batch_params=True,
                )
                if getattr(stats_var, "feasibility", None) == "infeasible":
                    for _, item in uncached:
                        _decrement_live_node(int(item.example_idx))
                        _mark_proven_if_finished(int(item.example_idx))
                else:
                    for jj, (orig_i, item) in enumerate(uncached):
                        node_bounds = IntervalState(
                            lower=bounds_b.lower[jj : jj + 1].detach().clone(),
                            upper=bounds_b.upper[jj : jj + 1].detach().clone(),
                        )
                        alpha_i = AlphaState({k: v[jj].detach().clone() for k, v in alpha_b.alpha_by_relu_input.items()})
                        beta_i = BetaState({k: v[jj].detach().clone() for k, v in beta_b.beta_by_relu_input.items()})
                        branch = None
                        if hasattr(stats_var, "branch_choices") and stats_var.branch_choices:
                            try:
                                branch = stats_var.branch_choices[jj]
                            except Exception:
                                branch = None
                        batch_results[orig_i] = (node_bounds, alpha_i, beta_i, branch)
                        cache = node_cache_by_example.get(int(item.example_idx))
                        if cache is not None:
                            cache.put(
                                split_state=item.split_state,
                                value=NodeEvalCacheValue(
                                    bounds=node_bounds,
                                    best_alpha=alpha_i.detach_clone(),
                                    best_beta=beta_i.detach_clone(),
                                    stats=stats_var,
                                    branch=branch,
                                ),
                            )
            except ValueError:
                for orig_i, item in uncached:
                    example_idx = int(item.example_idx)
                    try:
                        bounds, best_alpha, best_beta, stats, branch, _hit = eval_bab_alpha_beta_node(
                            module,
                            _slice_input_spec(input_spec_n, example_idx=example_idx),
                            linear_spec_C=_slice_linear_spec_C(linear_spec_C, example_idx=example_idx, device=device),
                            split_state=item.split_state,
                            warm_start_alpha=item.warm_start,
                            warm_start_beta=item.warm_start_beta,
                            cfg=cfg,
                            cache=node_cache_by_example.get(example_idx),
                        )
                        if getattr(stats, "feasibility", None) == "infeasible":
                            _consume_live_node(example_idx)
                            continue
                        batch_results[orig_i] = (
                            IntervalState(lower=bounds.lower.detach().clone(), upper=bounds.upper.detach().clone()),
                            best_alpha.detach_clone(),
                            best_beta.detach_clone(),
                            branch,
                        )
                    except ValueError:
                        _decrement_live_node(example_idx)
                        _mark_proven_if_finished(example_idx)

        for i, item in enumerate(items):
            example_idx = int(item.example_idx)
            if i in pruned_indices:
                continue
            if i in cached_results:
                node_bounds, alpha_i, beta_i, branch = cached_results[i]
                _process_evaluated_node(
                    item,
                    bounds=node_bounds,
                    best_alpha=alpha_i,
                    best_beta=beta_i,
                    branch_hint=branch,
                )
                continue
            if i in batch_results:
                node_bounds, alpha_i, beta_i, branch = batch_results[i]
                _process_evaluated_node(
                    item,
                    bounds=node_bounds,
                    best_alpha=alpha_i,
                    best_beta=beta_i,
                    branch_hint=branch,
                )
                continue
            if status_by_example[example_idx] is None:
                _decrement_live_node(example_idx)
                _mark_proven_if_finished(example_idx)

    for example_idx in range(num_examples):
        if status_by_example[example_idx] is None:
            if live_nodes[example_idx] == 0:
                status_by_example[example_idx] = "proven"
            else:
                status_by_example[example_idx] = "unknown"

    per_example = tuple(
        BabPerExampleResult(
            example_idx=example_idx,
            status=status_by_example[example_idx] or "unknown",
            nodes_visited=visited_by_example[example_idx],
            nodes_evaluated=evaluated_by_example[example_idx],
            nodes_expanded=expanded_by_example[example_idx],
            best_lower=best_lower_by_example[example_idx],
            best_upper=best_upper_by_example[example_idx],
        )
        for example_idx in range(num_examples)
    )
    statuses = [r.status for r in per_example]
    best_lower = min(r.best_lower for r in per_example)
    best_upper = max(r.best_upper for r in per_example)
    return BabResult(
        status=_aggregate_bab_status(statuses),
        nodes_visited=sum(visited_by_example),
        nodes_evaluated=sum(evaluated_by_example),
        nodes_expanded=sum(expanded_by_example),
        max_queue=max_q,
        batch_rounds=batch_rounds,
        avg_batch_fill_rate=(
            0.0
            if int(cfg.node_batch_size) <= 1 or batch_rounds == 0
            else float(batch_popped_total) / float(batch_rounds * int(cfg.node_batch_size))
        ),
        best_lower=best_lower,
        best_upper=best_upper,
        per_example=per_example,
    )
