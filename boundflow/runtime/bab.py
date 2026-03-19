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
    Split constraints for chain-structured MLP ReLU nodes.

    Mapping: relu_input_name -> split tensor int{-1,0,+1} with shape [H].
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
        cur_i = int(cur[neuron_idx].item())
        if cur_i != 0 and cur_i != int(split_value):
            raise ValueError(f"conflicting split for {relu_input}[{neuron_idx}]: existing={cur_i} new={split_value}")
        new_by = dict(self.by_relu_input)
        new_t = cur.clone()
        new_t[int(neuron_idx)] = int(split_value)
        new_by[relu_input] = new_t
        return ReluSplitState(by_relu_input=new_by)

    @staticmethod
    def empty(module: BFTaskModule, *, device: torch.device) -> "ReluSplitState":
        module.validate()
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
                raise NotImplementedError(f"ReluSplitState expects rank-2 weights, got {tuple(w_t.shape)}")
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


@dataclass(order=True)
class _QueueItem:
    # heap key: smallest lower bound first (hardest node).
    priority: float
    node_id: int
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


def prune_infeasible_first_layer_items(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    items: List[Tuple[int, _QueueItem]],
    cache: Optional[NodeEvalCache],
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
        if cache is not None:
            v = cache.get(split_state=it.split_state)
            if v is not None and getattr(v.stats, "feasibility", None) == "infeasible":
                pruned.append(orig_i)
                continue
        st = check_first_layer_infeasible_split(module, spec, relu_split_state=it.split_state.by_relu_input)
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
    relu_split = {k: v for k, v in split_state.by_relu_input.items() if v is not None}
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, input_spec_n, relu_split_state=relu_split)
    _ = interval_env
    best = None
    best_gap = None
    for name, pre in relu_pre.items():
        l = pre.lower
        u = pre.upper
        amb = (l < 0) & (u > 0)
        if not amb.any():
            continue
        gap = (u - l).mean(dim=0)  # [H]
        gap = torch.where(amb.any(dim=0), gap, torch.full_like(gap, -1.0))
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
    Minimal BaB loop for chain-structured MLP using alpha-CROWN as a bound oracle.

    - Prunes nodes when lower >= threshold.
    - Declares UNSAFE when a fully split (no ambiguous ReLU) node has lower < threshold.

    This is intended as a correctness-first skeleton; batching/caching comes later.
    """
    cfg = config or BabConfig()
    input_spec_n = _normalize_input_spec(input_spec)
    device = input_spec_n.center.device
    module.validate()
    task = module.get_entry_task()
    if any(op.op_type in {"conv2d", "flatten"} for op in task.ops):
        raise NotImplementedError("BaB conv graphs not yet supported; PR6 only extends alpha-beta-CROWN oracle")

    root_split = ReluSplitState.empty(module, device=device)
    heap: List[_QueueItem] = []
    heapq.heappush(heap, _QueueItem(priority=float("-inf"), node_id=0, split_state=root_split, warm_start=None))
    node_id = 1
    visited = 0
    evaluated = 0
    expanded = 0
    batch_rounds = 0
    batch_popped_total = 0
    max_q = 1
    global_best_lower = float("-inf")
    global_best_upper = float("inf")

    if int(cfg.node_batch_size) < 1:
        raise ValueError(f"node_batch_size must be >=1, got {cfg.node_batch_size}")

    node_cache: Optional[NodeEvalCache] = None
    if (
        cfg.oracle == "alpha_beta"
        and bool(cfg.enable_node_eval_cache)
        and not cfg.use_1d_linf_input_restriction_patch
    ):
        node_cache = NodeEvalCache(module=module, input_spec=input_spec_n, linear_spec_C=linear_spec_C, cfg=cfg)

    while heap and visited < int(cfg.max_nodes):
        # Batched BaB is currently only implemented for the alpha-beta oracle without the 1D patch.
        do_batch = cfg.oracle == "alpha_beta" and int(cfg.node_batch_size) > 1 and not cfg.use_1d_linf_input_restriction_patch
        if not do_batch:
            item = heapq.heappop(heap)
            visited += 1
            restricted_spec = input_spec_n
            if cfg.use_1d_linf_input_restriction_patch:
                restricted_spec = _restrict_input_spec_linf_1d_for_first_layer_splits(
                    module, input_spec_n, split_state=item.split_state
                )
                if restricted_spec is None:
                    continue
            try:
                if cfg.oracle == "alpha_beta":
                    bounds, best_alpha, best_beta, stats, branch_hint, _hit = eval_bab_alpha_beta_node(
                        module,
                        restricted_spec,
                        linear_spec_C=linear_spec_C,
                        split_state=item.split_state,
                        warm_start_alpha=item.warm_start,
                        warm_start_beta=item.warm_start_beta,
                        cfg=cfg,
                        cache=node_cache,
                    )
                    if getattr(stats, "feasibility", None) == "infeasible":
                        continue
                    best_beta = best_beta.detach_clone()
                else:
                    bounds, best_alpha, _stats = run_alpha_crown_mlp(
                        module,
                        restricted_spec,
                        linear_spec_C=linear_spec_C,
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
            except ValueError:
                continue

            evaluated += 1
            node_lower = float(bounds.lower.min().item())
            node_upper = float(bounds.upper.max().item())
            global_best_lower = max(global_best_lower, node_lower)
            global_best_upper = min(global_best_upper, node_upper)

            if node_lower >= float(cfg.threshold) - float(cfg.tol):
                continue

            branch = (
                branch_hint
                if cfg.oracle == "alpha_beta" and bool(cfg.use_branch_hint)
                else None
            )
            if branch is None:
                branch = _pick_branch(module, input_spec_n, split_state=item.split_state)
            if branch is None:
                return BabResult(
                    status="unsafe",
                    nodes_visited=visited,
                    nodes_evaluated=evaluated,
                    nodes_expanded=expanded,
                    max_queue=max_q,
                    batch_rounds=batch_rounds,
                    avg_batch_fill_rate=0.0,
                    best_lower=global_best_lower,
                    best_upper=global_best_upper,
                )

            relu_input, idx = branch
            left = item.split_state.with_split(relu_input=relu_input, neuron_idx=idx, split_value=-1)
            right = item.split_state.with_split(relu_input=relu_input, neuron_idx=idx, split_value=+1)

            heapq.heappush(
                heap,
                _QueueItem(
                    priority=node_lower,
                    node_id=node_id,
                    split_state=left,
                    warm_start=best_alpha,
                    warm_start_beta=best_beta,
                ),
            )
            node_id += 1
            heapq.heappush(
                heap,
                _QueueItem(
                    priority=node_lower,
                    node_id=node_id,
                    split_state=right,
                    warm_start=best_alpha,
                    warm_start_beta=best_beta,
                ),
            )
            node_id += 1
            expanded += 1
            max_q = max(max_q, len(heap))
            continue

        # Batch path: pop top-K nodes and evaluate bounds in one oracle call.
        items: List[_QueueItem] = []
        for _ in range(min(int(cfg.node_batch_size), len(heap), int(cfg.max_nodes) - visited)):
            items.append(heapq.heappop(heap))
        visited += len(items)
        if not items:
            break
        batch_rounds += 1
        batch_popped_total += len(items)

        if input_spec_n.center.dim() != 2:
            raise NotImplementedError("batched BaB currently expects input center rank-2 [B,I]")
        if int(input_spec_n.center.shape[0]) != 1:
            raise NotImplementedError("batched BaB currently expects input batch B==1 (node-batch uses batch dim)")
        x0 = input_spec_n.center

        # PR-3A: reuse cached node evaluations when possible.
        cached_bounds: Dict[int, IntervalState] = {}
        cached_alpha: Dict[int, AlphaState] = {}
        cached_beta: Dict[int, BetaState] = {}
        uncached: List[Tuple[int, _QueueItem]] = []
        pruned_indices: List[int] = []
        if node_cache is not None:
            for i, it in enumerate(items):
                v = node_cache.get(split_state=it.split_state)
                if v is None:
                    uncached.append((i, it))
                else:
                    if getattr(v.stats, "feasibility", None) == "infeasible":
                        pruned_indices.append(i)
                        continue
                    cached_bounds[i] = v.bounds
                    cached_alpha[i] = v.best_alpha
                    cached_beta[i] = v.best_beta
        else:
            uncached = list(enumerate(items))

        uncached, pruned_new = prune_infeasible_first_layer_items(
            module,
            input_spec_n,
            items=uncached,
            cache=node_cache,
            cfg=cfg,
        )
        pruned_indices.extend(pruned_new)

        bounds_b: Optional[IntervalState] = None
        alpha_b: Optional[AlphaState] = None
        beta_b: Optional[BetaState] = None
        stats_var: Any = None
        if uncached:
            center_b = x0.expand(len(uncached), -1).clone()
            batch_spec = type(input_spec_n)(
                value_name=input_spec_n.value_name, center=center_b, perturbation=input_spec_n.perturbation
            )

            C_b = None
            if linear_spec_C is not None:
                C = linear_spec_C
                if not torch.is_tensor(C):
                    C = torch.as_tensor(C, device=device)
                if C.dim() == 2:
                    C = C.unsqueeze(0)
                if C.dim() != 3:
                    raise ValueError(f"linear_spec_C expects rank-2/3, got {tuple(C.shape)}")
                if int(C.shape[0]) != 1:
                    raise NotImplementedError("batched BaB currently expects linear_spec_C batch B==1")
                C_b = C.expand(len(uncached), int(C.shape[1]), int(C.shape[2])).clone()

            # Stack split_state / warm-start alpha,beta along node dimension.
            relu_split_b: Dict[str, torch.Tensor] = {}
            warm_alpha_b: Dict[str, torch.Tensor] = {}
            warm_beta_b: Dict[str, torch.Tensor] = {}
            for relu_input, base_split in items[0].split_state.by_relu_input.items():
                _ = base_split
                relu_split_b[relu_input] = torch.stack(
                    [it.split_state.by_relu_input[relu_input] for _i, it in uncached], dim=0
                )
                alpha_rows: List[torch.Tensor] = []
                beta_rows: List[torch.Tensor] = []
                for _i, it in uncached:
                    if it.warm_start is not None and relu_input in it.warm_start.alpha_by_relu_input:
                        a = it.warm_start.alpha_by_relu_input[relu_input]
                        a_t = a if torch.is_tensor(a) else torch.as_tensor(a, device=device)
                        alpha_rows.append(a_t.to(device=device, dtype=x0.dtype))
                    else:
                        alpha_rows.append(
                            torch.full((int(base_split.shape[0]),), float(cfg.alpha_init), device=device, dtype=x0.dtype)
                        )
                    if it.warm_start_beta is not None and relu_input in it.warm_start_beta.beta_by_relu_input:
                        b = it.warm_start_beta.beta_by_relu_input[relu_input]
                        b_t = b if torch.is_tensor(b) else torch.as_tensor(b, device=device)
                        beta_rows.append(b_t.to(device=device, dtype=x0.dtype))
                    else:
                        beta_rows.append(
                            torch.full((int(base_split.shape[0]),), float(cfg.beta_init), device=device, dtype=x0.dtype)
                        )
                warm_alpha_b[relu_input] = torch.stack(alpha_rows, dim=0)
                warm_beta_b[relu_input] = torch.stack(beta_rows, dim=0)

            warm_alpha_state = AlphaState(alpha_by_relu_input=warm_alpha_b)
            warm_beta_state = BetaState(beta_by_relu_input=warm_beta_b)

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
                    warm_start_alpha=warm_alpha_state,
                    warm_start_beta=warm_beta_state,
                    objective=cfg.objective,
                    spec_reduce=cfg.spec_reduce,
                    soft_tau=float(cfg.soft_tau),
                    lb_weight=float(cfg.lb_weight),
                    ub_weight=float(cfg.ub_weight),
                    per_batch_params=True,
                )
                if getattr(stats_var, "feasibility", None) == "infeasible":
                    continue
            except ValueError:
                bounds_b = None
                alpha_b = None
                beta_b = None
                for i, it in uncached:
                    try:
                        b1, a1, beta1, st1, branch1, _hit = eval_bab_alpha_beta_node(
                            module,
                            input_spec_n,
                            linear_spec_C=linear_spec_C,
                            split_state=it.split_state,
                            warm_start_alpha=it.warm_start,
                            warm_start_beta=it.warm_start_beta,
                            cfg=cfg,
                            cache=node_cache,
                        )
                        if getattr(st1, "feasibility", None) == "infeasible":
                            continue
                        cached_bounds[i] = IntervalState(
                            lower=b1.lower.detach().clone(), upper=b1.upper.detach().clone()
                        )
                        cached_alpha[i] = a1.detach_clone()
                        cached_beta[i] = beta1.detach_clone()
                        if node_cache is not None:
                            node_cache.put(
                                split_state=it.split_state,
                                value=NodeEvalCacheValue(
                                    bounds=cached_bounds[i],
                                    best_alpha=cached_alpha[i],
                                    best_beta=cached_beta[i],
                                    stats=st1,
                                    branch=branch1,
                                ),
                            )
                    except ValueError:
                        continue

        # Process each node's result.
        for i, it in enumerate(items):
            if i in pruned_indices:
                continue
            if i in cached_bounds:
                node_bounds = cached_bounds[i]
                alpha_i = cached_alpha[i]
                beta_i = cached_beta[i]
                branch = None
                if node_cache is not None:
                    cv = node_cache.get(split_state=it.split_state)
                    if cv is not None:
                        branch = cv.branch
            else:
                if bounds_b is None or alpha_b is None or beta_b is None:
                    # No evaluation result for this node (e.g. per-node fallback also failed); skip it.
                    continue
                # Locate position in uncached list.
                j = [jj for jj, (orig_i, _it) in enumerate(uncached) if orig_i == i]
                if not j:
                    raise RuntimeError("internal error: missing node eval result")
                jj = int(j[0])
                node_bounds = IntervalState(lower=bounds_b.lower[jj : jj + 1], upper=bounds_b.upper[jj : jj + 1])
                alpha_i = AlphaState({k: v[jj].detach().clone() for k, v in alpha_b.alpha_by_relu_input.items()})
                beta_i = BetaState({k: v[jj].detach().clone() for k, v in beta_b.beta_by_relu_input.items()})
                if node_cache is not None:
                    branch = None
                    if hasattr(stats_var, "branch_choices") and stats_var.branch_choices:
                        try:
                            branch = stats_var.branch_choices[jj]
                        except Exception:
                            branch = None
                    node_cache.put(
                        split_state=it.split_state,
                        value=NodeEvalCacheValue(
                            bounds=IntervalState(
                                lower=node_bounds.lower.detach().clone(),
                                upper=node_bounds.upper.detach().clone(),
                            ),
                            best_alpha=alpha_i.detach_clone(),
                            best_beta=beta_i.detach_clone(),
                            stats=stats_var,
                            branch=branch,
                        ),
                    )
                else:
                    branch = None
            node_lower = float(node_bounds.lower.min().item())
            node_upper = float(node_bounds.upper.max().item())
            global_best_lower = max(global_best_lower, node_lower)
            global_best_upper = min(global_best_upper, node_upper)

            evaluated += 1
            if node_lower >= float(cfg.threshold) - float(cfg.tol):
                continue

            if not bool(cfg.use_branch_hint):
                branch = None
            if branch is None:
                branch = _pick_branch(module, input_spec_n, split_state=it.split_state)
            if branch is None:
                return BabResult(
                    status="unsafe",
                    nodes_visited=visited,
                    nodes_evaluated=evaluated,
                    nodes_expanded=expanded,
                    max_queue=max_q,
                    batch_rounds=batch_rounds,
                    avg_batch_fill_rate=0.0 if int(cfg.node_batch_size) <= 1 else float(batch_popped_total) / float(batch_rounds * int(cfg.node_batch_size)),
                    best_lower=global_best_lower,
                    best_upper=global_best_upper,
                )

            relu_input, idx = branch
            left = it.split_state.with_split(relu_input=relu_input, neuron_idx=idx, split_value=-1)
            right = it.split_state.with_split(relu_input=relu_input, neuron_idx=idx, split_value=+1)

            heapq.heappush(
                heap,
                _QueueItem(
                    priority=node_lower,
                    node_id=node_id,
                    split_state=left,
                    warm_start=alpha_i,
                    warm_start_beta=beta_i.detach_clone(),
                ),
            )
            node_id += 1
            heapq.heappush(
                heap,
                _QueueItem(
                    priority=node_lower,
                    node_id=node_id,
                    split_state=right,
                    warm_start=alpha_i,
                    warm_start_beta=beta_i.detach_clone(),
                ),
            )
            node_id += 1
            expanded += 1
            max_q = max(max_q, len(heap))

    if not heap:
        return BabResult(
            status="proven",
            nodes_visited=visited,
            nodes_evaluated=evaluated,
            nodes_expanded=expanded,
            max_queue=max_q,
            batch_rounds=batch_rounds,
            avg_batch_fill_rate=0.0 if int(cfg.node_batch_size) <= 1 or batch_rounds == 0 else float(batch_popped_total) / float(batch_rounds * int(cfg.node_batch_size)),
            best_lower=global_best_lower,
            best_upper=global_best_upper,
        )
    return BabResult(
        status="unknown",
        nodes_visited=visited,
        nodes_evaluated=evaluated,
        nodes_expanded=expanded,
        max_queue=max_q,
        batch_rounds=batch_rounds,
        avg_batch_fill_rate=0.0 if int(cfg.node_batch_size) <= 1 or batch_rounds == 0 else float(batch_popped_total) / float(batch_rounds * int(cfg.node_batch_size)),
        best_lower=global_best_lower,
        best_upper=global_best_upper,
    )
