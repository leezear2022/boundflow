# Phase 7A PR-11：shared CROWN benchmark 摘要

**日期**: 2026-04-08  
**环境**: `conda run -n boundflow` / Torch 2.5.1 / CUDA / NVIDIA GeForce RTX 4090  
**脚本**: `scripts/bench_phase7a_shared_crown_path_attribution.py`  
**提交基线**: `62cc086`

---

## 目的

PR-11 不继续扩 operator algebra，而是先把 PR-10 的两条 shared CROWN 主线收益拆开量化：

- structured ReLU backward
- layout-only `permute/reshape` backward

对照方式不是历史 git commit 对比，而是在同一进程内用 patch 强制回退到 dense baseline：

- `relu_barrier`: 把 `_backprop_relu_step(...)` 换成旧的 dense reference
- `layout_only`: 把 `_backprop_permute_step(...)` 换成 dense materialize

---

## 命令

```bash
conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cuda \
  --profile bench \
  --workloads all \
  --warmup 5 \
  --iters 20
```

---

## 结果

| workload | compare_target | structured ms p50 | baseline ms p50 | speedup |
|---|---|---:|---:|---:|
| `relu_heavy_mlp` | `relu_barrier` | 3.936 | 2.610 | 0.66x |
| `residual_relu_mlp` | `relu_barrier` | 2.406 | 1.574 | 0.65x |
| `concat_relu_mlp` | `relu_barrier` | 3.457 | 2.246 | 0.65x |
| `permute_reshape_linear` | `layout_only` | 1.320 | 1.471 | 1.11x |

---

## 观测

1. `layout_only` 这条线已经出现正收益。
   `permute_reshape_linear` 上结构化 `ReindexInputLinearOperator` 比 dense layout barrier 快约 `1.11x`，说明 batch-preserving `permute`/`reshape` 的 shared CROWN 路径已经有了最小性能回报。

2. structured ReLU backward 还没有 latency 正收益。
   三个 `relu_barrier` workload 都落在 `0.65x~0.66x`，当前结构化 ReLU 路径比 dense reference 更慢。这个结果和 PR-11 的定位一致：先量化，不预设一定加速。

3. 剩余 dense fallback 热点已经被定位出来。
   `relu_heavy_mlp` 与 `residual_relu_mlp` 的 `split_pos_neg_dense_by_op` 全部落在 `RightMatmulLinearOperator`。
   `concat_relu_mlp` 则是 `RightMatmulLinearOperator` + `SliceInputLinearOperator`。

4. 现阶段最明确的后续优化目标不是再改 benchmark，而是先消热路径 dense 点。
   PR-12 应优先处理 `RightMatmulLinearOperator.split_pos_neg()`，再用同一脚本复跑；如果需要第二优先级，再看 `SliceInputLinearOperator` 在 concat 路径上的 dense fallback。

---

## 结论

PR-11 给出的不是“structured path 一定更快”，而是更有价值的两条事实：

- layout-only shared CROWN 已经开始兑现小幅性能收益；
- structured ReLU backward 的主要性能瓶颈已经从“看不见的 dense barrier”收敛成了可点名的 operator hotspot，首先就是 `RightMatmulLinearOperator.split_pos_neg()`。

---

## PR-12 复跑说明（2026-04-09）

PR-12 本轮只落地了一个安全改动：把 `SliceInputLinearOperator.split_pos_neg()` 从 dense fallback 改成结构化传递。

没有继续做 `RightMatmulLinearOperator.split_pos_neg()`，原因是当前 `split_pos_neg()` 的 contract 要求**逐元素精确**对齐 `to_dense().clamp_min(0)` / `clamp_max(0)`；而 `A @ rhs` 的常见四项拆分只能保证 `pos + neg == original`，不能保证逐元素正负部精确一致，已经被 PR-10 的 ReLU 对齐测试证伪。

### PR-12 CUDA 复跑

命令保持不变：

```bash
conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cuda \
  --profile bench \
  --workloads all \
  --warmup 5 \
  --iters 20
```

| workload | PR-11 speedup | PR-12 speedup | 变化 |
|---|---:|---:|---:|
| `relu_heavy_mlp` | `0.66x` | `0.66x` | 基本持平 |
| `residual_relu_mlp` | `0.65x` | `0.66x` | 小幅回升 |
| `concat_relu_mlp` | `0.65x` | `0.51x` | 明显变差 |
| `permute_reshape_linear` | `1.11x` | `1.11x` | 持平 |

### PR-12 计数变化

- `relu_heavy_mlp`：`split_pos_neg_dense_by_op = {"RightMatmulLinearOperator": 8}`
- `residual_relu_mlp`：`split_pos_neg_dense_by_op = {"RightMatmulLinearOperator": 4}`
- `concat_relu_mlp`：`split_pos_neg_dense_by_op = {"RightMatmulLinearOperator": 6}`

`SliceInputLinearOperator` 已经从计数里消失，说明 concat 路径上的 slice dense barrier 已被清掉；但这并没有转化为稳定的 latency 收益，尤其 `concat_relu_mlp` 还出现了明显回退。

### 更新后的结论

- `SliceInputLinearOperator` 不再是 shared CROWN ReLU 路径的剩余 dense 热点。
- `RightMatmulLinearOperator` 现在是唯一明确、稳定复现的 `split_pos_neg_dense` 热点。
- PR-12 之后，下一步不应该继续做“看起来像是 exact 的 `RightMatmul` 四项拆分”，而应该先明确新的 operator 表示、证明条件，或者接受更弱但显式标注的 over-approximation contract。

---

## PR-13 接口重构说明（2026-04-09）

PR-13 没有尝试优化 `RightMatmulLinearOperator` 本身，而是把 ReLU backward 从“直接依赖 `split_pos_neg()`”改成“调用专用的 `relu_relax_pullback()` 接口”。

这轮的目标只有一个：把 ReLU caller 和 exact sign split contract 解耦，给后续 `RightMatmul.relu_relax_pullback()` 的单独实现腾接口位置。当前默认实现仍然通过旧的 split-based 公式回退，所以按设计不会改善 `RightMatmul` 的 dense fallback。

### PR-13 CUDA 复跑

| workload | PR-12 speedup | PR-13 speedup |
|---|---:|---:|
| `relu_heavy_mlp` | `0.66x` | `0.54x` |
| `residual_relu_mlp` | `0.66x` | `0.57x` |
| `concat_relu_mlp` | `0.51x` | `0.46x` |
| `permute_reshape_linear` | `1.11x` | `1.12x` |

### PR-13 计数变化

- `relu_heavy_mlp`：`split_pos_neg_dense_by_op = {"RightMatmulLinearOperator": 8}`
- `residual_relu_mlp`：`split_pos_neg_dense_by_op = {"RightMatmulLinearOperator": 4}`
- `concat_relu_mlp`：`split_pos_neg_dense_by_op = {"RightMatmulLinearOperator": 6}`

### PR-13 结论

- `split_pos_neg_dense` 的剩余热点没有变化，仍然只有 `RightMatmulLinearOperator`。
- stdout JSON schema、workload 集合和计数键保持不变。
- 这轮是纯接口重构，不应拿性能数字判断成败；真正的性能工作要放到后续 `RightMatmul.relu_relax_pullback()` 的 sound 实现里。

---

## PR-14 复跑说明（2026-04-10）

PR-14 没有改变 `RightMatmulLinearOperator.split_pos_neg()` 的 exact contract，而是在 `relu_relax_pullback()` 上做了专用实现。

这版实现的关键点是：

- 不再通过 `self.split_pos_neg()` 进入 `_split_pos_neg_dense(self)`
- 直接在 `RightMatmul.relu_relax_pullback()` 内 materialize 当前 dense 系数
- 精确构造 ReLU pullback 后的 `A_out` 和 `delta_b`
- 对 `concat` 路径再补一层 `SliceInputLinearOperator.relu_relax_pullback()` 委托，避免 slice 包装层把调用重新打回默认 split helper

这不是“`RightMatmul` 已经支持 exact structured sign split”。它只是把 ReLU 专用路径从旧的 split hotspot 上挪开，并保留 exact dense reference 行为。

### PR-14 CUDA 复跑

| workload | PR-13 speedup | PR-14 speedup | 变化 |
|---|---:|---:|---:|
| `relu_heavy_mlp` | `0.54x` | `0.65x` | 明显回升 |
| `residual_relu_mlp` | `0.57x` | `0.68x` | 明显回升 |
| `concat_relu_mlp` | `0.46x` | `0.53x` | 明显回升 |
| `permute_reshape_linear` | `1.12x` | `1.11x` | 基本持平 |

### PR-14 计数变化

- `relu_heavy_mlp`：`split_pos_neg_dense_by_op = {}`
- `residual_relu_mlp`：`split_pos_neg_dense_by_op = {}`
- `concat_relu_mlp`：`split_pos_neg_dense_by_op = {}`

### 更新后的结论

- ReLU workload 上的 `split_pos_neg_dense` 热点已经清零。
- PR-14 的收益来自 `relu_relax_pullback()` 专用实现绕开旧 split 路径，不代表 `RightMatmul` 获得了通用的 structured sign split。
- 三个 ReLU workload 都比 PR-13 回升，但仍低于 dense ReLU barrier；下一步如果继续做，应优先评估如何减少 `relu_relax_pullback()` 内部的 dense materialization 成本，而不是再回头改 `split_pos_neg()` contract。
