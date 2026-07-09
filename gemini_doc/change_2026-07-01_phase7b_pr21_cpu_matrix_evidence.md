# Phase 7B PR-21：Formal CPU Matrix Evidence

**日期**: 2026-07-01

## 背景

PR-19 建立了 `phase7b_crossover_matrix.v1`，PR-20 建立了 `phase7b_cost_model_v1` 后处理。PR-21 用正式 CPU 参数跑完整 matrix：

```text
workloads = all
scales = smoke,small,bench
policies = structured,dense_barrier,auto
warmup = 5
iters = 20
device = cpu
```

CUDA 在当前机器不可用：

```text
torch.cuda.is_available() == False
```

因此本轮只形成 CPU evidence，不声明 CUDA 结论。

## 命令

```bash
mkdir -p out/phase7b

conda run --no-capture-output -n boundflow python scripts/bench_phase7b_crossover_matrix.py \
  --device cpu \
  --workloads all \
  --scales smoke,small,bench \
  --policies structured,dense_barrier,auto \
  --warmup 5 \
  --iters 20 \
  > out/phase7b/phase7b_pr22_cpu_matrix.json

conda run --no-capture-output -n boundflow python scripts/postprocess_phase7b_cost_model.py \
  out/phase7b/phase7b_pr22_cpu_matrix.json \
  --min-relative-margin 0.05 \
  > out/phase7b/phase7b_pr22_cpu_cost_model.json
```

## 结果摘要

`phase7b_cost_model_v1` 显示：

| Workload | Scale | Recommended final policy | Confidence | Gap |
|---|---:|---|---|---:|
| `permute_reshape_linear` | `bench` | `structured` | high | 0.700 |
| `permute_reshape_linear` | `small` | `structured` | high | 0.207 |
| `permute_reshape_linear` | `smoke` | `dense_barrier` | medium | 0.095 |
| `concat_relu_mlp` | `small` | `dense_barrier` | medium | 0.061 |
| `concat_relu_mlp` | `smoke` | `dense_barrier` | medium | 0.055 |
| `residual_relu_mlp` | `small` | `dense_barrier` | medium | 0.067 |
| `residual_relu_mlp` | `smoke` | `dense_barrier` | medium | 0.074 |

其余 ReLU-heavy rules 多为 low confidence，尤其 `relu_heavy_mlp` 在三个 scale 上 gap 都很小。

所有 rule 的 guardrails 均满足：

```text
unknown_materialization_calls == 0
split_pos_neg_dense_total == 0
```

## 解释

PR-18 的旧 auto 规则曾经是：

```text
layout_only -> dense_barrier
relu_barrier -> structured
```

PR-21 的 CPU evidence 说明这个规则过粗：

- `permute_reshape_linear smoke` 上 `dense_barrier` 仍可作为 medium-confidence 选择。
- `permute_reshape_linear small/bench` 上 `structured` 明显更好，并达到 high confidence。
- ReLU workloads 上没有足够稳定的 high-confidence 规则，不能把 dense barrier 或 structured 继续硬编码推进。

## 结论

PR-21 只证明 CPU 上两个 high-confidence promotion：

```text
cpu + permute_reshape_linear + small -> structured
cpu + permute_reshape_linear + bench -> structured
```

这些规则适合进入 planner v2。其他 workload/scale 继续留在 evidence report，不改变 runtime 默认策略。
