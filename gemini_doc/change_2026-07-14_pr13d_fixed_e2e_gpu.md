# 2026-07-14：PR-13D Fixed-Stream 与 Same-Solver GPU E2E

## 阶段判定

PR-13D 已完成一组 **reduced chain-CNN** 的固定查询流和真实 host-side BaB 双层评估。结果证明
Query Runtime 能把现有 solver 选出的兼容节点交给物理 αβ batch executor，并在不改变 solver
状态、节点数和最终判定的前提下保留普通 batching 的收益；它没有证明 runtime 相比已有
`batched original` 还有额外算法级加速。

```text
fixed replay correctness:       PASS（16/16，0 mismatch）
same-solver status/node count:  PASS
custom CUDA stream:             PASS
dispatch-plan cache:            PASS（可观测）
PR-12 compiled Planner:         N/A（αβ/split capability 不兼容）
non-toy/VNN-COMP:               NOT DONE
PR-13D:                         VALIDATED-REDUCED
```

## 权威配置

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU；
- PyTorch：2.12.1+cu132；Python 3.12.13；
- workload：两层 3×3 Conv + ReLU、144 个 ReLU、6×6 输入；
- hard case：16 个实际 BaB node；另含 root-proven 和 root-unsafe sanity；
- αβ step=1，batch=8，warmup=1，独立重复 5 次；
- 代码基线：`fda5b82`；工件：`artifacts/phase7a-pr13/pr13d-bab-runtime-v5-20260714/`。

## Fixed-stream 结果

| Variant | p50 / p90 | Throughput | Peak delta |
|---|---:|---:|---:|
| per-node original | 1892.94 / 1921.28 ms | 8.45 query/s | 94,720 B |
| batched original | 20.09 / 20.72 ms | 796.43 query/s | 202,240 B |
| BoundFlow runtime dense | 19.61 / 19.87 ms | 815.84 query/s | 202,240 B |

BoundFlow runtime 相对逐节点为 **96.52×**，相对 batched original 为 **1.024×**。16 个 query
的 bounds、branch 和 α/β state 在数值容差内全部一致；精确 state content hash 有浮点归约顺序
导致的差异时单独记录，不用 hash 相等替代数值正确性。

## True E2E 结果

| Workload | per-node | batched original | runtime | Runtime / per-node | Runtime / batched |
|---|---:|---:|---:|---:|---:|
| hard, 16 nodes, unknown | 1906.00 ms | 188.06 ms | 191.86 ms | 9.93× | 0.980× |
| safe root, proven | 9.64 ms | 9.22 ms | 10.09 ms | 0.955× | 0.914× |
| unsafe root, unsafe | 4.28 ms | 4.26 ms | 4.98 ms | 0.859× | 0.855× |

三种 variant 的最终 status 和 nodes evaluated 完全一致。hard case 中 runtime 相对 batched
original 慢约 2.0%，说明 headline 收益来自 batching；单节点 easy case 的 runtime 固定开销可达
约 14%。论文不得把这部分写成“Query abstraction 本身带来 9.93×”。

## Runtime 机制证据

- hard E2E：16 queries、5 physical batches、平均 batch 3.2、0 loss/duplicate/invalid；
- queue wait p50/p90/p99：18/39/57 μs；
- dispatch-plan cache：1 miss、4 hits、1 entry；
- `compiled_plan_cache_applicable=false`、`pr12_planner_dispatches=0`：αβ/split 查询安全回退
  dense，不伪造 PR-12 Planner coverage；
- GPU payload/state content hash 从热路径移除，hash 只用于离线 recorder；
- 自定义 Torch CUDA stream 测试只同步该 stream event，executor 观察到的 stream ID 与调用
  stream 一致。

## 限制

- chain-CNN、16 nodes，仍不是 mini-ResNet/VNN-COMP non-toy；
- OOM 只有 deterministic fault-injection 拆批证据，没有真实 CUDA OOM；
- memory budget 使用动态 payload byte estimate，不是完整 allocator peak model；
- 没有 branch/prune/GPU-active-time 分解；
- αβ/split 不兼容 PR-12 plain-CROWN fused TIR，因而没有多后端 Planner 或 compiled cache
  消融；
- unsafe workload 不产生显式 counterexample，故 primal counterexample replay 为 N/A。

这些限制决定 PR-13 只能进入 `VALIDATED-REDUCED` closure，不能标记 full `VALIDATED`。
